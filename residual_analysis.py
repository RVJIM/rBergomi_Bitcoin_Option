"""
residual_analysis.py
====================
Statistical analysis of calibration residuals for the rBergomi model.

For each calibration date and method:
  1. Loads calibrated parameters (H, eta, rho, xi0 curve) from the saved CSVs.
  2. Re-runs the pricer to obtain per-point model IVs.
  3. Collects residuals = model_IV - market_IV  (in percentage points).

Then tests:
  - One-sample t-test:          H0: mean(residuals) = 0
  - Wilcoxon signed-rank test:  H0: median(residuals) = 0  (non-parametric)

Also reports per-date bias_pp for a quick sanity check.

Usage:
    py residual_analysis.py                        # all three methods
    py residual_analysis.py --method hybrid_euler  # one method only
    py residual_analysis.py --method hybrid_euler --method hybrid_mixed
"""

import sys, argparse, logging
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime
from scipy import stats

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("Results/residual_analysis.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ResidualAnalysis")

# ── Project imports ──────────────────────────────────────────────────────────
from iv_surface_builder import IVSurfaceBuilder
from src.rbergomi.calibrator import Calibrator, ForwardVarianceCurve
from src.rbergomi.utils import implied_vol_batch
from src.rbergomi.pricer import rBergomiPricer
from src.config.surfaces import SURFACE_CFG
from src.config.methods import METHODS

DATA_DIR   = Path("data/option")
TABLES_DIR = Path("Results/calibration/tables")
OUTPUT_DIR = Path("Results/calibration/residuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_METHODS = ["cholesky_euler", "hybrid_euler", "hybrid_mixed"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_calibration_csv(method: str) -> pl.DataFrame:
    """Load and concatenate all per-date CSVs for a given method."""
    method_dir = TABLES_DIR / method
    if not method_dir.exists():
        raise FileNotFoundError(f"No calibration results found at {method_dir}")
    frames = []
    for csv_path in sorted(method_dir.glob(f"calibration_*_{method}.csv")):
        frames.append(pl.read_csv(csv_path))
    if not frames:
        raise FileNotFoundError(f"No CSV files in {method_dir}")
    return pl.concat(frames)


def build_xi0_curve(market_data: dict, calibrator: Calibrator) -> ForwardVarianceCurve:
    """Reconstruct the forward variance curve from market data (ATM IVs)."""
    # Use the same internal method as Calibrator.calibrate()
    calibrator._xi0_curve = None  # reset
    xi0_curve, _ = calibrator._estimate_forward_variance_curve(market_data)
    return xi0_curve


def compute_residuals(
    H: float, eta: float, rho: float,
    xi0_curve: ForwardVarianceCurve,
    market_data: dict,
    method_cfg: dict,
) -> np.ndarray:
    """
    Re-price the surface with given params and return per-point residuals
    (model_IV - market_IV) in percentage points.
    """
    pricer = rBergomiPricer(
        H=H, eta=eta, rho=rho, xi=xi0_curve,
        n_paths=method_cfg.get("n_paths", 8192),
        n_steps=method_cfg.get("n_steps", 365),
        scheme=method_cfg.get("scheme", "hybrid"),
        kappa=method_cfg.get("kappa", 0.0),
        seed=method_cfg.get("seed", 42),
        antithetic=method_cfg.get("antithetic", True),
    )

    S0      = market_data["S"]
    K_arr   = market_data["K"]
    tau_arr = market_data["tau"]
    iv_mkt  = market_data["iv_market"]

    model_prices = np.full(len(K_arr), np.nan)
    for T in np.unique(tau_arr):
        mask    = tau_arr == T
        K_group = K_arr[mask]
        indices = np.where(mask)[0]
        try:
            prices, _ = pricer.price_multiple_strikes(S0, K_group, T, option_type='put')
            model_prices[indices] = prices
        except Exception:
            for j, idx in enumerate(indices):
                try:
                    price, _ = pricer.price_inverse_put(S0, K_group[j], T)
                    model_prices[idx] = price
                except Exception:
                    continue

    model_ivs = implied_vol_batch(
        model_prices, S0, K_arr, tau_arr, option_type='put', r=0.0
    )
    valid = ~np.isnan(model_ivs)
    errors_pp = (model_ivs[valid] - iv_mkt[valid]) * 100
    return errors_pp


def analyse_method(method: str, builder: IVSurfaceBuilder) -> dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"  METHOD: {method}")
    logger.info(f"{'='*60}")

    df = load_calibration_csv(method)
    logger.info(f"  Found {len(df)} calibration dates")

    # Load method config for pricer settings (n_paths, n_steps, scheme, ...)
    # Strip non-Calibrator keys
    raw_cfg   = METHODS[method].copy()
    raw_cfg.pop("label", None)
    raw_cfg.pop("maxiter", None)
    pricer_cfg = {k: raw_cfg[k] for k in
                  ("n_paths", "n_steps", "scheme", "kappa", "seed",
                   "antithetic", "pricing_method")
                  if k in raw_cfg}

    calibrator = Calibrator(**raw_cfg)

    all_residuals  = []
    per_date_bias  = []

    for row in df.iter_rows(named=True):
        date_str = str(row["date"])[:10]  # "YYYY-MM-DD"
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"  Cannot parse date '{date_str}' — skipping")
                continue

        H, eta, rho = float(row["H"]), float(row["eta"]), float(row["rho"])
        logger.info(f"  {date_str[:10]}  H={H:.4f}  eta={eta:.3f}  rho={rho:.3f}")

        try:
            surface     = builder.get_iv_surface(target_time=dt, **SURFACE_CFG)
            market_data = builder.export_for_rbergomi(surface)
            xi0_curve   = build_xi0_curve(market_data, calibrator)

            residuals = compute_residuals(H, eta, rho, xi0_curve, market_data, pricer_cfg)
            bias_pp   = float(np.mean(residuals))

            all_residuals.append(residuals)
            per_date_bias.append({"date": date_str[:10], "bias_pp": bias_pp,
                                  "n": len(residuals)})
            logger.info(f"    -> {len(residuals)} points  bias={bias_pp:+.2f} pp")

        except Exception as e:
            logger.warning(f"    FAILED: {e}")
            continue

    if not all_residuals:
        logger.error("  No residuals collected — skipping statistical tests.")
        return {}

    pooled = np.concatenate(all_residuals)
    n_total = len(pooled)
    mean_r  = float(np.mean(pooled))
    std_r   = float(np.std(pooled, ddof=1))

    # ── Statistical tests ────────────────────────────────────────────────────
    t_stat, t_pval = stats.ttest_1samp(pooled, popmean=0.0)
    w_stat, w_pval = stats.wilcoxon(pooled, alternative='two-sided')

    logger.info(f"\n  ─── Pooled residual statistics ({method}) ───")
    logger.info(f"  N total points  : {n_total}")
    logger.info(f"  Mean (bias)     : {mean_r:+.4f} pp")
    logger.info(f"  Std dev         : {std_r:.4f} pp")
    logger.info(f"  t-test  (H0: mu=0): t={t_stat:.3f}  p={t_pval:.4f}")
    logger.info(f"  Wilcoxon        : W={w_stat:.1f}     p={w_pval:.4f}")
    if t_pval < 0.05:
        logger.info(f"  => REJECTED (alpha=5%): residuals NOT centered at zero")
    else:
        logger.info(f"  => NOT REJECTED (alpha=5%): no evidence of non-zero mean")

    # ── Save per-date bias CSV ────────────────────────────────────────────────
    bias_df = pl.DataFrame(per_date_bias)
    bias_path = OUTPUT_DIR / f"bias_per_date_{method}.csv"
    bias_df.write_csv(bias_path)
    logger.info(f"\n  Per-date bias saved to {bias_path}")

    return {
        "method": method,
        "n_total": n_total,
        "mean_pp": mean_r,
        "std_pp": std_r,
        "t_stat": float(t_stat),
        "t_pval": float(t_pval),
        "w_stat": float(w_stat),
        "w_pval": float(w_pval),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual statistical analysis")
    parser.add_argument("--method", action="append", dest="methods",
                        choices=ALL_METHODS,
                        help="Method(s) to analyse (default: all)")
    args = parser.parse_args()
    methods = args.methods or ALL_METHODS

    builder = IVSurfaceBuilder(str(DATA_DIR))
    builder.build_index()

    summary = []
    for m in methods:
        res = analyse_method(m, builder)
        if res:
            summary.append(res)

    if summary:
        logger.info("\n" + "="*60)
        logger.info("  SUMMARY TABLE")
        logger.info("="*60)
        logger.info(f"  {'Method':<20} {'N':>6} {'Bias(pp)':>10} {'t-pval':>10} {'W-pval':>10}")
        logger.info(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
        for r in summary:
            logger.info(
                f"  {r['method']:<20} {r['n_total']:>6} "
                f"{r['mean_pp']:>+10.3f} {r['t_pval']:>10.4f} {r['w_pval']:>10.4f}"
            )

        # Save summary CSV
        sum_df = pl.DataFrame(summary)
        sum_path = OUTPUT_DIR / "residual_test_summary.csv"
        sum_df.write_csv(sum_path)
        logger.info(f"\n  Summary saved to {sum_path}")
