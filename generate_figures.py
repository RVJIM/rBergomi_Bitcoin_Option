"""
generate_figures.py
===================
Generates Chapter 3 publication-quality figures from calibration CSV results.

For each (date, method) combination found in Results/calibration/tables/:
    1. Re-extracts the IV surface from the Deribit data
    2. Reconstructs the forward-variance curve xi0 from ATM IVs
    3. Runs the rBergomi pricer at the calibrated (H, eta, rho) with high path count
    4. Saves four figure types:
         skew_{date}_{method}.png              smile fit by maturity
         residuals_heatmap_{date}_{method}.png moneyness x maturity error heatmap
         residuals_hist_{date}_{method}.png    histogram of residuals
         iv_surface_3d_{date}_{method}.png     3-D implied vol surface

Output:  Results/calibration/figures/{method}/

Usage:
    python generate_figures.py                          # all methods, all dates
    python generate_figures.py --method hybrid_mixed    # one method only
    python generate_figures.py --date 20220509          # one date only
    python generate_figures.py --method hybrid_mixed --date 20220509
"""

import sys
import logging
import argparse
import warnings
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Results/figures_generation.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("GenerateFigures")

# ── Project imports ───────────────────────────────────────────────────────────
from iv_surface_builder import IVSurfaceBuilder
from src.rbergomi.pricer import rBergomiPricer
from src.rbergomi.calibrator import ForwardVarianceCurve
from src.rbergomi.utils import implied_vol_batch

# ── Shared config (single source of truth) ────────────────────────────────────
from src.config.surfaces import SURFACE_CFG
from src.config.methods import METHOD_CFG, METHOD_LABELS, METHOD_COLORS
from src.config.plot_style import apply_thesis_style, PALETTE, SURFACE_CMAP

apply_thesis_style()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/option")
TABLES_DIR  = Path("Results/calibration/tables")
FIGURES_DIR = Path("Results/calibration/figures")


# =============================================================================
#  FORWARD-VARIANCE RECONSTRUCTION
# =============================================================================

def estimate_xi0(market_data: Dict) -> ForwardVarianceCurve:
    """Reconstruct xi0 curve from ATM market IVs (xi0(T) = sigma_ATM(T)^2)."""
    S0       = market_data["S"]
    K_arr    = market_data["K"]
    tau_arr  = market_data["tau"]
    iv_mkt   = market_data["iv_market"]
    moneyness = market_data.get("moneyness", K_arr / S0)

    maturities, xi_vals = [], []
    for T in sorted(np.unique(tau_arr)):
        mask = tau_arr == T
        mon  = moneyness[mask]
        iv   = iv_mkt[mask]
        atm_idx = np.argmin(np.abs(mon - 1.0))
        atm_iv  = float(iv[atm_idx])
        if atm_iv > 0:
            maturities.append(T)
            xi_vals.append(atm_iv ** 2)

    if not maturities:
        atm_idx = np.argmin(np.abs(moneyness - 1.0))
        atm_iv  = float(iv_mkt[atm_idx])
        maturities = [float(tau_arr.min())]
        xi_vals    = [atm_iv ** 2]

    return ForwardVarianceCurve(np.array(maturities), np.array(xi_vals))


# =============================================================================
#  MODEL IV COMPUTATION
# =============================================================================

def compute_model_ivs(
    H: float, eta: float, rho: float,
    xi0: ForwardVarianceCurve,
    market_data: Dict,
    method: str,
) -> np.ndarray:
    """Run the pricer at calibrated params and return model IVs array."""
    cfg = METHOD_CFG[method]
    pricer = rBergomiPricer(
        H=H, eta=eta, rho=rho, xi=xi0,
        n_paths=cfg["n_paths"],
        n_steps=cfg["n_steps"],
        scheme=cfg["scheme"],
        kappa=cfg["kappa"],
        seed=cfg["seed"],
        antithetic=cfg["antithetic"],
        pricing_method=cfg["pricing_method"],
    )

    S0      = market_data["S"]
    K_arr   = market_data["K"]
    tau_arr = market_data["tau"]
    n_pts   = len(K_arr)

    model_prices = np.full(n_pts, np.nan)
    for T in np.unique(tau_arr):
        mask    = tau_arr == T
        K_group = K_arr[mask]
        indices = np.where(mask)[0]
        try:
            prices, _ = pricer.price_multiple_strikes(S0, K_group, T, option_type="put")
            model_prices[indices] = prices
        except Exception as e:
            logger.warning(f"  Batch pricing failed T={T:.4f}: {e}")
            for j, idx in enumerate(indices):
                try:
                    price, _ = pricer.price_inverse_put(S0, K_group[j], T)
                    model_prices[idx] = price
                except Exception:
                    pass

    model_ivs = implied_vol_batch(
        model_prices, S0, K_arr, tau_arr, option_type="put", r=0.0
    )
    return model_ivs


# =============================================================================
#  FIGURE GENERATORS
# =============================================================================

def _method_suffix(method: str) -> str:
    return method  # e.g. "hybrid_mixed"


def plot_smile_fit(
    market_data: Dict,
    model_ivs: np.ndarray,
    date_str: str,
    method: str,
    label: str,
    out_dir: Path,
):
    """Smile fit: market IV dots vs model IV line, grouped by maturity."""
    K_arr    = market_data["K"]
    tau_arr  = market_data["tau"]
    iv_mkt   = market_data["iv_market"]
    moneyness = market_data.get("moneyness", K_arr / market_data["S"])

    unique_taus = sorted(np.unique(tau_arr))
    n_mat = min(len(unique_taus), 4)
    # pick n_mat evenly spaced maturities
    indices = np.linspace(0, len(unique_taus) - 1, n_mat, dtype=int)
    plot_taus = [unique_taus[i] for i in indices]

    cmap = plt.cm.plasma
    colors = [cmap(i / max(n_mat - 1, 1)) for i in range(n_mat)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tau, color in zip(plot_taus, colors):
        mask = tau_arr == tau
        mon  = moneyness[mask]
        iv_m = iv_mkt[mask] * 100
        iv_o = model_ivs[mask] * 100 if model_ivs is not None else None
        order = np.argsort(mon)
        t_days = int(round(tau * 365))
        ax.scatter(mon[order], iv_m[order], s=30, color=color,
                   alpha=0.7, label=f"Market {t_days}d", zorder=3)
        if iv_o is not None:
            valid = ~np.isnan(iv_o)
            if valid.sum() > 1:
                ax.plot(mon[order][valid[order]], iv_o[order][valid[order]],
                        color=color, lw=1.8, ls="--",
                        label=f"Model {t_days}d")

    ax.set_xlabel("Moneyness $K/S_0$")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"{METHOD_LABELS[method]}  —  {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}\n"
                 f"({label.replace('_', ' ')})")
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.tight_layout()
    fname = out_dir / f"skew_{date_str}_{method}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    skew -> {fname.name}")


def plot_residuals_heatmap(
    market_data: Dict,
    model_ivs: np.ndarray,
    date_str: str,
    method: str,
    label: str,
    out_dir: Path,
):
    """Residual heatmap: (moneyness bin x maturity bin) -> mean signed error."""
    K_arr    = market_data["K"]
    tau_arr  = market_data["tau"]
    iv_mkt   = market_data["iv_market"]
    moneyness = market_data.get("moneyness", K_arr / market_data["S"])

    valid = ~np.isnan(model_ivs)
    if valid.sum() < 5:
        logger.warning(f"    heatmap: too few valid model IVs for {date_str} {method}")
        return

    resid = (model_ivs - iv_mkt) * 100  # pp
    mon   = moneyness

    mon_bins = np.linspace(0.80, 1.20, 9)
    tau_bins = np.array(sorted(np.unique(np.round(tau_arr * 365).astype(int))))
    if len(tau_bins) < 2:
        tau_bins = np.array([7, 30, 60, 90])

    n_mon = len(mon_bins) - 1
    n_tau = len(tau_bins)
    grid  = np.full((n_tau, n_mon), np.nan)

    for ti, T_day in enumerate(tau_bins):
        T = T_day / 365.0
        t_mask = (np.abs(tau_arr - T) < 2 / 365) & valid
        for mi in range(n_mon):
            m_mask = (mon >= mon_bins[mi]) & (mon < mon_bins[mi + 1]) & t_mask
            if m_mask.sum() > 0:
                grid[ti, mi] = np.nanmean(resid[m_mask])

    fig, ax = plt.subplots(figsize=(8, 4))
    vmax = np.nanmax(np.abs(grid)) if not np.all(np.isnan(grid)) else 5.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    mon_labels = [f"{(mon_bins[i] + mon_bins[i+1])/2:.2f}" for i in range(n_mon)]
    tau_labels = [str(t) for t in tau_bins]

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", norm=norm,
                   origin="lower")
    ax.set_xticks(range(n_mon))
    ax.set_xticklabels(mon_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_tau))
    ax.set_yticklabels(tau_labels, fontsize=8)
    ax.set_xlabel("Moneyness bin centre")
    ax.set_ylabel("Maturity (days)")
    ax.set_title(f"Residuals (pp): {METHOD_LABELS[method]}  —  {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Model IV − Market IV (pp)")
    fig.tight_layout()
    fname = out_dir / f"residuals_heatmap_{date_str}_{method}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    heatmap -> {fname.name}")


def plot_residuals_hist(
    market_data: Dict,
    model_ivs: np.ndarray,
    date_str: str,
    method: str,
    label: str,
    out_dir: Path,
):
    """Histogram of calibration residuals in pp."""
    iv_mkt = market_data["iv_market"]
    valid  = ~np.isnan(model_ivs)
    if valid.sum() < 5:
        return
    resid = (model_ivs[valid] - iv_mkt[valid]) * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(resid, bins=20, color=METHOD_COLORS[method], edgecolor="white",
            alpha=0.85, density=True)
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Model IV − Market IV (pp)")
    ax.set_ylabel("Density")
    ax.set_title(f"Residuals: {METHOD_LABELS[method]}  —  {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}")
    mu, sd = np.nanmean(resid), np.nanstd(resid)
    ax.text(0.97, 0.95, f"$\\mu$={mu:.2f}pp\n$\\sigma$={sd:.2f}pp",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    fig.tight_layout()
    fname = out_dir / f"residuals_hist_{date_str}_{method}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    hist   -> {fname.name}")


def plot_surface_3d(
    market_data: Dict,
    model_ivs: np.ndarray,
    date_str: str,
    method: str,
    label: str,
    out_dir: Path,
):
    """3-D interpolated surface (model) + scatter (market) for the IV surface."""
    from scipy.interpolate import griddata as _griddata
    from matplotlib.colors import Normalize

    K_arr    = market_data["K"]
    tau_arr  = market_data["tau"]
    iv_mkt   = market_data["iv_market"]
    moneyness = market_data.get("moneyness", K_arr / market_data["S"])
    valid = ~np.isnan(model_ivs)

    tau_days = tau_arr * 365

    fig = plt.figure(figsize=(8, 5))
    ax  = fig.add_subplot(111, projection="3d")

    # Build interpolated model surface (like Chapter 2)
    if valid.sum() > 4:
        mon_v = moneyness[valid]
        tau_v = tau_days[valid]
        iv_v  = model_ivs[valid] * 100

        mon_grid = np.linspace(mon_v.min(), mon_v.max(), 60)
        tau_grid = np.linspace(tau_v.min(), tau_v.max(), 60)
        mg, tg   = np.meshgrid(mon_grid, tau_grid)

        iv_grid = _griddata(
            np.column_stack([mon_v, tau_v]), iv_v, (mg, tg), method="linear"
        )
        iv_floor = np.nanpercentile(iv_v, 2)
        iv_ceil  = np.nanpercentile(iv_v, 98)
        iv_grid  = np.clip(iv_grid, iv_floor, iv_ceil)
        norm = Normalize(vmin=iv_floor, vmax=iv_ceil)

        surf = ax.plot_surface(
            mg, tg, iv_grid, cmap="plasma", norm=norm,
            alpha=0.85, antialiased=True, edgecolor="none",
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.12,
                     label="IV (%)")

    # Overlay market observations as scatter
    ax.scatter(moneyness, tau_days, iv_mkt * 100,
               color="black", s=18, alpha=0.7, label="Market", zorder=5)

    ax.set_xlabel("Moneyness", labelpad=6)
    ax.set_ylabel("TTM (days)", labelpad=6)
    ax.set_zlabel("IV (%)", labelpad=6)
    ax.set_title(f"{METHOD_LABELS[method]}  —  {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.view_init(elev=25, azim=45)
    fig.tight_layout()
    fname = out_dir / f"iv_surface_3d_{date_str}_{method}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    3d     -> {fname.name}")


# =============================================================================
#  MAIN
# =============================================================================

def load_csv_results(method: str, date_filter: Optional[str] = None) -> pl.DataFrame:
    """Load all calibration CSVs for a given method."""
    method_dir = TABLES_DIR / method
    if not method_dir.exists():
        return pl.DataFrame()
    files = sorted(method_dir.glob(f"calibration_*_{method}.csv"))
    if date_filter:
        files = [f for f in files if date_filter in f.name]
    if not files:
        return pl.DataFrame()
    frames = [pl.read_csv(f) for f in files]
    return pl.concat(frames, how="diagonal")


def run_generate_figures(
    methods: List[str],
    date_filter: Optional[str] = None,
):
    logger.info("=" * 70)
    logger.info("  FIGURE GENERATION — rBergomi Chapter 3")
    logger.info(f"  Methods: {methods}")
    logger.info(f"  Date filter: {date_filter or 'all'}")
    logger.info("=" * 70)

    # Build IV surface index once
    logger.info("\n[Step 1] Loading option data index...")
    builder = IVSurfaceBuilder(str(DATA_DIR))
    builder.build_index()

    for method in methods:
        logger.info(f"\n{'-'*60}")
        logger.info(f"  METHOD: {METHOD_LABELS[method]}")
        logger.info(f"{'-'*60}")

        out_dir = FIGURES_DIR / method
        out_dir.mkdir(parents=True, exist_ok=True)

        df = load_csv_results(method, date_filter)
        if df.is_empty():
            logger.warning(f"  No CSV results found for {method}. Skipping.")
            continue

        logger.info(f"  Loaded {len(df)} calibration results.")

        for row in df.iter_rows(named=True):
            date_str_raw = str(row["date"])
            # Normalise to YYYYMMDD regardless of source format
            if len(date_str_raw) == 8 and date_str_raw.isdigit():
                dt = datetime.strptime(date_str_raw, "%Y%m%d").replace(hour=12)
            else:
                dt = datetime.fromisoformat(date_str_raw).replace(hour=12)
            date_str = dt.strftime("%Y%m%d")

            H   = float(row["H"])
            eta = float(row["eta"])
            rho = float(row["rho"])
            lbl = str(row.get("label", ""))

            logger.info(f"\n  [{method}] {date_str}  H={H:.4f} eta={eta:.3f} rho={rho:.3f}")
            try:
                surface = builder.get_iv_surface(target_time=dt, **SURFACE_CFG)
                if surface.height < 5:
                    logger.warning(f"  Surface too sparse ({surface.height} pts). Skipping.")
                    continue
                market_data = builder.export_for_rbergomi(surface)
            except Exception as e:
                logger.warning(f"  Surface extraction failed: {e}")
                continue

            # Reconstruct xi0
            xi0 = estimate_xi0(market_data)

            # Run pricer
            try:
                model_ivs = compute_model_ivs(H, eta, rho, xi0, market_data, method)
            except Exception as e:
                logger.warning(f"  Pricing failed: {e}")
                model_ivs = None

            # Generate figures
            try:
                plot_smile_fit(market_data, model_ivs, date_str, method, lbl, out_dir)
            except Exception as e:
                logger.warning(f"  Smile plot failed: {e}")
            try:
                plot_residuals_heatmap(market_data, model_ivs, date_str, method, lbl, out_dir)
            except Exception as e:
                logger.warning(f"  Heatmap failed: {e}")
            try:
                plot_residuals_hist(market_data, model_ivs, date_str, method, lbl, out_dir)
            except Exception as e:
                logger.warning(f"  Hist failed: {e}")
            try:
                plot_surface_3d(market_data, model_ivs, date_str, method, lbl, out_dir)
            except Exception as e:
                logger.warning(f"  3D surface failed: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("  Figure generation complete.")
    logger.info(f"  Output: {FIGURES_DIR}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Chapter 3 figures from calibration CSV results"
    )
    parser.add_argument("--method", choices=list(METHOD_CFG.keys()),
                        default=None, help="Run one method only")
    parser.add_argument("--date", default=None,
                        help="Run one date only (YYYYMMDD)")
    args = parser.parse_args()

    methods = [args.method] if args.method else list(METHOD_CFG.keys())
    run_generate_figures(methods=methods, date_filter=args.date)
