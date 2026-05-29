"""
fix_calibration_bias.py
=======================
Investigates and fixes the systematic upward shift in rBergomi model IVs.

ROOT CAUSE ANALYSIS
-------------------
The current calibration pipeline sets the piecewise-constant forward variance
curve as:

    xi_0(t) = sigma_ATM(T_i)^2    for t in [T_{i-1}, T_i]

But under the rBergomi model the expected integrated variance over [0, T] is:

    E[ (1/T) ∫₀ᵀ V_t dt ] = (1/T) ∫₀ᵀ xi_0(t) dt

For a piecewise-constant curve with k segments (T_0=0, T_1, ..., T_k):

    E[IV²(T_k)] = (1/T_k) Σᵢ₌₁ᵏ xi_0(T_i) · (T_i - T_{i-1})
                ≠ sigma_ATM(T_k)^2   (unless the IV term structure is flat)

For a downward-sloping term structure (BTC: higher short-maturity IV), shorter
maturities carry higher sigma^2, so the integral overshoots the target for
longer maturities → model IVs are translated UPWARD.

FIX: Bootstrap forward variance correctly
------------------------------------------
The correct piecewise-constant FORWARD variance f_i on [T_{i-1}, T_i] is
obtained by inverting:

    sigma_k^2 · T_k = Σᵢ₌₁ᵏ f_i · (T_i - T_{i-1})

Recursively:

    f_i = (sigma_i^2 · T_i - sigma_{i-1}^2 · T_{i-1}) / (T_i - T_{i-1})

The first segment (i=1) gives f_1 = sigma_1^2, consistent with the current
code for the shortest maturity. All subsequent segments are corrected.

ALTERNATIVE APPROACHES TESTED
------------------------------
A.  Correct xi bootstrap (described above)         ← main fix
B.  Flat xi = mean(sigma_ATM²) across maturities   ← simple baseline
C.  Re-optimize xi as a single free parameter       ← during calibration

Usage:
    python fix_calibration_bias.py
    python fix_calibration_bias.py --date 20241204
    python fix_calibration_bias.py --date 20250203 --method hybrid_euler
    python fix_calibration_bias.py --all-dates
"""

import argparse
import logging
import sys
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("Results/fix_calibration_bias.log",
                            mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("FixBias")

# ── Project imports ───────────────────────────────────────────────────────────
from iv_surface_builder import IVSurfaceBuilder
from src.rbergomi.calibrator import Calibrator, ForwardVarianceCurve
from src.rbergomi.pricer import rBergomiPricer
from src.rbergomi.utils import implied_vol_batch, bs_d1
from src.config.surfaces import SURFACE_CFG
from src.config.methods import METHODS
from src.config.plot_style import apply_thesis_style, PALETTE

DATA_DIR   = Path("data/option")
TABLES_DIR = Path("Results/calibration/tables")
OUT_DIR    = Path("Results/fix_bias")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

apply_thesis_style()


# =============================================================================
# CORRECT FORWARD VARIANCE BOOTSTRAP
# =============================================================================

def bootstrap_forward_variance(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
    floor: float = 1e-6,
) -> np.ndarray:
    """
    Bootstrap piecewise-constant FORWARD variances from total ATM IVs.

    Given ATM implied vols sigma_1, ..., sigma_n at sorted maturities
    T_1 < ... < T_n (with T_0 = 0), the forward variance on [T_{i-1}, T_i] is:

        f_i = (sigma_i^2 · T_i - sigma_{i-1}^2 · T_{i-1}) / (T_i - T_{i-1})

    A non-monotone total-variance term structure can produce f_i < 0 (calendar
    spread arbitrage in the data); these are floored at `floor` to keep the
    model well-defined.

    Parameters
    ----------
    maturities : np.ndarray
        Sorted maturities T_1 < ... < T_n  (in years, all > 0)
    atm_ivs : np.ndarray
        ATM implied vol at each maturity (same units as maturities)
    floor : float
        Minimum forward variance (prevents negative xi)

    Returns
    -------
    np.ndarray
        Forward variances f_1, ..., f_n
    """
    n = len(maturities)
    assert n == len(atm_ivs), "maturities and atm_ivs must have the same length"

    total_var = atm_ivs ** 2 * maturities   # sigma_i^2 * T_i
    T_prev = np.concatenate([[0.0], maturities[:-1]])
    tv_prev = np.concatenate([[0.0], total_var[:-1]])
    dT = maturities - T_prev

    forward_var = (total_var - tv_prev) / dT  # may be negative

    # Log any problematic segments
    neg_mask = forward_var < 0
    if neg_mask.any():
        for i in np.where(neg_mask)[0]:
            log.warning(
                f"  Negative forward variance at T={maturities[i]:.4f}: "
                f"f={forward_var[i]:.6f}  "
                f"(total_var decreased: {tv_prev[i]:.6f} -> {total_var[i]:.6f})"
            )

    return np.maximum(forward_var, floor)


def make_corrected_xi_curve(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
) -> ForwardVarianceCurve:
    """
    Build a ForwardVarianceCurve using the correct forward-variance bootstrap.
    """
    forward_vars = bootstrap_forward_variance(maturities, atm_ivs)
    return ForwardVarianceCurve(maturities, forward_vars)


def make_naive_xi_curve(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
) -> ForwardVarianceCurve:
    """
    Build a ForwardVarianceCurve using the CURRENT (naive) approach:
    xi_0(t) = sigma_ATM(T_i)^2  (total variance, not forward variance).
    """
    return ForwardVarianceCurve(maturities, atm_ivs ** 2)


# =============================================================================
# DIAGNOSTIC HELPERS
# =============================================================================

def extract_atm_ivs_from_market(
    market_data: dict,
    calibrator: Calibrator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use Calibrator's internal ATM finder to get (maturities, atm_ivs).
    Returns sorted arrays.
    """
    calibrator._xi0_curve = None
    xi_curve, atm_ivs_dict = calibrator._estimate_forward_variance_curve(market_data)

    maturities = np.array(sorted(atm_ivs_dict.keys()))
    atm_ivs    = np.array([atm_ivs_dict[T] for T in maturities])
    return maturities, atm_ivs


def price_surface(
    H: float, eta: float, rho: float,
    xi_curve: ForwardVarianceCurve,
    market_data: dict,
    method_cfg: dict,
    n_paths: int = 5_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-price the market surface and return (model_ivs, market_ivs) in decimal.
    Only returns points where IV inversion succeeded.
    """
    pricer = rBergomiPricer(
        H=H, eta=eta, rho=rho, xi=xi_curve,
        n_paths=n_paths,
        n_steps=method_cfg.get("n_steps", 100),
        scheme=method_cfg.get("scheme", "hybrid"),
        kappa=method_cfg.get("kappa", 6),
        seed=method_cfg.get("seed", 42),
        antithetic=method_cfg.get("antithetic", False),
        pricing_method=method_cfg.get("pricing_method", "euler"),
    )

    S0      = market_data["S"]
    K_arr   = market_data["K"]
    tau_arr = market_data["tau"]
    iv_mkt  = market_data["iv_market"]

    model_prices = np.full(len(K_arr), np.nan)
    for T in np.unique(tau_arr):
        mask = tau_arr == T
        idx  = np.where(mask)[0]
        try:
            prices, _ = pricer.price_multiple_strikes(S0, K_arr[mask], T, option_type='put')
            model_prices[idx] = prices
        except Exception as e:
            log.debug(f"    Batch failed T={T:.4f}: {e}")
            for j, ii in enumerate(idx):
                try:
                    p, _ = pricer.price_inverse_put(S0, K_arr[ii], T)
                    model_prices[ii] = p
                except Exception:
                    pass

    model_ivs = implied_vol_batch(model_prices, S0, K_arr, tau_arr,
                                  option_type='put', r=0.0)
    valid = ~np.isnan(model_ivs)
    return model_ivs[valid], iv_mkt[valid], tau_arr[valid]


def bias_stats(model_ivs: np.ndarray, market_ivs: np.ndarray) -> dict:
    """Return error statistics in percentage points."""
    err = (model_ivs - market_ivs) * 100
    return {
        "bias_pp":  float(np.mean(err)),
        "rmse_pp":  float(np.sqrt(np.mean(err**2))),
        "mae_pp":   float(np.mean(np.abs(err))),
        "std_pp":   float(np.std(err, ddof=1)),
        "n":        len(err),
    }


# =============================================================================
# TERM STRUCTURE VISUALISATION
# =============================================================================

def plot_xi_comparison(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
    date_str: str,
) -> None:
    """
    Plot naive vs corrected xi curves alongside the market ATM term structure.
    """
    fwd_vars = bootstrap_forward_variance(maturities, atm_ivs)
    naive_vars = atm_ivs ** 2

    # Expected model ATM IV under each curve
    # naive:     E[IV²(T_k)] = (1/T_k) Σ sigma_i² · ΔT_i   (overshoots for k>1)
    # corrected: E[IV²(T_k)] = (1/T_k) Σ f_i · ΔT_i = sigma_k²  (exact by construction)
    T_prev = np.concatenate([[0.0], maturities[:-1]])
    dT = maturities - T_prev

    naive_model_atm_sq = np.array([
        np.sum(naive_vars[:k+1] * dT[:k+1]) / maturities[k]
        for k in range(len(maturities))
    ])
    corr_model_atm_sq  = np.array([
        np.sum(fwd_vars[:k+1] * dT[:k+1]) / maturities[k]
        for k in range(len(maturities))
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left: xi_0 curves ──────────────────────────────────────────────────
    ax = axes[0]
    ax.step(maturities * 252, naive_vars,  where='post', color=PALETTE['tertiary'],
            lw=2, label='Naive: $\\xi_0(T) = \\sigma_{\\mathrm{ATM}}^2(T)$')
    ax.step(maturities * 252, fwd_vars,    where='post', color=PALETTE['primary'],
            lw=2, label='Corrected: forward variance $f_i$')
    ax.plot(maturities * 252, atm_ivs**2,  'o--', color=PALETTE['neutral'],
            lw=1, ms=5, label='Market $\\sigma_{\\mathrm{ATM}}^2(T)$')
    ax.set_xlabel("Maturity (days)")
    ax.set_ylabel("Variance")
    ax.set_title(f"Forward Variance Curve — {date_str}")
    ax.legend(fontsize=9)

    # ── Right: implied ATM IV recovery ────────────────────────────────────
    ax = axes[1]
    ax.plot(maturities * 252, atm_ivs * 100,
            'o-', color=PALETTE['neutral'], lw=1.5, ms=6,
            label='Market ATM IV')
    ax.plot(maturities * 252, np.sqrt(naive_model_atm_sq) * 100,
            's--', color=PALETTE['tertiary'], lw=1.5, ms=6,
            label='Naive model ATM IV')
    ax.plot(maturities * 252, np.sqrt(corr_model_atm_sq) * 100,
            '^-', color=PALETTE['primary'], lw=1.5, ms=6,
            label='Corrected model ATM IV')
    ax.set_xlabel("Maturity (days)")
    ax.set_ylabel("ATM IV (%)")
    ax.set_title(f"ATM IV Recovery — {date_str}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / f"xi_comparison_{date_str}.png", dpi=150)
    plt.close(fig)
    log.info(f"  Figure saved: xi_comparison_{date_str}.png")


def plot_iv_surface_comparison(
    market_data: dict,
    model_ivs_naive: np.ndarray,
    model_ivs_corr: np.ndarray,
    iv_mkt: np.ndarray,
    taus: np.ndarray,
    date_str: str,
) -> None:
    """
    Scatter plots: naive vs corrected model IV vs market IV, coloured by maturity.
    """
    S0 = market_data["S"]
    moneyness = market_data["K"] / S0

    # Only use valid (non-nan) intersections
    valid = (~np.isnan(model_ivs_naive)) & (~np.isnan(model_ivs_corr))
    m_n  = model_ivs_naive[valid] * 100
    m_c  = model_ivs_corr[valid] * 100
    m_mk = iv_mkt[valid] * 100
    tau_ = taus[valid] * 252  # in days

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, model_iv, title_suffix in zip(
        axes, [m_n, m_c],
        ["Naive $\\xi_0 = \\sigma_{\\mathrm{ATM}}^2$  (current)",
         "Corrected $\\xi_0$ = forward variance  (fix)"]
    ):
        sc = ax.scatter(m_mk, model_iv, c=tau_, cmap='plasma', alpha=0.7, s=25)
        lim = [min(m_mk.min(), model_iv.min()) - 2,
               max(m_mk.max(), model_iv.max()) + 2]
        ax.plot(lim, lim, '--', color=PALETTE['neutral'], lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Market IV (%)")
        ax.set_ylabel("Model IV (%)")
        ax.set_title(title_suffix, fontsize=10)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Maturity (days)")

        bias = float(np.mean(model_iv - m_mk))
        rmse = float(np.sqrt(np.mean((model_iv - m_mk)**2)))
        ax.text(0.03, 0.97, f"Bias={bias:+.1f} pp\nRMSE={rmse:.1f} pp",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(f"Model vs Market IV — {date_str}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / f"iv_scatter_{date_str}.png", dpi=150)
    plt.close(fig)
    log.info(f"  Figure saved: iv_scatter_{date_str}.png")


# =============================================================================
# SINGLE DATE ANALYSIS
# =============================================================================

def analyse_date(
    dt: datetime,
    method: str,
    builder: IVSurfaceBuilder,
    n_paths: int = 3_000,
) -> Optional[dict]:
    """
    Run diagnostic on a single calibration date.
    Loads existing calibration result, builds both xi curves, re-prices, compares.
    """
    date_str = dt.strftime("%Y%m%d")
    log.info(f"\n{'='*70}")
    log.info(f"  DATE: {date_str}   METHOD: {method}")
    log.info(f"{'='*70}")

    # ── Load calibration result ───────────────────────────────────────────────
    csv_path = TABLES_DIR / method / f"calibration_{date_str}_{method}.csv"
    if not csv_path.exists():
        log.warning(f"  No calibration result found: {csv_path}")
        return None

    row = pl.read_csv(csv_path).row(0, named=True)
    H   = float(row["H"])
    eta = float(row["eta"])
    rho = float(row["rho"])
    log.info(f"  Calibrated params: H={H:.4f}  eta={eta:.3f}  rho={rho:.3f}")

    # ── Load market data ──────────────────────────────────────────────────────
    try:
        surface     = builder.get_iv_surface(target_time=dt, **SURFACE_CFG)
        market_data = builder.export_for_rbergomi(surface)
    except Exception as e:
        log.warning(f"  Cannot load market data: {e}")
        return None

    # Apply moneyness filter (same as calibration)
    mlo, mhi = 0.80, 1.20
    S0 = market_data["S"]
    mono = market_data["K"] / S0
    keep = (mono >= mlo) & (mono <= mhi)
    market_data = {
        "S":         S0,
        "K":         market_data["K"][keep],
        "tau":       market_data["tau"][keep],
        "iv_market": market_data["iv_market"][keep],
        "n_points":  int(keep.sum()),
    }

    if market_data["n_points"] < 5:
        log.warning("  Too few market points after moneyness filter")
        return None

    # ── Build calibrator (only for ATM extraction) ────────────────────────────
    cfg = {k: v for k, v in METHODS[method].items()
           if k not in ("label", "maxiter")}
    calibrator = Calibrator(**cfg)

    maturities, atm_ivs = extract_atm_ivs_from_market(market_data, calibrator)
    log.info(f"  ATM IVs: { {f'T={T*252:.0f}d': f'{iv*100:.1f}%' for T, iv in zip(maturities, atm_ivs)} }")

    # ── Build xi curves ───────────────────────────────────────────────────────
    xi_naive = make_naive_xi_curve(maturities, atm_ivs)
    xi_corr  = make_corrected_xi_curve(maturities, atm_ivs)

    fwd_vars = bootstrap_forward_variance(maturities, atm_ivs)
    log.info("  Forward variance bootstrap:")
    log.info(f"    {'Maturity':>10}  {'sigma_ATM':>10}  {'Naive xi':>10}  "
             f"{'Forward xi':>12}  {'Diff%':>8}")
    for T, sig, fv in zip(maturities, atm_ivs, fwd_vars):
        naive_xi = sig**2
        log.info(f"    {T*252:>10.1f}d  {sig*100:>9.2f}%  {naive_xi:>10.5f}  "
                 f"{fv:>12.5f}  {(fv - naive_xi) / naive_xi * 100:>+8.2f}%")

    # ── Price with both curves ────────────────────────────────────────────────
    pricer_cfg = {
        "n_steps": cfg.get("n_steps", 100),
        "scheme":  cfg.get("scheme", "hybrid"),
        "kappa":   cfg.get("kappa", 6),
        "seed":    cfg.get("seed", 42),
        "antithetic": cfg.get("antithetic", False),
        "pricing_method": cfg.get("pricing_method", "euler"),
    }

    log.info(f"\n  Pricing with NAIVE xi  (n_paths={n_paths:,}) ...")
    m_iv_naive, iv_mkt_n, tau_n = price_surface(
        H, eta, rho, xi_naive, market_data, pricer_cfg, n_paths=n_paths)
    stats_naive = bias_stats(m_iv_naive, iv_mkt_n)
    log.info(f"    Bias={stats_naive['bias_pp']:+.2f} pp  "
             f"RMSE={stats_naive['rmse_pp']:.2f} pp  "
             f"N={stats_naive['n']}")

    log.info(f"\n  Pricing with CORRECTED xi  (n_paths={n_paths:,}) ...")
    m_iv_corr, iv_mkt_c, tau_c = price_surface(
        H, eta, rho, xi_corr, market_data, pricer_cfg, n_paths=n_paths)
    stats_corr = bias_stats(m_iv_corr, iv_mkt_c)
    log.info(f"    Bias={stats_corr['bias_pp']:+.2f} pp  "
             f"RMSE={stats_corr['rmse_pp']:.2f} pp  "
             f"N={stats_corr['n']}")

    log.info(f"\n  IMPROVEMENT:")
    log.info(f"    RMSE:  {stats_naive['rmse_pp']:.2f} → {stats_corr['rmse_pp']:.2f} pp  "
             f"(Δ = {stats_corr['rmse_pp'] - stats_naive['rmse_pp']:+.2f})")
    log.info(f"    Bias:  {stats_naive['bias_pp']:+.2f} → {stats_corr['bias_pp']:+.2f} pp")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_xi_comparison(maturities, atm_ivs, f"{date_str}_{method}")

    plot_iv_surface_comparison(
        market_data,
        m_iv_naive, m_iv_corr, iv_mkt_c, tau_c, f"{date_str}_{method}",
    )

    return {
        "date":           date_str,
        "method":         method,
        "H":              H, "eta": eta, "rho": rho,
        "n_maturities":   len(maturities),
        "bias_naive_pp":  stats_naive["bias_pp"],
        "rmse_naive_pp":  stats_naive["rmse_pp"],
        "bias_corr_pp":   stats_corr["bias_pp"],
        "rmse_corr_pp":   stats_corr["rmse_pp"],
        "delta_rmse_pp":  stats_corr["rmse_pp"] - stats_naive["rmse_pp"],
        "delta_bias_pp":  stats_corr["bias_pp"]  - stats_naive["bias_pp"],
        "n_points":       stats_naive["n"],
    }


# =============================================================================
# THEORETICAL ILLUSTRATION
# =============================================================================

def illustrate_bootstrap_error() -> None:
    """
    Illustrate the integrated-variance error analytically for a simple term
    structure without needing market data.
    """
    log.info("\n" + "="*70)
    log.info("  THEORETICAL ILLUSTRATION: Forward Variance Bootstrap Error")
    log.info("="*70)

    # Toy term structure: BTC-like — short maturities carry higher IV
    maturities = np.array([7, 14, 30, 60, 90]) / 252.0
    atm_ivs    = np.array([0.80, 0.70, 0.60, 0.55, 0.52])  # 80%, 70%, ...

    total_var  = atm_ivs**2 * maturities
    naive_vars = atm_ivs**2
    fwd_vars   = bootstrap_forward_variance(maturities, atm_ivs)

    T_prev = np.concatenate([[0.0], maturities[:-1]])
    dT = maturities - T_prev

    log.info(f"\n  Toy term structure (BTC-like, downward sloping):")
    log.info(f"  {'T(d)':>6}  {'sigma(%)':>9}  {'Naive xi':>10}  "
             f"{'Fwd xi':>10}  {'Naive E[IV²]':>14}  {'Correct E[IV²]':>15}")

    for k in range(len(maturities)):
        naive_eiv2 = np.sum(naive_vars[:k+1] * dT[:k+1]) / maturities[k]
        corr_eiv2  = np.sum(fwd_vars[:k+1]  * dT[:k+1]) / maturities[k]
        T_d = maturities[k] * 252
        log.info(
            f"  {T_d:>6.0f}  {atm_ivs[k]*100:>8.1f}%  "
            f"{naive_vars[k]:>10.5f}  {fwd_vars[k]:>10.5f}  "
            f"{naive_eiv2:>12.5f}  {corr_eiv2:>14.5f}  "
            f"[target={atm_ivs[k]**2:.5f}]"
        )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(maturities*252, atm_ivs*100, 'o-', color=PALETTE['neutral'],
            lw=2, label='Market ATM IV', ms=7)
    ax.set_xlabel("Maturity (days)"); ax.set_ylabel("IV (%)")
    ax.set_title("Input: BTC-like ATM IV term structure")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.step(maturities*252, naive_vars, where='post', color=PALETTE['tertiary'],
            lw=2.5, label='Naive $\\xi_0 = \\sigma^2$  ← current code')
    ax.step(maturities*252, fwd_vars, where='post', color=PALETTE['primary'],
            lw=2.5, label='Correct forward variance $f_i$  ← fix')
    ax.plot(maturities*252, atm_ivs**2, 'o--', color=PALETTE['neutral'],
            lw=1, ms=5, label='Target $\\sigma_{\\mathrm{ATM}}^2$')
    ax.set_xlabel("Maturity (days)"); ax.set_ylabel("Variance")
    ax.set_title("Naive vs corrected piecewise-constant $\\xi_0(t)$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "bootstrap_illustration.png", dpi=150)
    plt.close(fig)
    log.info(f"\n  Illustration saved to {OUT_DIR}/figures/bootstrap_illustration.png")

    # Show the error magnitude
    log.info("\n  Error from naive bootstrap (model ATM IV % deviation):")
    for k in range(len(maturities)):
        naive_eiv2 = np.sum(naive_vars[:k+1] * dT[:k+1]) / maturities[k]
        model_atm_iv = np.sqrt(naive_eiv2) * 100
        market_atm_iv = atm_ivs[k] * 100
        log.info(
            f"    T={maturities[k]*252:.0f}d: "
            f"market={market_atm_iv:.1f}%  model={model_atm_iv:.1f}%  "
            f"diff={model_atm_iv - market_atm_iv:+.1f} pp"
        )


# =============================================================================
# WHAT DOES THE FIX CHANGE IN THE CALIBRATOR?
# =============================================================================

def print_implementation_plan() -> None:
    """Print the exact lines in calibrator.py that need to change."""
    log.info("\n" + "="*70)
    log.info("  IMPLEMENTATION PLAN: Changes needed in calibrator.py")
    log.info("="*70)
    log.info("""
Method to modify: Calibrator._estimate_forward_variance_curve()
File: src/rbergomi/calibrator.py  (~line 490)

CURRENT (wrong) — last lines of the method:
    return ForwardVarianceCurve(
        np.array(maturities), np.array(xi_values)
    ), atm_ivs

where xi_values[i] = atm_iv[i]^2  (total variance = naive xi)

FIX — replace with bootstrap:
    maturities_arr = np.array(maturities)
    atm_ivs_arr    = np.array([atm_ivs[T] for T in maturities])

    # Forward variance bootstrap: f_i = (sigma_i² T_i - sigma_{i-1}² T_{i-1}) / ΔT_i
    total_var = atm_ivs_arr**2 * maturities_arr
    T_prev  = np.concatenate([[0.0], maturities_arr[:-1]])
    tv_prev = np.concatenate([[0.0], total_var[:-1]])
    forward_vars = np.maximum((total_var - tv_prev) / (maturities_arr - T_prev), 1e-6)

    return ForwardVarianceCurve(maturities_arr, forward_vars), atm_ivs

This change guarantees:
    E[IV²(T_k)] = sigma_ATM(T_k)^2   for ALL k, by construction.

Important: after this fix, (H, eta, rho) should be RE-CALIBRATED from scratch
because the old parameter set was tuned to compensate the xi bias. The new xi
is no longer overestimated, so rho/eta may shift.
""")


# =============================================================================
# MAIN
# =============================================================================

ALL_METHODS = ["cholesky_euler", "hybrid_euler", "hybrid_mixed"]


def collect_dates_for_method(method: str) -> list:
    """Return sorted list of datetime objects from existing calibration CSVs."""
    method_dir = TABLES_DIR / method
    if not method_dir.exists():
        return []
    dates = []
    for p in sorted(method_dir.glob(f"calibration_*_{method}.csv")):
        ds = p.stem.split("_")[1]
        try:
            dates.append(datetime.strptime(ds, "%Y%m%d").replace(hour=12))
        except ValueError:
            pass
    return dates


def run_method(
    method: str,
    dates: list,
    builder: IVSurfaceBuilder,
    n_paths: int,
) -> pl.DataFrame:
    """Run bias analysis for all dates of one method. Returns summary DataFrame."""
    log.info(f"\n{'#'*70}")
    log.info(f"#  METHOD: {method}  ({len(dates)} dates)")
    log.info(f"{'#'*70}")

    results = []
    for dt in dates:
        res = analyse_date(dt, method, builder, n_paths=n_paths)
        if res is not None:
            results.append(res)

    if not results:
        log.warning(f"  No results for method={method}")
        return pl.DataFrame()

    df = pl.DataFrame(results)

    # Per-method CSV
    out_csv = OUT_DIR / f"bias_comparison_{method}.csv"
    df.write_csv(out_csv)
    log.info(f"\n  Per-date results saved to {out_csv}")

    # Per-method summary table
    log.info(f"\n  {'Date':>10}  {'BiasNaive':>10}  {'RMSENaive':>10}  "
             f"{'BiasCorr':>9}  {'RMSECorr':>9}  {'ΔRMSE':>7}")
    for row in df.iter_rows(named=True):
        log.info(
            f"  {row['date']:>10}  {row['bias_naive_pp']:>+10.2f}  "
            f"{row['rmse_naive_pp']:>10.2f}  "
            f"{row['bias_corr_pp']:>+9.2f}  {row['rmse_corr_pp']:>9.2f}  "
            f"{row['delta_rmse_pp']:>+7.2f}"
        )

    avg_delta_rmse = float(df["delta_rmse_pp"].mean())
    avg_delta_bias = float(df["delta_bias_pp"].mean())
    log.info(f"\n  [{method}]  Mean ΔRMSE = {avg_delta_rmse:+.2f} pp  |  "
             f"Mean ΔBias = {avg_delta_bias:+.2f} pp")

    return df


def plot_cross_method_summary(all_results: Dict[str, pl.DataFrame]) -> None:
    """
    Bar chart comparing mean bias and RMSE (naive vs corrected) across all methods.
    """
    methods = [m for m in ALL_METHODS if m in all_results and len(all_results[m]) > 0]
    if not methods:
        return

    from src.config.plot_style import METHOD_COLORS

    metrics = {
        "bias_naive": [], "bias_corr": [],
        "rmse_naive": [], "rmse_corr": [],
    }
    for m in methods:
        df = all_results[m]
        metrics["bias_naive"].append(float(df["bias_naive_pp"].mean()))
        metrics["bias_corr"].append(float(df["bias_corr_pp"].mean()))
        metrics["rmse_naive"].append(float(df["rmse_naive_pp"].mean()))
        metrics["rmse_corr"].append(float(df["rmse_corr_pp"].mean()))

    x = np.arange(len(methods))
    w = 0.35
    labels_short = {"cholesky_euler": "Chol.+Euler",
                    "hybrid_euler":   "Hyb.+Euler",
                    "hybrid_mixed":   "Hyb.+Mixed"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (naive_key, corr_key), ylabel, title in zip(
        axes,
        [("bias_naive", "bias_corr"), ("rmse_naive", "rmse_corr")],
        ["Mean Bias (pp)", "Mean RMSE (pp)"],
        ["Mean Bias: Naive vs Corrected $\\xi_0$",
         "Mean RMSE: Naive vs Corrected $\\xi_0$"],
    ):
        bars_naive = ax.bar(x - w/2, metrics[naive_key], w,
                            label="Naive $\\xi_0$", color=PALETTE['tertiary'], alpha=0.85)
        bars_corr  = ax.bar(x + w/2, metrics[corr_key],  w,
                            label="Corrected $\\xi_0$", color=PALETTE['primary'], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([labels_short.get(m, m) for m in methods])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.axhline(0, color=PALETTE['neutral'], lw=0.8, ls='--')

        for bar in bars_naive:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:+.1f}", ha='center', va='bottom', fontsize=8)
        for bar in bars_corr:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:+.1f}", ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "figures" / "cross_method_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"\n  Cross-method summary figure saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose and fix calibration upward bias")
    parser.add_argument("--date",   default="20241204",
                        help="Single date to analyse (YYYYMMDD); ignored with --all-dates")
    parser.add_argument("--method", default=None,
                        choices=ALL_METHODS,
                        help="Single method (default: all three with --all-methods)")
    parser.add_argument("--all-methods", action="store_true",
                        help="Run on all three methods (cholesky_euler, hybrid_euler, hybrid_mixed)")
    parser.add_argument("--n-paths", type=int, default=3_000,
                        help="MC paths for re-pricing (fewer = faster, default=3000)")
    parser.add_argument("--all-dates", action="store_true",
                        help="Run on all available calibration dates (per method)")
    parser.add_argument("--theory-only", action="store_true",
                        help="Only run the theoretical illustration (no market data needed)")
    args = parser.parse_args()

    # Always show theoretical illustration and implementation plan
    illustrate_bootstrap_error()
    print_implementation_plan()

    if args.theory_only:
        sys.exit(0)

    # ── Determine which methods to run ────────────────────────────────────────
    if args.all_methods:
        methods_to_run = ALL_METHODS
    elif args.method:
        methods_to_run = [args.method]
    else:
        methods_to_run = [args.method or "hybrid_euler"]

    # ── Load market data builder (shared across methods/dates) ────────────────
    builder = IVSurfaceBuilder(str(DATA_DIR))
    builder.build_index()

    # ── Run analysis ──────────────────────────────────────────────────────────
    all_results: Dict[str, pl.DataFrame] = {}

    for method in methods_to_run:
        if args.all_dates:
            dates = collect_dates_for_method(method)
            if not dates:
                log.warning(f"  No calibration CSVs found for method={method} — skipping")
                continue
            log.info(f"\n  Found {len(dates)} dates for method={method}")
        else:
            dates = [datetime.strptime(args.date, "%Y%m%d").replace(hour=12)]

        df = run_method(method, dates, builder, n_paths=args.n_paths)
        if len(df) > 0:
            all_results[method] = df

    # ── Cross-method summary ──────────────────────────────────────────────────
    if len(all_results) > 1:
        plot_cross_method_summary(all_results)

        # Aggregated CSV across all methods
        combined = pl.concat(list(all_results.values()))
        combined.write_csv(OUT_DIR / "bias_comparison_all_methods.csv")

        log.info("\n" + "="*70)
        log.info("  OVERALL SUMMARY (all methods)")
        log.info("="*70)
        log.info(f"  {'Method':<20}  {'N dates':>7}  {'BiasNaive':>10}  "
                 f"{'RMSENaive':>10}  {'BiasCorr':>9}  {'RMSECorr':>9}  {'ΔRMSE':>7}")
        for method, df in all_results.items():
            log.info(
                f"  {method:<20}  {len(df):>7}  "
                f"{float(df['bias_naive_pp'].mean()):>+10.2f}  "
                f"{float(df['rmse_naive_pp'].mean()):>10.2f}  "
                f"{float(df['bias_corr_pp'].mean()):>+9.2f}  "
                f"{float(df['rmse_corr_pp'].mean()):>9.2f}  "
                f"{float(df['delta_rmse_pp'].mean()):>+7.2f}"
            )

        # Verdict
        all_delta = [float(df["delta_rmse_pp"].mean()) for df in all_results.values()]
        overall   = np.mean(all_delta)
        log.info(f"\n  Overall mean ΔRMSE across all methods: {overall:+.2f} pp")
        if overall < -1.0:
            log.info("  => FIX WORKS consistently across all methods")
        elif overall < 0:
            log.info("  => Modest improvement — re-calibrating (H, eta, rho) will amplify the gain")
        else:
            log.info("  => No consistent improvement — investigate other sources of bias")
