"""
run_all_calibrations.py
=======================
Production calibration of the rBergomi model using 3 simulation/pricing methods
on event-based snapshot dates (day-before, day-of, day-after) plus calm baselines.

Methods:
    1. Cholesky + Euler  -->  Results/calibration/tables/cholesky_euler/
    2. Hybrid  + Euler   -->  Results/calibration/tables/hybrid_euler/
    3. Hybrid  + Mixed   -->  Results/calibration/tables/hybrid_mixed/

All three methods use the unified Calibrator from src/rbergomi/.

Usage:
    python run_all_calibrations.py                     # all 3 methods
    python run_all_calibrations.py --cholesky           # Cholesky+Euler only
    python run_all_calibrations.py --hybrid-euler       # Hybrid+Euler only
    python run_all_calibrations.py --hybrid-mixed       # Hybrid+Mixed only
    python run_all_calibrations.py --hybrid-euler --hybrid-mixed

Requires: numba  (pip install numba)
"""

import sys
import logging
import time
import argparse
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# ── Logging (Windows-safe) ───────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Results/thesis_pipeline.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("RunAll")

# ── Project imports ──────────────────────────────────────────────────────────
from iv_surface_builder import IVSurfaceBuilder
from src.rbergomi.calibrator import Calibrator

# ── Shared config ─────────────────────────────────────────────────────────────
from src.config.surfaces import SURFACE_CFG
from src.config.methods import METHODS
from src.config.dates import (
    EVENT_DATES, LOW_IV_DATES, MEDIUM_IV_DATES, HIGH_IV_DATES,
    BASELINE_DATES, ALL_DATES, SNAPSHOT_DATE_STRINGS,
)

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR          = Path("data/option")
OUTPUT_BASE       = Path("Results/calibration")
MIN_SURFACE_POINTS = 15


# Dates are now imported from src.config.dates above


def build_snapshot_dates():
    """Return (datetime, label) tuples, sorted chronologically."""
    dates = []
    for ds in SNAPSHOT_DATE_STRINGS:
        dt = datetime.strptime(ds, "%Y%m%d").replace(hour=12, minute=0, second=0)
        label = ALL_DATES[ds]
        dates.append((dt, label))
    logger.info(f"Snapshot dates: {len(dates)} total "
                f"({len(EVENT_DATES)} event + {len(LOW_IV_DATES)} low-IV "
                f"+ {len(MEDIUM_IV_DATES)} mid-IV + {len(HIGH_IV_DATES)} high-IV)")
    for dt, label in dates:
        logger.info(f"  {dt.strftime('%Y-%m-%d')}  {label}")
    return dates


def check_baseline_atm_iv(builder):
    """
    For each baseline (non-event) date, extract the IV surface and compute
    the delta-neutral ATM implied volatility (shortest available maturity,
    moneyness closest to 1.0).

    Prints a table comparing the measured ATM IV against the current regime
    classification (Low / Medium / High) so the user can verify or correct it.

    The check runs quickly (~1s per date) because it only extracts the surface
    without running any Monte Carlo.
    """
    # Map date string -> current regime label
    regime_labels = {}
    for ds in LOW_IV_DATES:
        regime_labels[ds] = "Low    (<45%)"
    for ds in MEDIUM_IV_DATES:
        regime_labels[ds] = "Medium (45-70%)"
    for ds in HIGH_IV_DATES:
        regime_labels[ds] = "High   (>70%)"

    logger.info("\n" + "=" * 70)
    logger.info("  BASELINE ATM-IV VERIFICATION CHECK")
    logger.info("  (measures ATM IV for shortest available maturity)")
    logger.info("=" * 70)
    logger.info(f"  {'Date':<12}  {'Current regime':<18}  {'ATM IV':<10}  {'Auto-class':<18}  {'Match?'}")
    logger.info(f"  {'-'*12}  {'-'*18}  {'-'*10}  {'-'*18}  {'-'*6}")

    reclassify = []

    for ds in sorted(regime_labels.keys()):
        dt = datetime.strptime(ds, "%Y%m%d").replace(hour=12, minute=0, second=0)
        current_regime = regime_labels[ds]
        label = BASELINE_DATES[ds]

        try:
            surface = builder.get_iv_surface(target_time=dt, **SURFACE_CFG)
            if surface.height < 5:
                logger.info(f"  {ds:<12}  {current_regime:<18}  {'N/A':<10}  "
                            f"{'N/A (sparse)':<18}  [SKIP]")
                continue

            # Find shortest maturity available
            col_ttm = "ttm" if "ttm" in surface.columns else "maturity"
            col_iv  = "iv"  if "iv"  in surface.columns else "implied_vol"
            col_mon = "moneyness" if "moneyness" in surface.columns else "moneyness_fwd"

            min_ttm = surface[col_ttm].min()
            near_mat = surface.filter(pl.col(col_ttm) == min_ttm)

            # Pick row with moneyness closest to 1.0
            near_mat = near_mat.with_columns(
                (pl.col(col_mon) - 1.0).abs().alias("_dist_atm")
            )
            atm_row = near_mat.sort("_dist_atm").head(1)
            atm_iv  = float(atm_row[col_iv][0])  # already in % from Deribit

            # Auto-classify
            if atm_iv < 45.0:
                auto = "Low    (<45%)"
            elif atm_iv < 70.0:
                auto = "Medium (45-70%)"
            else:
                auto = "High   (>70%)"

            match = "[OK]" if auto.strip().split()[0] == current_regime.strip().split()[0] else "[!] MISMATCH"
            if "MISMATCH" in match:
                reclassify.append((ds, label, current_regime, auto, f"{atm_iv:.1f}%"))

            logger.info(f"  {ds:<12}  {current_regime:<18}  {atm_iv:>6.1f}%    "
                        f"{auto:<18}  {match}")

        except Exception as e:
            logger.warning(f"  {ds:<12}  {current_regime:<18}  ERROR: {e}")

    logger.info("=" * 70)
    if reclassify:
        logger.warning(f"\n  [!] {len(reclassify)} date(s) may need reclassification:")
        for ds, lbl, cur, auto, iv in reclassify:
            logger.warning(f"      {ds}  ({lbl})")
            logger.warning(f"        current={cur.strip()}  measured={iv}  suggested={auto.strip()}")
        logger.warning("\n  To fix: move the date key between LOW_IV_DATES / MEDIUM_IV_DATES / HIGH_IV_DATES")
    else:
        logger.info("  All baseline dates match their IV-regime classification. [OK]")
    logger.info("")


# ── Helper: save calibration CSV ─────────────────────────────────────────────
def save_csv(result_dict, date_str, method_name):
    out_dir = OUTPUT_BASE / "tables" / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_{date_str}_{method_name}.csv"
    pl.DataFrame([result_dict]).write_csv(out_path)
    logger.info(f"  CSV -> {out_path}")


# ── Generic model runner ─────────────────────────────────────────────────────
def run_method(method_name, method_cfg, builder, snapshot_dates, h_lb: float = 0.01):
    """
    Run one calibration method across all snapshot dates.

    Parameters
    ----------
    h_lb : float
        Lower bound for H.  Typical values:
            0.01  — default (conservative, avoids extreme roughness)
            0.001 — wide-H sensitivity analysis
            0.0   — no effective lower bound (H free down to near 0)
        Results are saved in a subfolder whose suffix encodes the bound:
            0.01  → <method>              (no suffix)
            0.001 → <method>_h001
            0.0   → <method>_hfree
    """
    label    = method_cfg.pop("label")
    maxiter  = method_cfg.pop("maxiter")

    # ── Determine H bounds and output-folder suffix ──────────────────────────
    # *** THIS IS THE SINGLE PLACE TO CHANGE THE H LOWER BOUND ***
    # Pass a different --h-lb value on the command line (see argparse below).
    if abs(h_lb - 1e-4) < 1e-9:
        h_bounds_override = {"H": (1e-4, 0.49)}
        suffix = "_hfree"
        logger.info("  [hfree mode] H bounds: (1e-4, 0.49)")
    elif abs(h_lb - 0.001) < 1e-9:
        h_bounds_override = {"H": (0.001, 0.49)}
        suffix = "_h001"
        logger.info("  [wide-H mode] H bounds: (0.001, 0.49)")
    else:
        h_bounds_override = None   # uses DEFAULT_BOUNDS from calibrator.py (0.01)
        suffix = ""
        logger.info("  [default H] H bounds: (0.01, 0.49)")

    method_name = method_name + suffix

    logger.info("\n" + "=" * 70)
    logger.info(f"  METHOD: {label}")
    logger.info(f"  scheme={method_cfg['scheme']}, pricing={method_cfg['pricing_method']}")
    logger.info(f"  n_paths={method_cfg['n_paths']}, n_steps={method_cfg['n_steps']}, "
                f"popsize={method_cfg['popsize']}")
    logger.info("=" * 70)

    calibrator = Calibrator(**method_cfg)
    results = []

    for i, (dt, date_label) in enumerate(snapshot_dates, 1):
        date_str = dt.strftime("%Y%m%d")
        logger.info(f"\n[{method_name} {i}/{len(snapshot_dates)}] "
                    f"{date_label} ({dt.date()})")

        surface = builder.get_iv_surface(target_time=dt, **SURFACE_CFG)
        if surface.height < MIN_SURFACE_POINTS:
            logger.warning(f"  Skip: only {surface.height} points")
            continue

        market_data = builder.export_for_rbergomi(surface)
        try:
            t0 = time.perf_counter()
            result = calibrator.calibrate(
                market_data=market_data,
                method='differential_evolution',
                maxiter=maxiter,
                tol=0.5,
                date=dt,
                bounds=h_bounds_override,
            )
            elapsed = time.perf_counter() - t0
            logger.info(f"  H={result.H:.4f}  eta={result.eta:.3f}  "
                        f"rho={result.rho:.3f}  RMSE={result.rmse:.2f}pp  "
                        f"[{elapsed:.1f}s]  {result.quality_flag}")

            row = result.to_dict() if hasattr(result, 'to_dict') else {
                "date": date_str, "H": result.H, "eta": result.eta,
                "rho": result.rho, "rmse": result.rmse, "mae": result.mae,
                "n_points": result.n_points, "n_evals": result.n_evals,
                "elapsed_s": elapsed, "label": date_label,
            }
            # Add method metadata
            row["method_name"] = method_name
            row["label"] = date_label
            save_csv(row, date_str, method_name)
            plt.close('all')
            results.append(result)

        except Exception as e:
            logger.warning(f"  Calibration FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

    logger.info(f"\n{label}: {len(results)}/{len(snapshot_dates)} successful")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="rBergomi calibration: 3 methods x event dates"
    )
    parser.add_argument("--cholesky", action="store_true",
                        help="Run Cholesky+Euler only")
    parser.add_argument("--hybrid-euler", action="store_true",
                        help="Run Hybrid+Euler only")
    parser.add_argument("--hybrid-mixed", action="store_true",
                        help="Run Hybrid+Mixed only")
    parser.add_argument("--check-iv", action="store_true",
                        help="Only verify baseline ATM-IV regime classification, then exit")
    parser.add_argument(
        "--h-lb", type=str, default="0.01",
        choices=["0.01", "0.001", "1e-4"],
        help=(
            "Lower bound for the Hurst exponent H during calibration.\n"
            "  0.01  — default, conservative (results in tables/<method>/)\n"
            "  0.001 — wide-H sensitivity  (results in tables/<method>_h001/)\n"
            "  1e-4  — minimal lower bound (results in tables/<method>_hfree/)\n"
            "The upper bound is always 0.49."
        ),
    )
    # Keep --wide-h as a legacy alias for --h-lb 0.001
    parser.add_argument("--wide-h", action="store_true",
                        help="Legacy alias for --h-lb 0.001 (kept for backward compat).")
    args = parser.parse_args()

    # Resolve h_lb value
    if args.wide_h:
        h_lb_value = 0.001
    elif args.h_lb == "1e-4":
        h_lb_value = 1e-4
    else:
        h_lb_value = float(args.h_lb)

    # If no flags, run all
    run_all = not (args.cholesky or args.hybrid_euler or args.hybrid_mixed)
    methods_to_run = []
    if run_all or args.cholesky:     methods_to_run.append("cholesky_euler")
    if run_all or args.hybrid_euler: methods_to_run.append("hybrid_euler")
    if run_all or args.hybrid_mixed: methods_to_run.append("hybrid_mixed")

    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("  rBERGOMI CALIBRATION PIPELINE (PRODUCTION)")
    logger.info(f"  Methods:  {', '.join(methods_to_run)}")
    logger.info(f"  Dates:    {len(SNAPSHOT_DATE_STRINGS)} snapshots")
    logger.info("=" * 70)

    # Check Numba availability
    try:
        import numba
        logger.info(f"  Numba {numba.__version__} detected [OK]")
    except ImportError:
        logger.warning("  Numba NOT available -- Hybrid+Mixed will be very slow!")
        logger.warning("  Install with: pip install numba")

    # 1. Build / load index
    logger.info("\n[Step 1] Loading option data index...")
    builder = IVSurfaceBuilder(str(DATA_DIR))
    builder.build_index()
    idx = builder._index if hasattr(builder, '_index') else None
    n_idx = idx.height if idx is not None else "loaded"
    logger.info(f"  Index ready: {n_idx} instruments")

    # --check-iv: verify baseline IV regime classification and exit
    if args.check_iv:
        logger.info("\n[Check] Verifying baseline ATM-IV regime classification...")
        check_baseline_atm_iv(builder)
        logger.info("Check complete. Rerun without --check-iv to start calibration.")
        sys.exit(0)

    # 2. Discover snapshot dates
    logger.info("\n[Step 2] Building snapshot date list...")
    snapshot_dates = build_snapshot_dates()
    if not snapshot_dates:
        logger.error("No snapshot dates found. Aborting.")
        sys.exit(1)

    # 3. Run each method
    all_results = {}
    for step_i, method_name in enumerate(methods_to_run, 3):
        logger.info(f"\n[Step {step_i}] Running {method_name}...")
        cfg = METHODS[method_name].copy()  # copy so pop() doesn't mutate
        all_results[method_name] = run_method(
            method_name, cfg, builder, snapshot_dates, h_lb=h_lb_value
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("  CALIBRATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Snapshot dates:   {len(snapshot_dates)}")
    for mn, res in all_results.items():
        logger.info(f"  {mn:20s}  {len(res)}/{len(snapshot_dates)} successful")
    logger.info(f"  Total time:       {elapsed_total/60:.1f} min")
    logger.info(f"\n  Results saved in:")
    for mn in methods_to_run:
        logger.info(f"    Results/calibration/tables/{mn}/")
    logger.info("=" * 70)
