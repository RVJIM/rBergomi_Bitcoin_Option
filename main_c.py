"""
================================================================================
AVAILABLE COMMANDS
================================================================================

General usage:
    python main_c.py                        # Interactive menu

Main pipeline:
    python main_c.py --all                  # Full pipeline (download -> calibration -> plots)
    python main_c.py --post-download        # Everything WITHOUT download (index -> calibration -> plots -> rBergomi)
    python main_c.py --download             # Download Deribit data only
    python main_c.py --index                # Build option index only
    python main_c.py --calibrate            # Calibration only
    python main_c.py --plots                # Generate plots only
    python main_c.py --animation            # Generate IV surface animation only
    python main_c.py --delta-test           # Quick delta plot test

Additional options:
    python main_c.py --force-rebuild        # Force index rebuild (with --index)

rBergomi Calibration:
    python main_c.py --rbergomi-test        # Quick rBergomi pricer test
    python main_c.py --rbergomi-snapshot    # Multi-date snapshot calibration (default: all 15)
    python main_c.py --rbergomi-snapshot --snapshot-category calm        # Lowest ATM IV dates (5)
    python main_c.py --rbergomi-snapshot --snapshot-category high_vol    # Highest ATM IV dates (5)
    python main_c.py --rbergomi-snapshot --snapshot-category steep_skew  # Steepest skew dates (5)
    python main_c.py --rbergomi-timeseries  # Time series calibration
    python main_c.py --rbergomi-timeseries-hv  # Time series calibration (top 25% volume)
    python main_c.py --rbergomi-events      # Market event calibration
    python main_c.py --rbergomi-plots       # Generate calibration figures
    python main_c.py --rbergomi-all         # Full rBergomi pipeline
    python main_c.py --rbergomi-compare    # Cholesky vs Hybrid side-by-side
    python main_c.py --rbergomi-both      # Snapshot calibration with both schemes

    --scheme cholesky|hybrid        # fBm scheme (default: hybrid)
    --kappa N                       # Truncation for hybrid scheme (default: 6)
    --pricing-method euler|mixed    # Pricing: log-Euler MC or Mixed Estimator (default: euler)

Bitcoin Volatility Analysis:
    python main_c.py --vol-kurtosis         # Fat tails and kurtosis analysis
    python main_c.py --vol-smile            # Volatility smile analysis
    python main_c.py --vol-term             # Volatility term structure analysis
    python main_c.py --vol-fbm              # fBm sample paths plot
    python main_c.py --vol-all              # All volatility analyses

Inverse Options Analysis:
    python main_c.py --inverse-comparison   # Greeks comparison: Direct vs Inverse
    python main_c.py --inverse-didactic     # Didactic payoff and decomposition analysis
    python main_c.py --inverse-all          # All inverse options analyses

Data Cleanup:
    python main_c.py --clean-status         # Show file/folder status
    python main_c.py --clean-temp           # Clean temporary files and cache
    python main_c.py --clean-results        # Clean obsolete Results folders
    python main_c.py --clean-all            # Full cleanup

BTC vs S&P 500 Comparison:
    python main_c.py --btc-spy              # BTC vs SPY volatility comparison


Thesis Post-Calibration Pipeline (Chapter 3):
    python main_c.py --thesis-pipeline      # Full post-calibration pipeline (check-iv + figures + tables)
    python main_c.py --check-iv             # Verify ATM IV regime labels for baseline dates
    python main_c.py --generate-figures     # Generate all figures from calibration CSVs
    python main_c.py --generate-figures --method hybrid_mixed --date 20220509   # Single method/date
    python main_c.py --populate-tables      # Generate all LaTeX .tex table files

    --method cholesky_euler|hybrid_euler|hybrid_mixed|all  # Filter for --generate-figures / analysis (default: all)
    --date YYYYMMDD                                        # Date filter for --generate-figures

Statistical Analysis:
    python main_c.py --compare-h-bounds                          # H-bound sensitivity (all methods)
    python main_c.py --compare-h-bounds --method hybrid_mixed    # Single method
    python main_c.py --residual-analysis                         # t-test + Wilcoxon signed-rank (all methods)
    python main_c.py --residual-analysis --method hybrid_euler   # Single method
    python main_c.py --compare-xi                                # Naive vs. bootstrap xi_0(t)
    python main_c.py --fix-bias                                  # Calibration bias analysis & correction

================================================================================
STRUCTURE AND DEPENDENCIES
================================================================================

main_c.py (this file)
    |
    |-- MAIN PIPELINE
    |   |-- --all             --> run_full_pipeline()
    |   |-- --post-download   --> run_post_download()
    |   |-- --download        --> deribit_data.py :: run_download_task()
    |   |-- --index           --> iv_surface_builder.py :: IVSurfaceBuilder
    |   |-- --calibrate       --> iv_surface_builder.py :: CalibrationRunner
    |   |-- --plots           --> iv_visualizer.py :: IVSurfaceVisualizer
    |   |-- --animation       --> iv_visualizer.py :: IVSurfaceVisualizer
    |   |-- --delta-test      --> iv_visualizer.py :: DeltaComparisonPlotter
    |
    |-- BITCOIN VOLATILITY ANALYSIS --> btc_volatility_analysis.py
    |   |-- --vol-kurtosis    --> run_kurtosis_analysis()
    |   |-- --vol-smile       --> run_smile_analysis()
    |   |-- --vol-term        --> run_term_structure_analysis()
    |   |-- --vol-fbm         --> run_fbm_paths()
    |   |-- --vol-all         --> run_all_volatility_analysis()
    |
    |-- INVERSE OPTIONS ANALYSIS --> inverse_options.py
    |   |-- --inverse-comparison --> run_comparison_analysis()
    |   |-- --inverse-didactic   --> run_didactic_analysis()
    |   |-- --inverse-all        --> run_all_inverse_analysis()
    |
    |-- rBERGOMI CALIBRATION --> src/rbergomi/ (pricer, calibrator, visualizer, utils)
    |   |-- --rbergomi-test          --> run_rbergomi_test()
    |   |-- --rbergomi-snapshot      --> run_rbergomi_snapshot()
    |   |-- --rbergomi-timeseries    --> run_rbergomi_timeseries()
    |   |-- --rbergomi-timeseries-hv --> run_rbergomi_timeseries_highvol()
    |   |-- --rbergomi-events        --> run_rbergomi_events()
    |   |-- --rbergomi-plots         --> run_rbergomi_plots()
    |   |-- --rbergomi-all           --> run_rbergomi_all()
    |   |-- --rbergomi-compare       --> run_rbergomi_compare()
    |   |-- --scheme cholesky|hybrid        (fBm scheme selection)
    |   |-- --kappa N                       (truncation for hybrid scheme)
    |   |-- --pricing-method euler|mixed    (pricing method selection)
    |
    |-- BTC vs S&P 500 COMPARISON --> btc_spy_volatility/
    |   |-- --btc-spy             --> btc_vs_sp.py :: run_btc_spy_comparison()

    |
    |-- THESIS POST-CALIBRATION (Chapter 3)
    |   |-- --thesis-pipeline     --> _run_thesis_pipeline()  [check-iv + figures + tables]
    |   |-- --check-iv            --> run_all_calibrations.py :: check_baseline_atm_iv()
    |   |-- --generate-figures    --> generate_figures.py     [--method, --date]
    |   |-- --populate-tables     --> populate_tables.py
    |
    |-- DATA CLEANUP --> data_cleaning.py
    |   |-- --clean-status        --> run_show_status()
    |   |-- --clean-temp          --> run_clean_temp()
    |   |-- --clean-results       --> run_clean_results()
    |   |-- --clean-all           --> run_clean_all()
    |
    |-- STATISTICAL ANALYSIS
        |-- --compare-h-bounds    --> compare_h_bounds.py  [--method]
        |-- --residual-analysis   --> residual_analysis.py [--method]
        |-- --compare-xi          --> compare_xi_approaches.py [--method]
        |-- --fix-bias            --> fix_calibration_bias.py [--method]

GENERATED OUTPUT (in Results/):
    - fat_tails_kurtosis/        <-- btc_volatility_analysis.py (--vol-kurtosis)
    - implied_volatility_smile/  <-- btc_volatility_analysis.py (--vol-smile)
    - volatility_term_structure/ <-- btc_volatility_analysis.py (--vol-term)
    - inverse_options/comparison/<-- inverse_options.py (--inverse-comparison)
    - inverse_options/didactic/  <-- inverse_options.py (--inverse-didactic)
    - calibration/               <-- src/rbergomi/ (--rbergomi-*)

================================================================================
"""

import asyncio
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
import logging
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# === GLOBAL CONFIGURATION ===
CONFIG = {
    # Thesis dates (Deribit data available from 2022-01-01)
    "start_date": datetime(2022, 1, 1, 0, 0, 0),
    "end_date": datetime(2025, 12, 31, 23, 59, 59),

    # Paths
    "data_folder": "data",
    "output_folder": "calibration_results",
    "figures_folder": "figures",

    # Download
    "concurrent_workers": 15,

    # IV Surface Calibration (legacy)
    "calibration": {
        "window_hours": 2.0,
        "min_volume": 0.1,          # Minimum volume in BTC
        "moneyness_range": (0.7, 1.4),
        "min_ttm_days": 7,
        "max_ttm_days": 180,
        "min_points": 20,           # Minimum points for valid calibration
        "frequency": "daily"        # daily, weekly, monthly
    },

    # rBergomi Calibration
    "rbergomi": {
        # Monte Carlo settings
        "n_paths": 10_000,          # MC paths for Nelder-Mead refinement (fine phase)
        "n_paths_coarse": 2_000,    # MC paths for Differential Evolution (coarse phase, ~5x faster)
        "n_steps": 50,              # Time steps
        "seed": 42,                 # Reproducibility

        # Optimizer settings
        "method": "differential_evolution",  # or 'nelder-mead'
        "maxiter": 25,              # Max optimizer iterations
        "tol": 1.0,                 # RMSE tolerance (pp)

        # Surface extraction
        "window_hours": 4.0,
        "min_volume": 0.05,
        "moneyness_range": (0.8, 1.2),  # Focus on ATM +/- 20%
        "min_ttm_days": 7,
        "max_ttm_days": 90,
        "min_points": 15,

        # Time series
        "frequency": "weekly",      # weekly, monthly

        # fBm simulation scheme: 'cholesky' or 'hybrid'
        "scheme": "hybrid",
        "kappa": 6,              # Truncation parameter for hybrid scheme

        # Pricing method: 'euler' (log-Euler spot MC) or 'mixed' (Mixed Estimator / Turbocharging)
        "pricing_method": "euler",

        # Initial guess (typical BTC values)
        "initial_guess": {
            "H": 0.07,
            "eta": 1.5,
            "rho": -0.7,
            "xi": 0.04
        },

        # Output
        "output_dir": "Results/calibration"
    },

    # Animation
    "animation": {
        "start_date": datetime(2024, 1, 1),
        "end_date": datetime(2024, 12, 31),
        "interval_days": 7,
        "fps": 2
    }
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Results/thesis_pipeline.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ThesisMain")


# === PIPELINE FUNCTIONS ===

async def run_download():
    """Download Deribit data."""
    logger.info("="*60)
    logger.info("PHASE 1: DOWNLOAD DERIBIT DATA")
    logger.info("="*60)

    from deribit_data import run_download_task, postprocess_add_expiration

    logger.info(f"Period: {CONFIG['start_date']} -> {CONFIG['end_date']}")

    # Download data
    await run_download_task(
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
        data_folder=CONFIG["data_folder"],
        max_concurrent=CONFIG["concurrent_workers"]
    )

    # Post-processing: add expiration_timestamp to existing files
    logger.info("="*60)
    logger.info("POST-PROCESSING: Adding expiration_timestamp")
    logger.info("="*60)
    postprocess_add_expiration(CONFIG["data_folder"])

    logger.info("Download and post-processing completed.")


def run_index_build(force_rebuild: bool = False):
    """Build the parquet file index."""
    logger.info("="*60)
    logger.info("PHASE 2: BUILD INDEX")
    logger.info("="*60)
    
    from iv_surface_builder import IVSurfaceBuilder
    
    data_dir = Path(CONFIG["data_folder"]) / "option"
    builder = IVSurfaceBuilder(str(data_dir))
    index = builder.build_index(force_rebuild=force_rebuild)
    
    logger.info(f"Index built: {index.height} instruments")

    # Statistics
    if index.height > 0:
        logger.info(f"  - Strike range: ${index['strike'].min():,.0f} - ${index['strike'].max():,.0f}")
        logger.info(f"  - Total trades: {index['n_trades'].sum():,}")
        logger.info(f"  - Calls: {index.filter(index['option_type'] == 'C').height}")
        logger.info(f"  - Puts: {index.filter(index['option_type'] == 'P').height}")
    
    return builder


def run_calibration(builder=None):
    """Run daily and event calibration."""
    logger.info("="*60)
    logger.info("PHASE 3: CALIBRATION")
    logger.info("="*60)
    
    from iv_surface_builder import IVSurfaceBuilder, CalibrationRunner
    
    if builder is None:
        data_dir = Path(CONFIG["data_folder"]) / "option"
        builder = IVSurfaceBuilder(str(data_dir))
        builder.build_index()
    
    runner = CalibrationRunner(builder, CONFIG["output_folder"])
    
    # 3.1 Daily calibration
    logger.info("Daily calibration...")
    daily_results = runner.run_daily_calibration(
        start=CONFIG["start_date"],
        end=CONFIG["end_date"],
        min_points=CONFIG["calibration"]["min_points"]
    )
    logger.info(f"  - Valid dates: {daily_results.height}")

    # 3.2 Event calibration
    logger.info("Market event calibration...")
    event_results = runner.run_event_calibration()
    logger.info(f"  - Event observations: {event_results.height}")
    
    return daily_results, event_results


def run_plots(builder=None):
    """Generate all thesis plots."""
    logger.info("="*60)
    logger.info("PHASE 4: PLOT GENERATION")
    logger.info("="*60)
    
    from iv_surface_builder import IVSurfaceBuilder, MARKET_EVENTS
    from iv_visualizer import IVSurfaceVisualizer, DeltaComparisonPlotter
    import polars as pl
    
    figures_dir = Path(CONFIG["figures_folder"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 4.1 Delta plots (explicitly requested)
    # First get realistic parameters from data
    logger.info("Extracting real parameters from data for Delta plots...")
    
    if builder is None:
        data_dir = Path(CONFIG["data_folder"]) / "option"
        builder = IVSurfaceBuilder(str(data_dir))
        builder.build_index()
    
    # Extract a recent surface to get realistic parameters
    # Try multiple dates to find one with data
    sample_dates = [
        datetime(2024, 3, 28, 12),   # High liquidity
        datetime(2024, 6, 15, 12),
        datetime(2023, 6, 15, 12),
        datetime(2022, 6, 13, 12),   # High volume
    ]

    sample_surface = None
    for sample_date in sample_dates:
        sample_surface = builder.get_iv_surface(
            target_time=sample_date,
            window_hours=4.0,
            min_volume=0.05,  # Lowered to find more data
            moneyness_range=(0.8, 1.2),
            min_ttm_days=7,
            max_ttm_days=60
        )
        if sample_surface.height > 0:
            logger.info(f"Using sample date: {sample_date.date()}")
            break
    
    # Parameters derived from real data (with fallback if no data)
    if sample_surface.height > 0:
        spot_real = sample_surface["spot"].mean()
        iv_real = sample_surface["iv"].mean() / 100  # From percentage to decimal
        # ATM strike (closest to moneyness = 1)
        atm_row = sample_surface.filter(
            (pl.col("moneyness") > 0.98) & (pl.col("moneyness") < 1.02)
        )
        if atm_row.height > 0:
            K_real = atm_row["strike"].mean()
        else:
            K_real = spot_real  # Fallback: ATM
        logger.info(f"Parameters from data: Spot=${spot_real:,.0f}, K=${K_real:,.0f}, σ={iv_real*100:.1f}%")
    else:
        # Fallback with reasonable values if no data available
        spot_real = 65_000
        K_real = 65_000
        iv_real = 0.55
        logger.warning("No data available, using fallback parameters")
    
    # r = 0 for crypto (no traditional risk-free rate; could use funding rate)
    r_crypto = 0.0
    T_30d = 30/365
    
    delta_plotter = DeltaComparisonPlotter(str(figures_dir))
    
    # S range centered on real spot (±40%)
    S_range_real = (spot_real * 0.6, spot_real * 1.4)
    
    delta_plotter.plot_delta_comparison(
        K=K_real, T=T_30d, r=r_crypto, sigma=iv_real,
        S_range=S_range_real,
        option_type="put",
        save_name="fig_delta_inverse_vs_traditional_put",
        show=False
    )
    
    delta_plotter.plot_delta_comparison(
        K=K_real, T=T_30d, r=r_crypto, sigma=iv_real,
        S_range=S_range_real,
        option_type="call",
        save_name="fig_delta_inverse_vs_traditional_call",
        show=False
    )
    
    delta_plotter.plot_all_greeks_comparison(
        K=K_real, T=T_30d, r=r_crypto, sigma=iv_real,
        S_range=S_range_real,
        save_name="fig_greeks_panel_complete",
        show=False
    )
    
    # 4.2 IV Surfaces per date chiave
    if builder is None:
        data_dir = Path(CONFIG["data_folder"]) / "option"
        builder = IVSurfaceBuilder(str(data_dir))
        builder.build_index()
    
    visualizer = IVSurfaceVisualizer(str(figures_dir))
    
    key_dates = [
        (datetime(2020, 3, 12, 12), "COVID Black Thursday"),
        (datetime(2021, 11, 10, 12), "BTC ATH $69k"),
        (datetime(2022, 11, 8, 12), "FTX Collapse"),
        (datetime(2024, 1, 10, 12), "ETF Approval"),
        (datetime(2024, 4, 20, 12), "Fourth Halving"),
        (datetime(2025, 1, 20, 12), "Trump Crypto EO"),
    ]
    
    for date, event_name in key_dates:
        logger.info(f"Generating IV surface for {event_name}...")
        
        surface = builder.get_iv_surface(
            target_time=date,
            window_hours=4.0,
            min_volume=0.05,
            moneyness_range=(0.6, 1.5),
            min_ttm_days=5,
            max_ttm_days=120
        )
        
        if surface.height >= 15:
            safe_name = event_name.lower().replace(" ", "_").replace("$", "")[:25]
            
            visualizer.plot_iv_surface_3d(
                surface,
                title=f"IV Surface - {event_name}",
                save_name=f"fig_surface_{safe_name}",
                show=False
            )
            
            visualizer.plot_iv_smile(
                surface,
                ttm_target=30/365,
                title=f"Volatility Smile - {event_name}",
                save_name=f"fig_smile_{safe_name}",
                show=False
            )
    
    # 4.3 Time series
    output_dir = Path(CONFIG["output_folder"])
    daily_file = output_dir / "daily_calibration.parquet"
    
    if daily_file.exists():
        logger.info("Generating time series...")
        daily_results = pl.read_parquet(daily_file)
        
        visualizer.plot_calibration_timeseries(
            daily_results,
            param="iv_atm_mean",
            title="Bitcoin ATM Implied Volatility (2020-2025)",
            save_name="fig_iv_timeseries_full",
            show=False,
            events=MARKET_EVENTS
        )
        
        visualizer.plot_calibration_timeseries(
            daily_results,
            param="spot",
            title="Bitcoin Spot Price (2020-2025)",
            save_name="fig_spot_timeseries_full",
            show=False,
            events=MARKET_EVENTS
        )
    
    # 4.4 Event comparisons
    event_file = output_dir / "event_calibration.parquet"
    
    if event_file.exists():
        logger.info("Generating event plots...")
        event_data = pl.read_parquet(event_file)
        
        for event in MARKET_EVENTS:
            safe_name = event["name"].lower().replace(" ", "_").replace("/", "_")[:25]
            visualizer.plot_event_comparison(
                event_data,
                event["name"],
                save_name=f"fig_event_{safe_name}",
                show=False
            )
    
    logger.info(f"Plots saved in {figures_dir}/")


def run_animation(builder=None):
    """Generate IV surface evolution GIFs (per-year + combined)."""
    logger.info("="*60)
    logger.info("PHASE 5: ANIMATION GENERATION (GIFs)")
    logger.info("="*60)

    from iv_surface_builder import IVSurfaceBuilder
    from iv_visualizer import IVSurfaceVisualizer

    if builder is None:
        data_dir = Path(CONFIG["data_folder"]) / "option"
        builder = IVSurfaceBuilder(str(data_dir))
        builder.build_index()

    visualizer = IVSurfaceVisualizer(CONFIG["figures_folder"])

    anim_cfg = CONFIG["animation"]
    interval_days = anim_cfg["interval_days"]
    fps = anim_cfg["fps"]

    # Per-year GIFs (2022-2025)
    for year in range(2022, 2026):
        logger.info(f"Generating GIF for {year}...")
        output_path = visualizer.create_surface_animation(
            builder=builder,
            start_date=datetime(year, 1, 1),
            end_date=datetime(year, 12, 31),
            interval_days=interval_days,
            fps=fps,
            save_name=f"iv_surface_{year}",
            show_scatter=False
        )
        if output_path:
            logger.info(f"  Saved: {output_path}")
        else:
            logger.warning(f"  No animation generated for {year}")

    # Combined GIF (all years)
    logger.info("Generating combined GIF (2022-2025)...")
    output_path = visualizer.create_surface_animation(
        builder=builder,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2025, 12, 31),
        interval_days=interval_days,
        fps=fps,
        save_name="iv_surface_all_years",
        show_scatter=False
    )
    if output_path:
        logger.info(f"  Saved: {output_path}")
    else:
        logger.warning("  No animation generated for combined years")

    logger.info("Animation generation complete.")


# === rBERGOMI FUNCTIONS ===

def run_rbergomi_test():
    """Quick rBergomi pricer test with Euler and Mixed Estimator."""
    logger.info("="*60)
    logger.info("rBERGOMI - TEST PRICER")
    logger.info("="*60)

    from src.rbergomi import rBergomiPricer, bs_price_inverse_put, CYTHON_AVAILABLE
    import time

    cfg = CONFIG["rbergomi"]

    # Check Cython availability
    print(f"\nCython optimization available: {CYTHON_AVAILABLE}")

    # Parameters
    H, eta, rho, xi = 0.07, 1.9, -0.7, 0.04
    S0, K, T = 50_000, 48_000, 30/365
    scheme = cfg["scheme"]
    kappa = cfg["kappa"]
    pricing_method = cfg["pricing_method"]

    print(f"\nModel parameters:")
    print(f"  H={H}, eta={eta}, rho={rho}, xi={xi}")
    print(f"  S0=${S0:,}, K=${K:,}, T={T*365:.0f} days")
    print(f"  scheme={scheme}" + (f", kappa={kappa}" if scheme == "hybrid" else ""))
    print(f"  pricing_method={pricing_method}")

    # Compare to BS
    bs_price = bs_price_inverse_put(S0, K, T, 0, xi**0.5)

    # Test pricing methods
    methods_to_test = [pricing_method]
    # If hybrid scheme and only one method selected, test both for comparison
    if scheme == "hybrid" and pricing_method != "mixed":
        methods_to_test = ["euler", "mixed"]
    elif scheme == "hybrid" and pricing_method == "mixed":
        methods_to_test = ["euler", "mixed"]

    for method in methods_to_test:
        if method == "mixed" and scheme != "hybrid":
            print(f"\n--- Pricing: {method} (skipped, requires hybrid scheme) ---")
            continue

        print(f"\n--- Scheme: {scheme}, Pricing: {method} ---")
        pricer = rBergomiPricer(
            H, eta, rho, xi, n_paths=20_000, n_steps=50,
            scheme=scheme, kappa=kappa, seed=42,
            pricing_method=method
        )
        print(f"  {pricer}")

        start = time.perf_counter()
        price, std_err = pricer.price_inverse_put(S0, K, T)
        elapsed = time.perf_counter() - start

        iv = pricer.implied_vol_inverse_put(S0, K, T)

        print(f"  rBergomi price: {price:.6f} BTC (+/- {std_err:.6f})")
        print(f"  BS price:       {bs_price:.6f} BTC")
        print(f"  Model IV:       {iv*100:.2f}%")
        print(f"  Time:           {elapsed*1000:.1f} ms")
        if method == "mixed":
            print(f"  (Mixed Estimator: conditional BS, no spot-path MC noise)")
        print(f"\n{pricer.get_performance_summary()}")

    logger.info("Test completed successfully!")


# ---------------------------------------------------------------------------
# DATA-DRIVEN SNAPSHOT DATE SELECTION
# ---------------------------------------------------------------------------
#
# Categories:
#   1. Calm       — lowest ATM IV  (model baseline in normal regime)
#   2. High vol   — highest ATM IV (stress test under extreme levels)
#   3. Steep skew — largest |IV_OTM_put - IV_OTM_call| / IV_ATM
#                   (hardest test for rBergomi's roughness-driven skew)
#
# Quality filters (per snapshot surface):
#   - >= 15 surface points  (strike x maturity combinations)
#   - >= 30 total trades    (VWAP IV stability)
#   - >= 3 distinct maturities (term structure needed to identify H)
#
# These thresholds ensure the surface is rich enough to constrain 3 params
# (H, eta, rho). The 15-point minimum gives ~5x overidentification;
# 30 trades keeps VWAP noise low (SE ~ sigma/sqrt(n)); 3 maturities let
# the optimizer disentangle H (term structure) from eta (vol-of-vol level).
# ---------------------------------------------------------------------------

MIN_SURFACE_POINTS = 15
MIN_TOTAL_TRADES = 30
MIN_MATURITIES = 3
N_DATES_PER_CATEGORY = 5


def _compute_surface_metrics(surface: pl.DataFrame) -> dict:
    """
    Compute ATM IV and skew metric from an IV surface.

    Returns dict with keys: atm_iv, skew, n_points, n_trades, n_maturities,
    or None if surface doesn't pass quality filters.
    """
    if surface.height < MIN_SURFACE_POINTS:
        return None

    n_trades = int(surface["total_trades"].sum()) if "total_trades" in surface.columns else surface.height
    if n_trades < MIN_TOTAL_TRADES:
        return None

    # Count distinct maturities
    n_mat = surface["ttm"].n_unique()
    if n_mat < MIN_MATURITIES:
        return None

    moneyness = surface["moneyness"].to_numpy()
    iv = surface["iv"].to_numpy()  # in % (e.g. 80 = 80%)

    # ATM IV: points with moneyness closest to 1.0 (within 5%)
    atm_mask = np.abs(moneyness - 1.0) < 0.05
    if atm_mask.sum() == 0:
        # Fallback: closest 3 points to ATM
        closest_idx = np.argsort(np.abs(moneyness - 1.0))[:3]
        atm_iv = float(np.mean(iv[closest_idx]))
    else:
        atm_iv = float(np.mean(iv[atm_mask]))

    # Skew: (mean IV of OTM puts) - (mean IV of OTM calls), normalized
    # OTM puts: moneyness < 0.95,  OTM calls: moneyness > 1.05
    otm_put_mask = moneyness < 0.95
    otm_call_mask = moneyness > 1.05

    if otm_put_mask.sum() >= 2 and otm_call_mask.sum() >= 2 and atm_iv > 0:
        skew = (float(np.mean(iv[otm_put_mask])) - float(np.mean(iv[otm_call_mask]))) / atm_iv
    else:
        skew = 0.0

    return {
        "atm_iv": atm_iv,
        "skew": skew,
        "n_points": surface.height,
        "n_trades": n_trades,
        "n_maturities": n_mat,
    }


def discover_snapshot_dates(
    builder,
    start: datetime = None,
    end: datetime = None,
    n_per_category: int = N_DATES_PER_CATEGORY,
    window_hours: float = 4.0,
    moneyness_range: tuple = (0.8, 1.2),
    min_ttm_days: int = 7,
    max_ttm_days: int = 90,
) -> dict:
    """
    Scan weekly dates and select the best N per category based on data.

    Returns dict with keys 'calm', 'high_vol', 'steep_skew', each a list
    of (datetime, label) tuples sorted chronologically.
    Also saves the full scan table to Results/calibration/snapshot_scan/.
    """
    if start is None:
        start = CONFIG["start_date"]
    if end is None:
        end = CONFIG["end_date"]

    # Generate weekly candidate dates (Saturday 12:00 UTC — peak Deribit liquidity)
    candidates = []
    current = start
    while current <= end:
        candidates.append(current.replace(hour=12, minute=0, second=0))
        current += timedelta(days=7)

    logger.info(f"Scanning {len(candidates)} weekly dates for snapshot selection...")

    scan_rows = []
    for i, dt in enumerate(candidates):
        if (i + 1) % 20 == 0:
            logger.info(f"  Scanned {i+1}/{len(candidates)}...")

        surface = builder.get_iv_surface(
            target_time=dt,
            window_hours=window_hours,
            moneyness_range=moneyness_range,
            min_ttm_days=min_ttm_days,
            max_ttm_days=max_ttm_days,
        )

        metrics = _compute_surface_metrics(surface)
        if metrics is None:
            continue

        scan_rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "datetime": dt,
            **metrics,
        })

    if not scan_rows:
        logger.error("No valid dates found during scan!")
        return {"calm": [], "high_vol": [], "steep_skew": []}

    logger.info(f"Scan complete: {len(scan_rows)}/{len(candidates)} dates passed quality filters")

    # Save scan table
    scan_dir = Path("Results/calibration/snapshot_scan")
    scan_dir.mkdir(parents=True, exist_ok=True)
    df_scan = pl.DataFrame([{k: v for k, v in r.items() if k != "datetime"} for r in scan_rows])
    df_scan.write_csv(scan_dir / "snapshot_scan_all.csv")
    logger.info(f"Full scan table saved to {scan_dir / 'snapshot_scan_all.csv'}")

    # --- Rank and select ---
    # 1. Calm: lowest ATM IV
    by_atm = sorted(scan_rows, key=lambda r: r["atm_iv"])
    calm = [(r["datetime"], f"Calm — ATM IV {r['atm_iv']:.1f}%") for r in by_atm[:n_per_category]]

    # 2. High vol: highest ATM IV
    high_vol = [(r["datetime"], f"High vol — ATM IV {r['atm_iv']:.1f}%") for r in by_atm[-n_per_category:]]
    high_vol.reverse()  # highest first

    # 3. Steep skew: largest absolute normalized skew
    by_skew = sorted(scan_rows, key=lambda r: abs(r["skew"]), reverse=True)
    steep_skew = [(r["datetime"], f"Steep skew — skew {r['skew']:+.3f}") for r in by_skew[:n_per_category]]

    # Sort each category chronologically
    for cat in [calm, high_vol, steep_skew]:
        cat.sort(key=lambda x: x[0])

    result = {"calm": calm, "high_vol": high_vol, "steep_skew": steep_skew}

    # Print summary
    for cat_name, dates in result.items():
        logger.info(f"\n  {cat_name.upper()} dates:")
        for dt, label in dates:
            logger.info(f"    {dt.strftime('%Y-%m-%d')}  {label}")

    return result


def _select_snapshot_dates(builder) -> list:
    """Let the user pick which snapshot dates to calibrate (data-driven)."""
    print("\n  Discovering best snapshot dates from data...")
    cfg = CONFIG["rbergomi"]
    categories = discover_snapshot_dates(
        builder,
        window_hours=cfg["window_hours"],
        moneyness_range=cfg["moneyness_range"],
        min_ttm_days=cfg["min_ttm_days"],
        max_ttm_days=cfg["max_ttm_days"],
    )

    all_dates = categories["calm"] + categories["high_vol"] + categories["steep_skew"]
    # Deduplicate (a date could appear in multiple categories)
    seen = set()
    unique_all = []
    for dt, label in all_dates:
        key = dt.strftime("%Y-%m-%d")
        if key not in seen:
            seen.add(key)
            unique_all.append((dt, label))
    unique_all.sort(key=lambda x: x[0])

    print(f"\n  Found {len(unique_all)} unique dates across 3 categories:\n")
    for cat_name, dates in categories.items():
        display = cat_name.replace("_", " ").title()
        print(f"    {display} ({len(dates)} dates):")
        for dt, label in dates:
            print(f"      {dt.strftime('%Y-%m-%d')}  {label}")
        print()

    print("  Select category:")
    print("    1. Calm / low ATM IV           (5 dates)")
    print("    2. High volatility / high ATM IV (5 dates)")
    print("    3. Steepest skew               (5 dates)")
    print(f"    4. All unique dates             ({len(unique_all)} dates)")
    print("    5. Single custom date")
    sel = input("  Select [1-5]: ").strip()

    if sel == "1":
        return categories["calm"]
    elif sel == "2":
        return categories["high_vol"]
    elif sel == "3":
        return categories["steep_skew"]
    elif sel == "4":
        return unique_all
    elif sel == "5":
        raw = input("  Enter date (YYYY-MM-DD): ").strip()
        try:
            dt = datetime.strptime(raw, "%Y-%m-%d").replace(hour=12)
            return [(dt, "custom")]
        except ValueError:
            print("  Invalid date format, falling back to calm dates.")
            return categories["calm"]
    else:
        print("  Invalid choice, falling back to calm dates.")
        return categories["calm"]


def run_rbergomi_snapshot(target_dates: list = None, builder=None):
    """rBergomi calibration on one or more snapshot dates."""
    logger.info("="*60)
    logger.info("rBERGOMI - SNAPSHOT CALIBRATION")
    logger.info("="*60)

    from iv_surface_builder import IVSurfaceBuilder
    from iv_visualizer import IVSurfaceVisualizer
    from src.rbergomi import Calibrator, CalibrationVisualizer

    cfg = CONFIG["rbergomi"]

    # Build index once
    if builder is None:
        data_dir = Path(CONFIG["data_folder"]) / "option"
        builder = IVSurfaceBuilder(str(data_dir))
        builder.build_index()

    # If no dates provided, discover from data
    if target_dates is None:
        categories = discover_snapshot_dates(
            builder,
            window_hours=cfg["window_hours"],
            moneyness_range=cfg["moneyness_range"],
            min_ttm_days=cfg["min_ttm_days"],
            max_ttm_days=cfg["max_ttm_days"],
        )
        target_dates = categories["calm"] + categories["high_vol"] + categories["steep_skew"]
        # Deduplicate
        seen = set()
        unique = []
        for dt, label in target_dates:
            key = dt.strftime("%Y-%m-%d")
            if key not in seen:
                seen.add(key)
                unique.append((dt, label))
        unique.sort(key=lambda x: x[0])
        target_dates = unique

    # Support legacy single-datetime call
    if isinstance(target_dates, datetime):
        target_dates = [(target_dates, "custom")]

    logger.info(f"Snapshot dates to calibrate: {len(target_dates)}")

    # Create calibrator
    calibrator = Calibrator(
        n_paths=cfg["n_paths"],
        n_paths_coarse=cfg["n_paths_coarse"],
        n_steps=cfg["n_steps"],
        seed=cfg["seed"],
        scheme=cfg["scheme"],
        kappa=cfg["kappa"],
        pricing_method=cfg["pricing_method"]
    )

    scheme = cfg["scheme"]
    pricing_method = cfg["pricing_method"]
    pipeline_tag = f"{scheme}_{pricing_method}"
    cal_viz = CalibrationVisualizer(cfg["output_dir"] + "/figures", scheme=scheme,
                                    pricing_method=pricing_method)
    iv_viz = IVSurfaceVisualizer(cfg["output_dir"] + "/figures")

    # Output directories for IV data
    iv_tables_dir = Path(cfg["output_dir"]) / "snapshot_surfaces"
    iv_tables_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (target_date, label) in enumerate(target_dates, 1):
        date_str = target_date.strftime("%Y%m%d")
        logger.info(f"\n[{i}/{len(target_dates)}] {label} — {target_date.strftime('%Y-%m-%d')}")

        # --- Extract IV surface ---
        surface = builder.get_iv_surface(
            target_time=target_date,
            window_hours=cfg["window_hours"],
            moneyness_range=cfg["moneyness_range"],
            min_ttm_days=cfg["min_ttm_days"],
            max_ttm_days=cfg["max_ttm_days"]
        )

        if surface.height < MIN_SURFACE_POINTS:
            logger.warning(f"  Skipping {target_date.date()}: only {surface.height} points "
                           f"(need >= {MIN_SURFACE_POINTS})")
            continue

        # --- Save IV surface table ---
        surface.write_csv(iv_tables_dir / f"iv_surface_{date_str}.csv")
        logger.info(f"  IV surface saved: {surface.height} points")

        # --- IV figures: smile (2D) + surface (3D) ---
        safe_label = label.split("—")[0].strip().lower().replace(" ", "_")[:20]

        iv_viz.plot_iv_surface_3d(
            surface,
            title=f"IV Surface — {target_date.strftime('%Y-%m-%d')}\n{label}",
            save_name=f"iv_surface_3d_{date_str}_{pipeline_tag}",
            show=False
        )

        iv_viz.plot_iv_smile(
            surface,
            ttm_target=30 / 365,
            title=f"Volatility Smile — {target_date.strftime('%Y-%m-%d')}",
            save_name=f"iv_smile_{date_str}_{pipeline_tag}",
            show=False
        )
        plt.close('all')

        # --- Calibration ---
        try:
            market_data = builder.export_for_rbergomi(surface)

            result = calibrator.calibrate_snapshot(
                builder=builder,
                target_date=target_date,
                method=cfg["method"],
                maxiter=cfg["maxiter"],
                window_hours=cfg["window_hours"],
                moneyness_range=cfg["moneyness_range"],
                min_ttm_days=cfg["min_ttm_days"],
                max_ttm_days=cfg["max_ttm_days"],
                save=True
            )
            print(f"\n{result}")
            results.append(result)

            # Calibration-specific plots
            cal_viz.plot_skew_reproduction(result, market_data,
                                           save_name=f"skew_{date_str}")
            cal_viz.plot_residuals_heatmap(result,
                                           save_name=f"residuals_heatmap_{date_str}")
            cal_viz.plot_residuals_histogram(result,
                                             save_name=f"residuals_hist_{date_str}")
            plt.close('all')

        except Exception as e:
            logger.warning(f"  Calibration for {target_date.date()} failed: {e}")
            continue

    logger.info(f"\nSnapshot calibration complete: {len(results)}/{len(target_dates)} succeeded")
    logger.info(f"Output saved to {cfg['output_dir']}")
    return results


def run_rbergomi_timeseries():
    """rBergomi time series calibration."""
    logger.info("="*60)
    logger.info("rBERGOMI - TIME SERIES CALIBRATION")
    logger.info("="*60)

    from iv_surface_builder import IVSurfaceBuilder
    from src.rbergomi import Calibrator, CalibrationVisualizer

    cfg = CONFIG["rbergomi"]

    # Build index
    data_dir = Path(CONFIG["data_folder"]) / "option"
    builder = IVSurfaceBuilder(str(data_dir))
    builder.build_index()

    # Get calibration dates
    frequency = cfg.get("frequency", "weekly")
    dates = builder.get_calibration_dates(
        start=CONFIG["start_date"],
        end=CONFIG["end_date"],
        frequency=frequency
    )

    logger.info(f"Calibrating {len(dates)} dates ({frequency})")

    # Create calibrator
    calibrator = Calibrator(
        n_paths=cfg["n_paths"],
        n_paths_coarse=cfg["n_paths_coarse"],
        n_steps=cfg["n_steps"],
        seed=cfg["seed"],
        scheme=cfg["scheme"],
        kappa=cfg["kappa"],
        pricing_method=cfg["pricing_method"]
    )

    # Run time series calibration
    results_df = calibrator.calibrate_timeseries(
        builder=builder,
        dates=dates,
        method=cfg["method"],
        maxiter=cfg["maxiter"],
        tol=cfg["tol"],
        window_hours=cfg["window_hours"],
        min_points=cfg["min_points"],
        moneyness_range=cfg["moneyness_range"],
        min_ttm_days=cfg["min_ttm_days"],
        max_ttm_days=cfg["max_ttm_days"],
        save_intermediate=True
    )

    # Generate visualizations
    if results_df.height > 0:
        from iv_surface_builder import MARKET_EVENTS
        viz = CalibrationVisualizer(cfg["output_dir"] + "/figures", scheme=cfg["scheme"],
                                       pricing_method=cfg["pricing_method"])
        viz.plot_parameter_stability(results_df, events=MARKET_EVENTS)
        viz.generate_calibration_table(results_df)
        viz.generate_summary_statistics(results_df)

    logger.info(f"Output saved to {cfg['output_dir']}")
    return results_df


def run_rbergomi_timeseries_highvol():
    """rBergomi time series calibration on top 25% volume dates."""
    logger.info("="*60)
    logger.info("rBERGOMI - TIME SERIES CALIBRATION (TOP 25% VOLUME)")
    logger.info("="*60)

    from iv_surface_builder import IVSurfaceBuilder
    from src.rbergomi import Calibrator, CalibrationVisualizer

    cfg = CONFIG["rbergomi"]

    # Build index
    data_dir = Path(CONFIG["data_folder"]) / "option"
    builder = IVSurfaceBuilder(str(data_dir))
    builder.build_index()

    # Get all candidate dates
    frequency = cfg.get("frequency", "weekly")
    all_dates = builder.get_calibration_dates(
        start=CONFIG["start_date"],
        end=CONFIG["end_date"],
        frequency=frequency
    )

    # Filter to top 25% by volume
    dates = builder.get_high_volume_dates(all_dates, top_pct=0.25)
    logger.info(f"Selected {len(dates)}/{len(all_dates)} high-volume dates ({frequency})")

    # Create calibrator
    calibrator = Calibrator(
        n_paths=cfg["n_paths"],
        n_paths_coarse=cfg["n_paths_coarse"],
        n_steps=cfg["n_steps"],
        seed=cfg["seed"],
        scheme=cfg["scheme"],
        kappa=cfg["kappa"],
        pricing_method=cfg["pricing_method"]
    )

    # Run time series calibration
    results_df = calibrator.calibrate_timeseries(
        builder=builder,
        dates=dates,
        method=cfg["method"],
        maxiter=cfg["maxiter"],
        tol=cfg["tol"],
        window_hours=cfg["window_hours"],
        min_points=cfg["min_points"],
        moneyness_range=cfg["moneyness_range"],
        min_ttm_days=cfg["min_ttm_days"],
        max_ttm_days=cfg["max_ttm_days"],
        save_intermediate=True
    )

    # Generate visualizations
    if results_df.height > 0:
        from iv_surface_builder import MARKET_EVENTS
        viz = CalibrationVisualizer(cfg["output_dir"] + "/figures", scheme=cfg["scheme"],
                                       pricing_method=cfg["pricing_method"])
        viz.plot_parameter_stability(results_df, events=MARKET_EVENTS)
        viz.generate_calibration_table(results_df)
        viz.generate_summary_statistics(results_df)

    logger.info(f"Output saved to {cfg['output_dir']}")
    return results_df


def run_rbergomi_events():
    """rBergomi market event calibration."""
    logger.info("="*60)
    logger.info("rBERGOMI - EVENT CALIBRATION")
    logger.info("="*60)

    from iv_surface_builder import IVSurfaceBuilder
    from src.rbergomi import Calibrator

    cfg = CONFIG["rbergomi"]

    # Build index
    data_dir = Path(CONFIG["data_folder"]) / "option"
    builder = IVSurfaceBuilder(str(data_dir))
    builder.build_index()

    # Create calibrator
    calibrator = Calibrator(
        n_paths=cfg["n_paths"],
        n_paths_coarse=cfg["n_paths_coarse"],
        n_steps=cfg["n_steps"],
        seed=cfg["seed"],
        scheme=cfg["scheme"],
        kappa=cfg["kappa"],
        pricing_method=cfg["pricing_method"]
    )

    # Run event calibration
    results_df = calibrator.calibrate_events(
        builder=builder,
        events=None,  # Uses MARKET_EVENTS from builder
        method=cfg["method"],
        maxiter=cfg["maxiter"]
    )

    logger.info(f"Output saved to {cfg['output_dir']}")
    return results_df


def run_rbergomi_plots():
    """Generate rBergomi calibration figures."""
    logger.info("="*60)
    logger.info("rBERGOMI - FIGURE GENERATION")
    logger.info("="*60)

    import polars as pl
    from src.rbergomi import CalibrationVisualizer
    from iv_surface_builder import MARKET_EVENTS

    cfg = CONFIG["rbergomi"]
    output_dir = Path(cfg["output_dir"])

    scheme = cfg["scheme"]
    pricing_method = cfg["pricing_method"]
    viz = CalibrationVisualizer(str(output_dir / "figures"), scheme=scheme,
                                pricing_method=pricing_method)

    # Load time series results if available (.parquet preferred, .csv as fallback)
    pipeline_tag = f"_{scheme}_{pricing_method}"
    ts_parquet = output_dir / "tables" / f"calibration_timeseries{pipeline_tag}.parquet"
    ts_csv = output_dir / "tables" / f"calibration_timeseries{pipeline_tag}.csv"

    if ts_parquet.exists():
        logger.info("Loading time series results (parquet)...")
        results_df = pl.read_parquet(ts_parquet)
    elif ts_csv.exists():
        logger.info("Loading time series results (csv)...")
        results_df = pl.read_csv(ts_csv)
        # Also persist as parquet for future runs
        results_df.write_parquet(ts_parquet)
        logger.info(f"Saved parquet copy: {ts_parquet}")
    else:
        logger.warning(f"Time series results not found: {ts_parquet}")
        logger.info("Run --rbergomi-timeseries or --rbergomi-timeseries-hv first")
        results_df = None

    if results_df is not None:
        # Generate all time series plots
        viz.plot_parameter_stability(results_df, events=MARKET_EVENTS)

        for param in ['H', 'eta', 'rho', 'xi']:
            if param in results_df.columns:
                viz.plot_parameter_with_spot(results_df, param, events=MARKET_EVENTS)

        viz.generate_calibration_table(results_df)
        viz.generate_summary_statistics(results_df)

    logger.info("Figure generation complete")


def run_rbergomi_all():
    """Full rBergomi pipeline."""
    logger.info("="*60)
    logger.info("rBERGOMI - FULL PIPELINE")
    logger.info("="*60)

    # 1. Test pricer
    run_rbergomi_test()

    # 2. Snapshot calibration (data-driven date selection)
    run_rbergomi_snapshot()

    # 3. Time series (this takes time!)
    # Uncomment to run full time series:
    # run_rbergomi_timeseries()

    # 4. Event calibration
    # run_rbergomi_events()

    # 5. Generate plots
    run_rbergomi_plots()

    logger.info("="*60)
    logger.info("rBERGOMI FULL PIPELINE COMPLETE")
    logger.info("="*60)


def run_rbergomi_compare():
    """Side-by-side comparison of Cholesky vs Hybrid scheme."""
    logger.info("="*60)
    logger.info("rBERGOMI - SCHEME COMPARISON (Cholesky vs Hybrid)")
    logger.info("="*60)

    from src.rbergomi import rBergomiPricer, bs_price_inverse_put
    from src.rbergomi.pricer import CholeskyScheme, HybridScheme
    import numpy as np
    import time

    cfg = CONFIG["rbergomi"]
    kappa = cfg["kappa"]

    # --- Model parameters ---
    H, eta, rho, xi = 0.07, 1.9, -0.7, 0.04
    S0, K, T = 50_000, 48_000, 30/365
    bs_price = bs_price_inverse_put(S0, K, T, 0, xi**0.5)

    n_paths_list = [10_000, 20_000, 50_000]
    n_steps_list = [50]

    print("\n" + "="*70)
    print("  CHOLESKY vs HYBRID SCHEME COMPARISON")
    print("="*70)
    print(f"\n  Model:  H={H}, eta={eta}, rho={rho}, xi={xi}")
    print(f"  Option: S0=${S0:,}, K=${K:,}, T={T*365:.0f}d")
    print(f"  BS flat-vol price: {bs_price:.6f} BTC")
    print(f"  Hybrid kappa: {kappa}")

    # --- 1. fBm variance sanity check ---
    print("\n" + "-"*70)
    print("  1. fBm VARIANCE CHECK (10k paths)")
    print("-"*70)
    expected_var = T**(2*H)
    for scheme_name, SchemeClass, kwargs in [
        ("Cholesky", CholeskyScheme, {"n_coarse": 50}),
        ("Hybrid",   HybridScheme,   {"kappa": kappa}),
    ]:
        sch = SchemeClass(H, **kwargs)
        sch.initialize(T, 50)
        fbm, _ = sch.generate_paths(10_000, np.random.default_rng(42))
        var = np.var(fbm[:, -1])
        print(f"  {scheme_name:10s}: var(Z^H_T) = {var:.4f}  "
              f"(expected {expected_var:.4f}, err = {abs(var-expected_var)/expected_var*100:.1f}%)")

    # --- 2. Pricing convergence ---
    print("\n" + "-"*70)
    print("  2. PRICING CONVERGENCE")
    print("-"*70)
    header = f"  {'n_paths':>8s}  {'Scheme':>10s}  {'Price':>10s}  {'StdErr':>10s}  {'IV%':>7s}  {'Time':>8s}"
    print(header)
    print("  " + "-"*62)

    results = {}
    for n_paths in n_paths_list:
        for scheme in ['cholesky', 'hybrid']:
            pricer = rBergomiPricer(
                H, eta, rho, xi,
                n_paths=n_paths, n_steps=50,
                scheme=scheme, kappa=kappa, seed=123
            )
            t0 = time.perf_counter()
            price, se = pricer.price_inverse_put(S0, K, T)
            elapsed = time.perf_counter() - t0
            iv = pricer.implied_vol_inverse_put(S0, K, T)

            results[(n_paths, scheme)] = {
                "price": price, "se": se, "iv": iv, "time": elapsed
            }
            print(f"  {n_paths:>8,d}  {scheme:>10s}  {price:>10.6f}  {se:>10.6f}  "
                  f"{iv*100:>6.2f}%  {elapsed*1000:>7.0f}ms")

    # --- 3. Summary ---
    print("\n" + "-"*70)
    print("  3. SUMMARY (50k paths)")
    print("-"*70)
    n = max(n_paths_list)
    rc = results[(n, 'cholesky')]
    rh = results[(n, 'hybrid')]
    print(f"  Price diff:   {abs(rc['price']-rh['price']):.6f} BTC "
          f"({abs(rc['price']-rh['price'])/rc['price']*100:.2f}%)")
    print(f"  IV diff:      {abs(rc['iv']-rh['iv'])*100:.2f} pp")
    print(f"  Speed ratio:  Cholesky {rc['time']*1000:.0f}ms vs "
          f"Hybrid {rh['time']*1000:.0f}ms "
          f"({rc['time']/rh['time']:.1f}x)")

    # --- 4. n_steps stability (Hybrid only, known Cholesky issue for n>50) ---
    print("\n" + "-"*70)
    print("  4. n_steps STABILITY (20k paths)")
    print("-"*70)
    print(f"  {'n_steps':>8s}  {'Cholesky IV%':>13s}  {'Hybrid IV%':>12s}")
    print("  " + "-"*38)
    for ns in [50, 100, 200]:
        row = []
        for scheme in ['cholesky', 'hybrid']:
            pricer = rBergomiPricer(
                H, eta, rho, xi,
                n_paths=20_000, n_steps=ns,
                scheme=scheme, kappa=kappa, seed=42
            )
            _, _ = pricer.price_inverse_put(S0, K, T)
            iv = pricer.implied_vol_inverse_put(S0, K, T)
            row.append(iv)
        flag = "  <<<" if abs(row[0] - row[1]) * 100 > 5 else ""
        print(f"  {ns:>8d}  {row[0]*100:>12.2f}%  {row[1]*100:>11.2f}%{flag}")

    print("\n" + "="*70)
    print("  NOTE: Cholesky is known to diverge for n_steps > n_coarse (50).")
    print("  The Hybrid scheme is stable for any n_steps.")
    print("="*70 + "\n")

    logger.info("Scheme comparison complete!")


def run_rbergomi_snapshot_both_schemes(target_dates: list = None):
    """Run snapshot calibration with both Cholesky and Hybrid, sequentially."""
    logger.info("="*60)
    logger.info("rBERGOMI - SNAPSHOT CALIBRATION (BOTH SCHEMES)")
    logger.info("="*60)

    from iv_surface_builder import IVSurfaceBuilder

    # Build index once, shared across both runs
    data_dir = Path(CONFIG["data_folder"]) / "option"
    builder = IVSurfaceBuilder(str(data_dir))
    builder.build_index()

    # Discover dates once (scheme-independent)
    if target_dates is None:
        cfg = CONFIG["rbergomi"]
        categories = discover_snapshot_dates(
            builder,
            window_hours=cfg["window_hours"],
            moneyness_range=cfg["moneyness_range"],
            min_ttm_days=cfg["min_ttm_days"],
            max_ttm_days=cfg["max_ttm_days"],
        )
        target_dates = categories["calm"] + categories["high_vol"] + categories["steep_skew"]
        seen = set()
        unique = []
        for dt, label in target_dates:
            key = dt.strftime("%Y-%m-%d")
            if key not in seen:
                seen.add(key)
                unique.append((dt, label))
        unique.sort(key=lambda x: x[0])
        target_dates = unique

    original_scheme = CONFIG["rbergomi"]["scheme"]

    for scheme in ["hybrid", "cholesky"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"  SCHEME: {scheme.upper()}")
        logger.info(f"{'='*60}")
        CONFIG["rbergomi"]["scheme"] = scheme
        run_rbergomi_snapshot(target_dates, builder=builder)

    # Restore original scheme
    CONFIG["rbergomi"]["scheme"] = original_scheme
    logger.info("\nBoth schemes completed. Results saved with _hybrid / _cholesky suffixes.")


def run_post_download():
    """Run the full pipeline AFTER download (Index -> Calibration -> Plots -> rBergomi)."""
    logger.info("="*60)
    logger.info("POST-DOWNLOAD PIPELINE - rBERGOMI THESIS")
    logger.info("="*60)
    logger.info(f"Period: {CONFIG['start_date'].date()} -> {CONFIG['end_date'].date()}")

    # Verify that data exists
    data_dir = Path(CONFIG["data_folder"]) / "option"
    if not data_dir.exists() or len(list(data_dir.glob("*.parquet"))) < 100:
        logger.error(f"Data not found in {data_dir}!")
        logger.error("Run first: python main_c.py --download")
        return

    n_files = len(list(data_dir.glob('*.parquet')))
    logger.info(f"Data found: {n_files} parquet files")

    # 1. Index
    logger.info("\n[1/6] Building index...")
    builder = run_index_build()

    # 2. Legacy IV surface calibration
    logger.info("\n[2/6] IV surface calibration (legacy)...")
    run_calibration(builder)

    # 3. Legacy plots
    logger.info("\n[3/6] Generating plots...")
    run_plots(builder)

    # 4. Animation
    logger.info("\n[4/6] Generating animation...")
    run_animation(builder)

    # 5. rBergomi snapshot calibration
    logger.info("\n[5/6] rBergomi calibration (snapshot)...")
    run_rbergomi_snapshot()

    # 6. rBergomi figures
    logger.info("\n[6/6] rBergomi figures...")
    run_rbergomi_plots()

    logger.info("="*60)
    logger.info("POST-DOWNLOAD PIPELINE COMPLETED")
    logger.info("="*60)
    print_summary()


def run_full_pipeline():
    """Run the entire pipeline."""
    logger.info("="*60)
    logger.info("FULL PIPELINE - rBERGOMI THESIS")
    logger.info("="*60)
    logger.info(f"Period: {CONFIG['start_date'].date()} -> {CONFIG['end_date'].date()}")

    # 1. Download (if needed)
    data_dir = Path(CONFIG["data_folder"]) / "option"
    if not data_dir.exists() or len(list(data_dir.glob("*.parquet"))) < 1000:
        logger.info("Data not found, starting download...")
        asyncio.run(run_download())
    else:
        logger.info(f"Existing data found: {len(list(data_dir.glob('*.parquet')))} files")

    # 2-6: Everything else
    run_post_download()


def print_summary():
    """Print output summary."""
    print("\n" + "="*60)
    print("OUTPUT SUMMARY")
    print("="*60)
    
    output_dir = Path(CONFIG["output_folder"])
    figures_dir = Path(CONFIG["figures_folder"])
    
    if output_dir.exists():
        print(f"\n📁 Calibration Results ({output_dir}):")
        for f in output_dir.glob("*.parquet"):
            print(f"   - {f.name}")
    
    if figures_dir.exists():
        png_files = list(figures_dir.glob("*.png"))
        pdf_files = list(figures_dir.glob("*.pdf"))
        gif_files = list(figures_dir.glob("*.gif"))
        
        print(f"\n📊 Figures ({figures_dir}):")
        print(f"   - PNG: {len(png_files)} files")
        print(f"   - PDF: {len(pdf_files)} files")
        print(f"   - GIF: {len(gif_files)} files")
    
    print("\n" + "="*60)


def _select_scheme_interactive():
    """Prompt user to select simulation pipeline for rBergomi commands.

    Three valid pipelines:
        1. Hybrid  + Euler  — Hybrid fBm, log-Euler spot MC
        2. Hybrid  + Mixed  — Hybrid fBm, Mixed Estimator (Turbocharging, no spot MC noise)
        3. Cholesky + Euler — Cholesky fBm, log-Euler spot MC

    Note: Cholesky + Mixed is not possible (Cholesky does not expose dW increments).
    """
    print("\n  Select simulation pipeline:")
    print("    1. Hybrid  + Euler   (kernel decomposition, log-Euler spot MC)")
    print("    2. Hybrid  + Mixed   (kernel decomposition, Mixed Estimator / Turbocharging)")
    print("    3. Cholesky + Euler  (Cholesky factorisation, log-Euler spot MC)")
    s = input("  Pipeline [1]: ").strip()

    if s == "2":
        CONFIG["rbergomi"]["scheme"] = "hybrid"
        CONFIG["rbergomi"]["pricing_method"] = "mixed"
        k = input("  Kappa (truncation, default 6): ").strip()
        if k.isdigit() and int(k) > 0:
            CONFIG["rbergomi"]["kappa"] = int(k)
        print(f"  -> hybrid + mixed (kappa={CONFIG['rbergomi']['kappa']})")
    elif s == "3":
        CONFIG["rbergomi"]["scheme"] = "cholesky"
        CONFIG["rbergomi"]["pricing_method"] = "euler"
        print("  -> cholesky + euler")
    else:
        CONFIG["rbergomi"]["scheme"] = "hybrid"
        CONFIG["rbergomi"]["pricing_method"] = "euler"
        k = input("  Kappa (truncation, default 6): ").strip()
        if k.isdigit() and int(k) > 0:
            CONFIG["rbergomi"]["kappa"] = int(k)
        print(f"  -> hybrid + euler (kappa={CONFIG['rbergomi']['kappa']})")


def interactive_menu():
    """Interactive menu for operation selection."""
    print("\n" + "="*60)
    print("THESIS: Pricing Bitcoin Inverse Options via rBergomi")
    print("="*60)
    print("\nSelect operation:\n")
    print("  --- Data & Pipeline ---")
    print("   1. Download Deribit data")
    print("   2. Build option index")
    print("   3. Full pipeline (download + everything)")
    print("   4. Post-download pipeline (everything WITHOUT download)")
    print("\n  --- Bitcoin Volatility Analysis ---")
    print("   5. Fat Tails & Kurtosis analysis")
    print("   6. Volatility Smile analysis")
    print("   7. Term Structure analysis")
    print("   8. fBm sample paths plot")
    print("   9. All volatility analyses")
    print("\n  --- Inverse Options Analysis ---")
    print("  10. Greeks comparison (Direct vs Inverse)")
    print("  11. Didactic analysis (Payoff & Decomposition)")
    print("  12. All inverse analyses")
    print("\n  --- BTC vs S&P 500 Comparison ---")
    print("  13. BTC vs SPY volatility comparison")

    print("\n  --- rBergomi Calibration ---")
    print("  15. rBergomi pricer test (Euler vs Mixed Estimator)")
    print("  16. Snapshot calibration (data-driven date selection)")
    print("  17. Time series calibration")
    print("  18. Time series calibration (top 25% volume)")
    print("  19. Market event calibration")
    print("  20. Generate calibration figures")
    print("  21. Full rBergomi pipeline")
    print("  22. Cholesky vs Hybrid comparison")
    print("  23. Snapshot calibration (both schemes)")
    print("      (Options 15-21: prompts for pipeline: Hybrid+Euler / Hybrid+Mixed / Cholesky+Euler)")
    print("\n  --- Plots & Visualization ---")
    print("  24. Generate IV plots")
    print("  25. Generate IV surface animations (GIFs)")
    print("  26. Delta plots only (quick test)")
    print("\n  --- Statistical Analysis ---")
    print("  32. H-bounds sensitivity comparison (0.01 / 0.001 / 1e-4)")
    print("  33. Residual analysis (t-test + Wilcoxon signed-rank)")
    print("  34. Xi approaches comparison (naive vs. bootstrap)")
    print("  35. Calibration bias analysis & correction")
    print("\n  --- Maintenance ---")
    print("  27. Run calibration (legacy)")
    print("  28. Show file/folder status")
    print("  29. Clean temporary files and cache")
    print("  30. Clean obsolete Results folders")
    print("  31. Full cleanup")
    print("\n   0. Exit\n")

    choice = input("Choice [0-35]: ").strip()

    # --- Data & Pipeline ---
    if choice == "1":
        asyncio.run(run_download())
    elif choice == "2":
        run_index_build()
    elif choice == "3":
        run_full_pipeline()
    elif choice == "4":
        run_post_download()
    # --- Bitcoin Volatility Analysis ---
    elif choice == "5":
        from btc_volatility_analysis import run_kurtosis_analysis
        run_kurtosis_analysis()
    elif choice == "6":
        from btc_volatility_analysis import run_smile_analysis
        run_smile_analysis()
    elif choice == "7":
        from btc_volatility_analysis import run_term_structure_analysis
        run_term_structure_analysis()
    elif choice == "8":
        from btc_volatility_analysis import run_fbm_paths
        run_fbm_paths()
    elif choice == "9":
        from btc_volatility_analysis import run_all_volatility_analysis
        run_all_volatility_analysis()
    # --- Inverse Options Analysis ---
    elif choice == "10":
        from inverse_options import run_comparison_analysis
        run_comparison_analysis()
    elif choice == "11":
        from inverse_options import run_didactic_analysis
        run_didactic_analysis()
    elif choice == "12":
        from inverse_options import run_all_inverse_analysis
        run_all_inverse_analysis()
    # --- BTC vs S&P 500 Comparison ---
    elif choice == "13":
        from btc_spy_volatility.btc_vs_sp import run_btc_spy_comparison
        run_btc_spy_comparison()
    # --- rBergomi Calibration ---
    elif choice in ("15", "16", "17", "18", "19", "20", "21"):
        _select_scheme_interactive()
        if choice == "15":
            run_rbergomi_test()
        elif choice == "16":
            from iv_surface_builder import IVSurfaceBuilder
            data_dir = Path(CONFIG["data_folder"]) / "option"
            _builder = IVSurfaceBuilder(str(data_dir))
            _builder.build_index()
            dates = _select_snapshot_dates(_builder)
            run_rbergomi_snapshot(dates, builder=_builder)
        elif choice == "17":
            run_rbergomi_timeseries()
        elif choice == "18":
            run_rbergomi_timeseries_highvol()
        elif choice == "19":
            run_rbergomi_events()
        elif choice == "20":
            run_rbergomi_plots()
        elif choice == "21":
            run_rbergomi_all()
    elif choice == "22":
        run_rbergomi_compare()
    elif choice == "23":
        run_rbergomi_snapshot_both_schemes()
    # --- Plots & Visualization ---
    elif choice == "24":
        run_plots()
    elif choice == "25":
        run_animation()
    elif choice == "26":
        from iv_visualizer import DeltaComparisonPlotter
        plotter = DeltaComparisonPlotter(CONFIG["figures_folder"])
        plotter.plot_delta_comparison(option_type="put", save_name="test_delta_put", show=True)
        plotter.plot_all_greeks_comparison(save_name="test_greeks", show=True)
    # --- Maintenance ---
    elif choice == "27":
        run_calibration()
    elif choice == "28":
        from data_cleaning import run_show_status
        run_show_status()
    elif choice == "29":
        from data_cleaning import run_clean_temp
        run_clean_temp()
    elif choice == "30":
        from data_cleaning import run_clean_results
        run_clean_results()
    elif choice == "31":
        from data_cleaning import run_clean_all
        run_clean_all()
    # --- Statistical Analysis ---
    elif choice == "32":
        _run_compare_h_bounds()
    elif choice == "33":
        _run_residual_analysis()
    elif choice == "34":
        _run_compare_xi()
    elif choice == "35":
        _run_fix_bias()
    elif choice == "0":
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(
        description="Master Thesis Pipeline - rBergomi on Bitcoin Inverse Options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Interactive menu
  python main.py --all            # Full pipeline
  python main.py --download       # Download only
  python main.py --calibrate      # Calibration only
  python main.py --plots          # Plots only
  python main.py --animation      # Animation only
  python main.py --delta-test     # Quick delta plot test

  # Bitcoin volatility analysis
  python main.py --vol-kurtosis   # Fat tails and kurtosis analysis
  python main.py --vol-smile      # Volatility smile analysis
  python main.py --vol-term       # Term structure analysis
  python main.py --vol-all        # All volatility analyses

  # Inverse Options analysis
  python main.py --inverse-comparison  # Greeks comparison
  python main.py --inverse-didactic    # Payoff and decomposition
  python main.py --inverse-all         # All analyses
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--post-download", action="store_true", help="Full pipeline WITHOUT download (Index->Calibration->Plots->rBergomi)")
    parser.add_argument("--download", action="store_true", help="Download data only")
    parser.add_argument("--index", action="store_true", help="Build index only")
    parser.add_argument("--calibrate", action="store_true", help="Calibration only")
    parser.add_argument("--plots", action="store_true", help="Generate plots only")
    parser.add_argument("--animation", action="store_true", help="Generate animation only")
    parser.add_argument("--delta-test", action="store_true", help="Quick delta plot test")
    parser.add_argument("--force-rebuild", action="store_true", help="Force index rebuild")

    # Bitcoin volatility analysis
    parser.add_argument("--vol-kurtosis", action="store_true", help="Fat tails and kurtosis analysis")
    parser.add_argument("--vol-smile", action="store_true", help="Volatility smile analysis")
    parser.add_argument("--vol-term", action="store_true", help="Volatility term structure analysis")
    parser.add_argument("--vol-all", action="store_true", help="All volatility analyses")
    parser.add_argument("--vol-fbm", action="store_true", help="fBm sample paths plot")

    # Inverse Options analysis
    parser.add_argument("--inverse-comparison", action="store_true", help="Greeks comparison: Direct vs Inverse")
    parser.add_argument("--inverse-didactic", action="store_true", help="Didactic payoff and decomposition analysis")
    parser.add_argument("--inverse-all", action="store_true", help="All inverse options analyses")

    # rBergomi Calibration
    parser.add_argument("--rbergomi-test", action="store_true", help="Quick rBergomi pricer test")
    parser.add_argument("--rbergomi-snapshot", action="store_true", help="Multi-date snapshot calibration")
    parser.add_argument("--snapshot-category", choices=["calm", "high_vol", "steep_skew", "all"],
                        default="all", help="Snapshot date category (default: all, data-driven)")
    parser.add_argument("--rbergomi-timeseries", action="store_true", help="Time series calibration")
    parser.add_argument("--rbergomi-timeseries-hv", action="store_true", help="Time series calibration (top 25%% volume)")
    parser.add_argument("--rbergomi-events", action="store_true", help="Market event calibration")
    parser.add_argument("--rbergomi-plots", action="store_true", help="Generate calibration figures")
    parser.add_argument("--rbergomi-all", action="store_true", help="Full rBergomi pipeline")
    parser.add_argument("--rbergomi-compare", action="store_true", help="Cholesky vs Hybrid side-by-side comparison")
    parser.add_argument("--rbergomi-both", action="store_true", help="Snapshot calibration with both schemes")
    parser.add_argument("--scheme", choices=["cholesky", "hybrid"], default=None,
                        help="fBm simulation scheme (default: hybrid)")
    parser.add_argument("--kappa", type=int, default=None,
                        help="Truncation parameter for hybrid scheme (default: 6)")
    parser.add_argument("--pricing-method", choices=["euler", "mixed"], default=None,
                        help="Pricing method: euler (log-Euler spot MC) or mixed (Mixed Estimator / Turbocharging)")

    # Data cleanup
    parser.add_argument("--clean-status", action="store_true", help="Show file and folder status")
    parser.add_argument("--clean-temp", action="store_true", help="Clean temporary files and cache")
    parser.add_argument("--clean-results", action="store_true", help="Clean obsolete Results folders")
    parser.add_argument("--clean-all", action="store_true", help="Full cleanup")

    # BTC vs S&P 500 comparison
    parser.add_argument("--btc-spy", action="store_true", help="BTC vs SPY volatility comparison")


    # Thesis pipeline (Chapter 3 post-calibration steps)
    parser.add_argument("--check-iv", action="store_true",
                        help="Verify ATM IV regime classification for baseline dates")
    parser.add_argument("--generate-figures", action="store_true",
                        help="Generate all thesis figures from calibration CSVs")
    parser.add_argument("--populate-tables", action="store_true",
                        help="Generate all LaTeX .tex table files from calibration CSVs")
    parser.add_argument("--thesis-pipeline", action="store_true",
                        help="Run full post-calibration thesis pipeline: check-iv + generate-figures + populate-tables")
    parser.add_argument("--method", choices=["cholesky_euler", "hybrid_euler", "hybrid_mixed", "all"],
                        default="all", help="Method filter for --generate-figures / analysis commands (default: all)")
    parser.add_argument("--date", default=None,
                        help="Date filter for --generate-figures (YYYYMMDD, default: all dates)")

    # Statistical Analysis (Chapter 3 supplementary)
    parser.add_argument("--compare-h-bounds", action="store_true",
                        help="Compare calibration across H lower bounds (0.01, 0.001, 1e-4)")
    parser.add_argument("--residual-analysis", action="store_true",
                        help="Statistical residual analysis: t-test + Wilcoxon signed-rank")
    parser.add_argument("--compare-xi", action="store_true",
                        help="Compare naive vs. bootstrap forward variance curve (xi approaches)")
    parser.add_argument("--fix-bias", action="store_true",
                        help="Analyze and correct systematic bias in calibrated IVs")

    args = parser.parse_args()
    
    # If no arguments, show interactive menu
    # Exclude args with defaults (scheme, kappa, snapshot_category, method, date) from the check
    action_args = {k: v for k, v in vars(args).items()
                   if k not in ("scheme", "kappa", "snapshot_category", "pricing_method",
                                "method", "date")}
    if not any(action_args.values()):
        interactive_menu()
        return
    
    # Apply CLI overrides to rBergomi config
    if args.scheme is not None:
        CONFIG["rbergomi"]["scheme"] = args.scheme
    if args.kappa is not None:
        CONFIG["rbergomi"]["kappa"] = args.kappa
    if args.pricing_method is not None:
        CONFIG["rbergomi"]["pricing_method"] = args.pricing_method

    # Execute requested operation
    if args.all:
        run_full_pipeline()
    elif args.post_download:
        run_post_download()
    elif args.download:
        asyncio.run(run_download())
    elif args.index:
        run_index_build(force_rebuild=args.force_rebuild)
    elif args.calibrate:
        run_calibration()
    elif args.plots:
        run_plots()
    elif args.animation:
        run_animation()
    elif args.delta_test:
        from iv_visualizer import DeltaComparisonPlotter
        Path(CONFIG["figures_folder"]).mkdir(exist_ok=True)
        plotter = DeltaComparisonPlotter(CONFIG["figures_folder"])
        plotter.plot_delta_comparison(option_type="put", save_name="delta_put", show=True)
        plotter.plot_all_greeks_comparison(save_name="greeks_panel", show=True)
        print(f"\nPlots saved in {CONFIG['figures_folder']}/")

    # Bitcoin volatility analysis
    elif args.vol_all:
        from btc_volatility_analysis import run_all_volatility_analysis
        run_all_volatility_analysis()
    elif args.vol_kurtosis:
        from btc_volatility_analysis import run_kurtosis_analysis
        run_kurtosis_analysis()
    elif args.vol_smile:
        from btc_volatility_analysis import run_smile_analysis
        run_smile_analysis()
    elif args.vol_term:
        from btc_volatility_analysis import run_term_structure_analysis
        run_term_structure_analysis()
    elif args.vol_fbm:
        from btc_volatility_analysis import run_fbm_paths
        run_fbm_paths()

    # Inverse Options analysis
    elif args.inverse_all:
        from inverse_options import run_all_inverse_analysis
        run_all_inverse_analysis()
    elif args.inverse_comparison:
        from inverse_options import run_comparison_analysis
        run_comparison_analysis()
    elif args.inverse_didactic:
        from inverse_options import run_didactic_analysis
        run_didactic_analysis()

    # rBergomi Calibration
    elif args.rbergomi_test:
        run_rbergomi_test()
    elif args.rbergomi_snapshot:
        from iv_surface_builder import IVSurfaceBuilder
        data_dir = Path(CONFIG["data_folder"]) / "option"
        _builder = IVSurfaceBuilder(str(data_dir))
        _builder.build_index()
        cat = args.snapshot_category
        if cat == "all":
            run_rbergomi_snapshot(builder=_builder)
        else:
            cfg = CONFIG["rbergomi"]
            categories = discover_snapshot_dates(
                _builder,
                window_hours=cfg["window_hours"],
                moneyness_range=cfg["moneyness_range"],
                min_ttm_days=cfg["min_ttm_days"],
                max_ttm_days=cfg["max_ttm_days"],
            )
            run_rbergomi_snapshot(categories[cat], builder=_builder)
    elif args.rbergomi_timeseries:
        run_rbergomi_timeseries()
    elif args.rbergomi_timeseries_hv:
        run_rbergomi_timeseries_highvol()
    elif args.rbergomi_events:
        run_rbergomi_events()
    elif args.rbergomi_plots:
        run_rbergomi_plots()
    elif args.rbergomi_all:
        run_rbergomi_all()
    elif args.rbergomi_compare:
        run_rbergomi_compare()
    elif args.rbergomi_both:
        run_rbergomi_snapshot_both_schemes()

    # Data cleanup
    elif args.clean_status:
        from data_cleaning import run_show_status
        run_show_status()
    elif args.clean_temp:
        from data_cleaning import run_clean_temp
        run_clean_temp()
    elif args.clean_results:
        from data_cleaning import run_clean_results
        run_clean_results()
    elif args.clean_all:
        from data_cleaning import run_clean_all
        run_clean_all()

    # BTC vs S&P 500 comparison
    elif args.btc_spy:
        from btc_spy_volatility.btc_vs_sp import run_btc_spy_comparison
        run_btc_spy_comparison()
    # -----------------------------------------------------------------------
    # Thesis pipeline: post-calibration steps (Chapter 3)
    # -----------------------------------------------------------------------
    elif args.thesis_pipeline:
        _run_thesis_pipeline(method=args.method, date_filter=args.date)
    elif args.check_iv:
        _run_check_iv()
    elif args.generate_figures:
        _run_generate_figures(method=args.method, date_filter=args.date)
    elif args.populate_tables:
        _run_populate_tables()

    # Statistical Analysis
    elif args.compare_h_bounds:
        _run_compare_h_bounds(method=args.method)
    elif args.residual_analysis:
        _run_residual_analysis(method=args.method)
    elif args.compare_xi:
        _run_compare_xi(method=args.method)
    elif args.fix_bias:
        _run_fix_bias(method=args.method)


def _run_check_iv():
    """Verify ATM IV regime classification for baseline dates."""
    logger.info("="*60)
    logger.info("CHECK IV — Baseline Date Regime Verification")
    logger.info("="*60)
    from run_all_calibrations import check_baseline_atm_iv
    from iv_surface_builder import IVSurfaceBuilder
    data_dir = Path(CONFIG["data_folder"]) / "option"
    builder = IVSurfaceBuilder(str(data_dir))
    builder.build_index()
    check_baseline_atm_iv(builder)
    logger.info("IV check complete.")


def _run_generate_figures(method: str = "all", date_filter: str = None):
    """Generate thesis figures from calibration CSVs."""
    logger.info("="*60)
    logger.info("GENERATE FIGURES — Chapter 3 Calibration Plots")
    logger.info("="*60)
    import subprocess, sys
    cmd = [sys.executable, "generate_figures.py"]
    if method != "all":
        cmd += ["--method", method]
    if date_filter:
        cmd += ["--date", date_filter]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("generate_figures.py exited with errors.")
    else:
        logger.info("Figure generation complete.")


def _run_populate_tables():
    """Generate LaTeX .tex table files from calibration CSVs."""
    logger.info("="*60)
    logger.info("POPULATE TABLES — Chapter 3 LaTeX Tables")
    logger.info("="*60)
    import subprocess, sys
    cmd = [sys.executable, "populate_tables.py"]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("populate_tables.py exited with errors.")
    else:
        logger.info("Table generation complete.")


def _run_thesis_pipeline(method: str = "all", date_filter: str = None):
    """Run the full post-calibration thesis pipeline.

    Steps:
      1. check-iv   — verify ATM IV regime labels for baseline dates
      2. generate-figures — produce all figures (smile fit, 3D surface, residuals)
      3. populate-tables  — generate all LaTeX .tex table files
    """
    logger.info("="*60)
    logger.info("THESIS PIPELINE — Post-Calibration (Chapter 3)")
    logger.info("="*60)
    _run_check_iv()
    _run_generate_figures(method=method, date_filter=date_filter)
    _run_populate_tables()
    logger.info("="*60)
    logger.info("Thesis pipeline complete. Ready for LaTeX compilation.")
    logger.info("  Figures  -> Results/calibration/figures/{method}/")
    logger.info("  Tables   -> Results/calibration/latex_tables/")
    logger.info("="*60)


def _run_compare_h_bounds(method: str = None):
    """Compare calibration results across H lower bounds (0.01, 0.001, 1e-4)."""
    logger.info("="*60)
    logger.info("COMPARE H BOUNDS — Sensitivity Analysis")
    logger.info("="*60)
    import subprocess, sys
    cmd = [sys.executable, "compare_h_bounds.py"]
    if method and method != "all":
        cmd += ["--method", method]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("compare_h_bounds.py exited with errors.")
    else:
        logger.info("H-bounds comparison complete.")


def _run_residual_analysis(method: str = None):
    """Statistical analysis of calibration residuals (t-test, Wilcoxon signed-rank)."""
    logger.info("="*60)
    logger.info("RESIDUAL ANALYSIS — Statistical Tests")
    logger.info("="*60)
    import subprocess, sys
    cmd = [sys.executable, "residual_analysis.py"]
    if method and method != "all":
        cmd += ["--method", method]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("residual_analysis.py exited with errors.")
    else:
        logger.info("Residual analysis complete.")


def _run_compare_xi(method: str = None):
    """Compare naive vs. bootstrap forward variance curve approaches."""
    logger.info("="*60)
    logger.info("COMPARE XI APPROACHES — Forward Variance Curve")
    logger.info("="*60)
    import subprocess, sys
    cmd = [sys.executable, "compare_xi_approaches.py"]
    if method and method != "all":
        cmd += ["--method", method]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("compare_xi_approaches.py exited with errors.")
    else:
        logger.info("Xi comparison complete.")


def _run_fix_bias(method: str = None):
    """Analyze and correct systematic calibration bias in rBergomi IVs."""
    logger.info("="*60)
    logger.info("FIX CALIBRATION BIAS — Bias Analysis & Correction")
    logger.info("="*60)
    import subprocess, sys
    cmd = [sys.executable, "fix_calibration_bias.py"]
    if method and method != "all":
        cmd += ["--method", method]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("fix_calibration_bias.py exited with errors.")
    else:
        logger.info("Bias analysis complete.")


if __name__ == "__main__":
    main()
