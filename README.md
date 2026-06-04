# rBergomi Calibration for Bitcoin Inverse Options on Deribit

> End-to-end pricing and calibration framework for BTC inverse options using the Rough Bergomi model.  
> **45% RMSE reduction · 20× faster calibration · 30 market snapshots across 7 major crypto stress events**

---

## Overview

This repository implements a full calibration pipeline for **Bitcoin inverse options** traded on Deribit (~90% global market share), using the **Rough Bergomi (rBergomi)** stochastic volatility model introduced by Bayer, Friz & Gatheral (2016).

The model captures two empirical features of crypto volatility that classical models fail to reproduce:
- **Rough dynamics**: Hurst exponent H ≈ 0.063 (far below 0.5), implying volatility driven by a fractional Brownian motion with very rough paths
- **Steep short-dated smile**: explosive term structure of ATM skew as T → 0, consistent with Deribit observed surfaces

The pipeline runs on **57,000 Deribit parquet files** fetched via the public API and covers 30 market snapshots from May 2022 to March 2025.

---

## Key Results

| Scheme | Mean RMSE (pp) | Time / Snapshot | 30-Snapshot Campaign |
|---|---|---|---|
| Cholesky + Euler (baseline) | 41.76 | 5.6 min | ~2.8 hours |
| Hybrid + Euler | ~32 | ~2 min | ~60 min |
| **Hybrid + Mixed (best)** | **22.83** | **17 sec** | **8.4 min** |

- **Best single-date fit**: 3.94 pp RMSE (calm summer 2023)
- **Crisis fits**: 18–27 pp on event days (LUNA collapse, FTX bankruptcy)
- **Variance reduction**: Romano-Touzi conditional MC + Mixed Estimator → 10–20× vs. naive Monte Carlo

---

## Market Snapshots & Stress Events

The 30 snapshots were selected to cover the full volatility regime spectrum:

| Event | Date | Regime |
|---|---|---|
| LUNA / UST collapse | May 2022 | Extreme stress |
| FTX bankruptcy | Nov 2022 | Extreme stress |
| SVB crisis | Mar 2023 | Medium stress |
| Calm summer | Jun–Aug 2023 | Low volatility |
| Bitcoin Spot ETF approval | Jan 2024 | Medium–high |
| 4th halving | Apr 2024 | Medium |
| BTC $100k milestone | Nov 2024 | High |
| Trump inauguration | Jan 2025 | Medium |

---

## Model

The rBergomi model under the risk-neutral measure:

$$dS_t = S_t \sqrt{V_t} \, dW_t^S$$

$$V_t = \xi_0(t) \cdot \exp\!\left(\eta \, \widetilde{W}_t^H - \tfrac{1}{2}\eta^2 t^{2H}\right)$$

where $\widetilde{W}^H$ is a Riemann-Liouville fractional Brownian motion with Hurst exponent $H \in (0, \tfrac{1}{2})$, and the three parameters to calibrate are:

| Parameter | Description |
|---|---|
| H | Roughness of volatility (fitted: ≈ 0.063) |
| η | Vol-of-vol |
| ρ | Spot-vol correlation |

### Calibration target

Minimise RMSE between model and market **implied volatilities** across all available strikes and maturities on a given snapshot date.

---

## Computational Schemes

Three simulation schemes were implemented and benchmarked:

### 1. Cholesky + Euler (baseline)
Standard approach: simulate fractional BM via Cholesky decomposition of the covariance matrix + Euler discretisation of the variance process. Exact but O(n²) memory and slow.

### 2. Hybrid + Euler
Replace Cholesky with the **Hybrid scheme** (Bennedsen, Lunde & Pakkanen, 2017): represents fBM as a sum of a Wiener integral (handled via Brownian increments) and a Volterra kernel approximation. Same Euler step.

### 3. Hybrid + Mixed (best)
Combines the Hybrid scheme with the **Mixed Estimator** for the price functional, based on the Romano-Touzi (1997) conditional Monte Carlo representation:

$$\text{Price} = \mathbb{E}\!\left[\text{BS}\!\left(S_0,\, \bar{V}_T,\, K,\, T\right)\right]$$

where the outer expectation is over variance paths and the inner Black-Scholes formula is computed analytically given each path's integrated variance $\bar{V}_T$. This eliminates the idiosyncratic noise from the payoff simulation, achieving **10–20× variance reduction** at no additional computational cost.

---

## Optimizer

Two-stage global-to-local optimization:

```
Stage 1: Differential Evolution (global)
  → population-based, derivative-free
  → robust to multimodal surfaces (common under stress)

Stage 2: Nelder-Mead (local refinement)
  → starts from DE best solution
  → fast convergence to local minimum
```

Each evaluation uses **10,000 Monte Carlo paths** per parameter set, accelerated via **Numba JIT compilation**.

---

## Tech Stack

```
Python 3.11+
├── NumPy          # vectorised simulation
├── SciPy          # optimization (DE + Nelder-Mead), integration
├── Polars         # fast parquet ingestion (57k files)
├── Pandas         # time series manipulation
├── Numba          # JIT acceleration of MC simulation kernel
└── Deribit API    # public REST API for options chain data
```

---

## Repository Structure

```
rBergomi_Bitcoin_Option/
│
├── src/
│   ├── rbergomi/
│   │   ├── pricer.py           # rBergomi Monte Carlo pricer (all 3 schemes)
│   │   ├── calibrator.py       # Calibrator + ForwardVarianceCurve + CalibrationResult
│   │   ├── mixed_estimator.py  # Romano-Touzi Mixed Estimator (variance reduction)
│   │   ├── utils.py            # BS pricing, IV inversion, formatting, timers
│   │   ├── visualizer.py       # Publication figures (smile fit, heatmaps, surfaces)
│   │   ├── rbergomi_core.pyx   # Optional Cython core for the pricer
│   │   └── __init__.py
│   └── config/
│       ├── dates.py            # Event and baseline calibration dates
│       ├── methods.py          # Method configs (scheme, paths, labels)
│       ├── surfaces.py         # IV surface extraction settings
│       └── plot_style.py       # Unified matplotlib style
│
├── Results/
│   ├── calibration/
│   │   ├── tables/             # CSV results per date — cholesky_euler / hybrid_euler / hybrid_mixed
│   │   ├── figures/            # Publication PNGs (regenerated locally via --generate-figures)
│   │   ├── latex_tables/       # .tex tables for Chapter 3
│   │   ├── h_bound_comparison/ # Sensitivity analysis on the Hurst lower bound
│   │   ├── residuals/          # Aggregate residual and bias analysis
│   │   ├── snapshot_scan/      # Per-date parameter snapshots (scan)
│   │   └── snapshot_surfaces/  # Per-date IV surface snapshots
│   ├── fat_tails_kurtosis/     # BTC return kurtosis analysis
│   ├── implied_volatility_smile/
│   ├── volatility_term_structure/
│   ├── inverse_options/        # Greeks comparison (direct vs inverse)
│   ├── fbm_paths/              # fBm path visualization
│   ├── fix_bias/               # Monte Carlo IV-bias diagnostics output
│   └── xi_comparison/          # Forward-variance estimation comparison output
│
├── comparison/                 # Cross-method variant comparison (charts + JSON summary)
├── main_c.py                   # CLI dispatcher for all pipeline steps
├── run_all_calibrations.py     # Production calibration: 3 methods × 30 dates
├── iv_surface_builder.py       # IV surface extraction from Deribit parquet files
├── iv_visualizer.py            # IV surface 3D plots and animations
├── inverse_options.py          # Inverse option Greeks and payoff analysis
├── deribit_data.py             # Async Deribit REST API downloader
├── btc_volatility_analysis.py  # Fat tails, smile, term structure analysis
├── generate_figures.py         # Reconstruct Chapter 3 figures from CSVs
├── populate_tables.py          # Generate LaTeX tables from calibration CSVs
├── data_cleaning.py            # Housekeeping utilities
├── compare_h_bounds.py         # Sensitivity study on the Hurst lower bound
├── compare_xi_approaches.py    # Forward-variance estimation comparison
├── fix_calibration_bias.py     # Monte Carlo IV-bias diagnostics
├── residual_analysis.py        # Aggregate residual / bias analysis
├── btc_spy_volatility/         # BTC vs S&P 500 volatility comparison (optional)
├── portfolio.html              # Visual overview of code architecture and thesis structure
├── volatility_data_2015_2024.csv
└── requirements.txt
```

> **Note:** the raw Deribit data (`data/`, ~2.1 GB of parquet files) is not tracked in the repository.
> Reconstruct locally via `python main_c.py --download`.
> The thesis LaTeX sources (`LaTeX/`), reference papers (`Paper/`) and notes (`docs/`) are also excluded from version control.

---

## References

- Bayer, C., Friz, P., & Gatheral, J. (2016). *Pricing under rough volatility*. Quantitative Finance.
- Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017). *Hybrid scheme for Brownian semistationary processes*. Finance and Stochastics.
- Romano, M., & Touzi, N. (1997). *Contingent claims and market completeness in a stochastic volatility model*. Mathematical Finance.
- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). *Volatility is rough*. Quantitative Finance.

---

## Author

**Riccardo Caruso** — MSc Computational Finance, University of Padua  
[LinkedIn](https://linkedin.com/in/riccardo-caruso-g/) · [GitHub](https://github.com/RVJIM)  
Thesis project, April 2026
