# Pricing and Calibration of Bitcoin Inverse Options via Rough Bergomi
---

> For a visual overview of the code architecture and thesis structure, open [portfolio.html](portfolio.html) in a browser.

## Overview

This repository contains the full Python implementation for the pricing and calibration of **Bitcoin inverse options** using the **rough Bergomi (rBergomi) model**.

Bitcoin options on Deribit are *inverse* contracts: premiums and payoffs are denominated in BTC, not USD. This introduces non-trivial pricing adjustments compared to standard equity options. The rBergomi model — a stochastic volatility model driven by fractional Brownian motion — is calibrated to market-implied volatility surfaces extracted from Deribit trade data across 22 dates spanning 2022–2025, covering major market events (LUNA collapse, FTX bankruptcy, BTC spot ETF approval, Bitcoin halving, BTC $100k, Trump inauguration) and baseline volatility regimes (low, medium, high IV).

Three simulation schemes are compared:

| Method | fBm Scheme | Pricing |
|--------|-----------|---------|
| `cholesky_euler` | Coarse-grid Cholesky | Log-Euler MC |
| `hybrid_euler` | Hybrid (BLP 2017) | Log-Euler MC |
| `hybrid_mixed` | Hybrid (BLP 2017) | Mixed Estimator (MP 2018) |

---

## Repository Structure

```
.
├── src/
│   ├── rbergomi/
│   │   ├── pricer.py           # rBergomi Monte Carlo pricer
│   │   ├── calibrator.py       # Calibrator + ForwardVarianceCurve + CalibrationResult
│   │   ├── mixed_estimator.py  # McCrickerd & Pakkanen (2018) variance reduction
│   │   ├── utils.py            # BS pricing, IV inversion, formatting, timers
│   │   ├── visualizer.py       # Publication figures (smile fit, heatmaps, surfaces)
│   │   └── __init__.py
│   └── config/
│       ├── dates.py            # Event and baseline calibration dates
│       ├── methods.py          # Method configs (scheme, paths, labels)
│       ├── surfaces.py         # IV surface extraction settings
│       └── plot_style.py       # Unified matplotlib style
│
├── data/
│   ├── option/                 # Deribit parquet files (one per instrument)
│   ├── option_index.parquet    # Lightweight query index
│   └── option-list.csv         # Instrument metadata
│
├── Results/
│   ├── calibration/
│   │   ├── tables/             # CSV results (one per date x method)
│   │   ├── figures/            # PNG figures (4 per calibration)
│   │   └── latex_tables/       # .tex tables for Chapter 3
│   ├── fat_tails_kurtosis/     # BTC return kurtosis analysis
│   ├── implied_volatility_smile/
│   ├── volatility_term_structure/
│   ├── inverse_options/        # Greeks comparison (direct vs inverse)
│   └── fbm_paths/              # fBm path visualization
│
├── main_c.py                   # CLI dispatcher for all pipeline steps
├── run_all_calibrations.py     # Production calibration: 3 methods x 22 dates
├── iv_surface_builder.py       # IV surface extraction from Deribit parquet files
├── iv_visualizer.py            # IV surface 3D plots and animations
├── inverse_options.py          # Inverse option Greeks and payoff analysis
├── deribit_data.py             # Async Deribit REST API downloader
├── btc_volatility_analysis.py  # Fat tails, smile, term structure analysis
├── generate_figures.py         # Reconstruct Chapter 3 figures from CSVs
├── populate_tables.py          # Generate LaTeX tables from calibration CSVs
├── data_cleaning.py            # Housekeeping utilities
├── btc_spy_volatility/         # BTC vs S&P 500 volatility comparison (optional)
└── requirements.txt
```

---

## Installation

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

**Optional but recommended** for ~10–50x speedup:

```bash
pip install numba
```

If Numba is not available, the pricer falls back to pure NumPy automatically.

---

## Usage

All pipeline steps are dispatched through `main_c.py`. Run with `--help` to see all available commands.

### 1. Download Deribit Data

Downloads all BTC option trades from 2022 to 2025 via the Deribit REST API. Saves one `.parquet` file per instrument under `data/option/`.

```bash
python main_c.py --download
```

> This step takes 2–4 hours due to API rate limits. Data is saved incrementally and the download can be resumed.

### 2. Build the Instrument Index

Scans all parquet files and creates a lightweight index (`data/option_index.parquet`) for fast surface queries.

```bash
python main_c.py --index
```

### 3. Volatility Analysis

Run the volatility characterization analyses (Chapters 1–2):

```bash
# Fat tails and kurtosis of BTC returns
python main_c.py --vol-kurtosis

# Implied volatility smile reconstruction
python main_c.py --vol-smile

# Volatility term structure
python main_c.py --vol-term

# Run all three at once
python main_c.py --vol-all
```

Results (figures + tables) are saved in `Results/fat_tails_kurtosis/`, `Results/implied_volatility_smile/`, `Results/volatility_term_structure/`.

### 4. Inverse Options Analysis

Compute and compare Greeks (delta, gamma) for direct vs inverse options:

```bash
python main_c.py --inverse-all
```

Results saved in `Results/inverse_options/`.

### 5. rBergomi Calibration

**Single date** (interactive, from `main_c.py`):

```bash
python main_c.py --rbergomi-snapshot
```

**Full production run** — calibrates all 3 methods across all 22 dates (~2–3 hours):

```bash
python run_all_calibrations.py
```

Output CSVs are written to `Results/calibration/tables/{method}/calibration_{date}_{method}.csv`.

Each CSV row contains:

| Column | Description |
|--------|-------------|
| `date` | Calibration timestamp |
| `spot` | BTC/USD spot price |
| `H` | Hurst parameter |
| `eta` | Vol-of-vol |
| `rho` | Spot-vol correlation |
| `xi` | Forward variance (from ATM IVs) |
| `rmse_pp` | RMSE in percentage points |
| `mae_pp` | MAE in percentage points |
| `time_seconds` | Total calibration time |
| `quality_flag` | `good` / `acceptable` / `borderline` / `poor` |

### 6. Figure Generation

Reconstruct all publication figures from calibration CSVs (50k MC paths):

```bash
python main_c.py --generate-figures
```

Produces 4 figures per (date, method) combination:
- **Smile fit**: market vs model IV by maturity
- **Residuals heatmap**: error magnitude over (moneyness, maturity)
- **Residuals histogram**: error distribution
- **3D IV surface**: market vs model comparison

Saved to `Results/calibration/figures/{method}/`.

### 7. LaTeX Table Generation

Generate `.tex` tables for direct inclusion in the thesis:

```bash
python main_c.py --populate-tables
```

Saved to `Results/calibration/latex_tables/`.

---

## Core Module: `src/rbergomi/`

### Pricer

```python
from src.rbergomi import rBergomiPricer

pricer = rBergomiPricer(
    H=0.10,        # Hurst parameter (roughness): H << 0.5 means rough vol
    eta=1.5,       # Vol-of-vol
    rho=-0.60,     # Spot-vol correlation
    xi=0.40**2,    # Forward variance (scalar or ForwardVarianceCurve)
    n_paths=10_000,
    n_steps=50,
    scheme='hybrid',         # 'hybrid' or 'cholesky'
    pricing_method='euler'   # 'euler' or 'mixed'
)

# Price a single inverse put
price_btc, std_err = pricer.price_inverse_put(S0=50000, K=48000, T=0.25)

# Price a batch of options
prices, std_errs = pricer.price_inverse_options_batch(
    S0=50000,
    strikes=[44000, 46000, 48000, 50000, 52000, 54000],
    T=0.25
)

# Compute implied volatility from price
iv = pricer.compute_implied_vol(S0=50000, K=48000, T=0.25,
                                price_btc=price_btc, option_type='put')
```

### Calibrator

```python
from src.rbergomi import Calibrator

calibrator = Calibrator(
    n_paths=10_000,
    n_steps=50,
    scheme='hybrid',
    pricing_method='mixed'
)

# Market data dict (output of IVSurfaceBuilder)
market_data = {
    'S': 50000.0,
    'K': [...],           # strike array
    'tau': [...],         # maturity in years
    'iv_market': [...]    # market implied vols (decimals, e.g. 0.65)
}

result = calibrator.calibrate(market_data)

print(result.H, result.eta, result.rho)
print(f"RMSE: {result.rmse_pp:.2f} pp  [{result.quality_flag}]")
```

### ForwardVarianceCurve

The forward variance `xi` is not a free optimization parameter. It is estimated directly from ATM implied volatilities before calibration begins:

```python
from src.rbergomi import ForwardVarianceCurve

xi_curve = ForwardVarianceCurve(
    maturities=[0.083, 0.167, 0.25, 0.50],          # T in years
    xi_values=[0.45**2, 0.50**2, 0.55**2, 0.52**2]  # ATM IV^2
)

# Piecewise-constant interpolation
xi_t = xi_curve(0.15)
```

---

## Model Background

The **rough Bergomi model** (Bayer, Friz & Gatheral 2016) specifies the instantaneous variance as:

$$V_t = \xi_0(t) \cdot \mathcal{E}\!\left(\eta \, \tilde{W}^H_t\right)$$

where $\tilde{W}^H_t$ is a Riemann-Liouville fractional Brownian motion with Hurst exponent $H \in (0, \frac{1}{2})$, and $\xi_0(t)$ is the forward variance curve estimated from market ATM volatilities.

The model has three free parameters: $(H, \eta, \rho)$.

**Bitcoin inverse options** have payoff denominated in BTC:

$$\text{Inverse Call} = \frac{\max(S_T - K,\ 0)}{S_T} \qquad \text{Inverse Put} = \frac{\max(K - S_T,\ 0)}{S_T}$$

Pricing is performed via Monte Carlo on the rBergomi variance path, with payoffs computed in BTC and IV extracted via inverse Black-Scholes.

**Calibration** is a two-phase optimization:
1. **Differential Evolution** (coarse, ~2k paths) for global search over $(H, \eta, \rho)$
2. **Nelder-Mead** (fine, ~10k paths) for local refinement

RMSE quality thresholds: `good` < 10 pp, `acceptable` < 20 pp, `borderline` < 30 pp, `poor` >= 30 pp.

---

## Calibration Dates

### Market Events (7 events x 3 days)

| Event | Dates |
|-------|-------|
| LUNA collapse | 2022-05-08, 09, 10 |
| FTX bankruptcy | 2022-11-07, 08, 09 |
| SVB bank run | 2023-03-09, 10, 11 |
| BTC spot ETF approval | 2024-01-09, 10, 11 |
| Bitcoin halving | 2024-04-19, 20, 21 |
| BTC $100k | 2024-12-04, 05, 06 |
| Trump inauguration | 2025-01-19, 20, 21 |

### Baseline Regimes (9 dates)

| Regime | Dates |
|--------|-------|
| Low IV | 2022-06-18, 2022-07-02, 2023-08-12 |
| Medium IV | 2023-09-30, 2024-02-10, 2024-09-14 |
| High IV | 2022-11-16, 2022-12-17, 2025-02-03 |

---

## Performance

| Task | Approximate Time |
|------|-----------------|
| Data download (2022–2025) | 2–4 hours |
| Index build | ~1 min |
| Single-date IV extraction | ~1 sec |
| Single-date calibration (10k paths) | ~15–30 sec |
| Single-date calibration with Numba | ~5–10 sec |
| Full calibration run (66 total) | ~2–3 hours |
| Figure generation (all) | ~30 min |

---

## References

- Bayer, C., Friz, P., & Gatheral, J. (2016). *Pricing under rough volatility*. Quantitative Finance, 16(6), 887–904.
- Bennedsen, M., Lunde, A., & Pakkanen, M. S. (2017). *Hybrid scheme for Brownian semistationary processes*. Finance and Stochastics, 21(4), 931–965.
- McCrickerd, R., & Pakkanen, M. S. (2018). *Turbocharging Monte Carlo pricing for the rough Bergomi model*. Quantitative Finance, 18(11), 1877–1886.
