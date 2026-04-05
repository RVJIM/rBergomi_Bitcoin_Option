# PROJECT MAP — rBergomi Thesis Codebase

## Architecture Overview

```
Thesis/
├── main_c.py                  # CLI orchestrator (all commands)
├── deribit_data.py            # Async Deribit API data download
├── data_cleaning.py           # File/folder cleanup utility
├── iv_surface_builder.py      # IV surface extraction + calibration runner
├── iv_visualizer.py           # IV surface & Greeks visualization
├── btc_volatility_analysis.py # Kurtosis, smile, term structure analysis
├── inverse_options.py         # Direct vs Inverse Greeks comparison + didactic
├── btc_spy_volatility/        # BTC vs S&P500 volatility comparison
│   └── btc_vs_sp.py
├── comparison/                # Model comparison outputs
│   ├── confronto_4modelli.html         # Interactive 4-model comparison table
│   ├── confronto_modelli.html          # Model comparison display
│   ├── comparison_BLP2017_MP2018_Codice.docx  # BLP2017 vs MP2018 technical doc
│   ├── model_comparison_remaining.json  # Multi-date calibration metrics
│   └── model_comparison_remaining2.json # Additional comparison results
├── src/
│   ├── rbergomi/              # Base rBergomi (Cholesky fBm)
│   │   ├── pricer.py          # MC pricer (Cholesky + Hybrid Scheme)
│   │   ├── calibrator.py      # DE + Nelder-Mead calibration engine
│   │   ├── mixed_estimator.py # Turbocharging (conditional BS)
│   │   ├── utils.py           # BS pricing, IV inversion, formatting
│   │   └── visualizer.py      # Calibration result plots & LaTeX tables
│   ├── blp2017/               # BLP2017 Hybrid Scheme (fast fBm)
│   │   ├── hybrid_scheme.py   # VolterraFBM, HybridSchemeBLP
│   │   ├── pricer.py          # rBergomiPricerBLP
│   │   └── calibrator.py      # CalibratorBLP
│   ├── mp2018/                # MP2018 Variance Reduction
│   │   ├── volterra.py        # NormalisedVolterra, antithetic sampling
│   │   ├── estimators.py      # Conditional MC, control variates, mixed
│   │   ├── pricer.py          # rBergomiPricerMP
│   │   └── calibrator.py      # CalibratorMP2018
│   └── hybrid/                # Two-Phase Pipeline (BLP2017 + MP2018)
│       ├── calibrator.py      # HybridCalibrator (Phase 1)
│       ├── pricer.py          # HybridPricer (Phase 2)
│       └── pipeline.py        # HybridPipeline orchestration
├── LaTeX/                     # Thesis chapters
│   ├── main.tex               # Document root
│   ├── Chapter_1.tex          # Crypto derivatives & inverse options
│   ├── Chapter_2.tex          # Stochastic → rough volatility theory
│   ├── Chapter_2_hybrid.tex   # Hybrid approach variant
│   ├── Chapter_3.tex          # Empirical analysis & calibration
│   ├── Chapter_3_hybrid.tex   # Hybrid approach variant
│   └── bibliography.tex       # ~30 references
├── data/option/               # ~57k parquet files (Deribit BTC options)
└── Results/                   # All generated output
    ├── calibration/           # rBergomi calibration results
    │   ├── figures/           # IV smile/surface/residuals plots
    │   │   ├── rbergomi/      # Base model (Cholesky scheme)
    │   │   ├── blp2017/       # BLP2017 Hybrid Scheme
    │   │   ├── mp2018/        # MP2018 (standalone, if any)
    │   │   └── hybrid/        # Two-Phase Pipeline (Hybrid+Euler)
    │   └── tables/            # CSV calibration results
    │       ├── rbergomi/      # 13 files
    │       ├── blp2017/       # 27 files
    │       ├── mp2018/        # (standalone, if any)
    │       └── hybrid/        # 14 files
    ├── fat_tails_kurtosis/    # Kurtosis analysis plots
    ├── implied_volatility_smile/
    ├── volatility_term_structure/
    ├── inverse_options/       # comparison/ + didactic/
    └── fbm_paths/             # fBm sample path figures
```

## Data Flow

```
Deribit API → deribit_data.py → data/option/*.parquet
                                      ↓
                          iv_surface_builder.py
                          (index + IV surface extraction)
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                  ↓
          btc_volatility_analysis  iv_visualizer    src/rbergomi/
          (Ch.1-2 figures)         (surface plots)  calibrator.py
                                                    (Ch.3 results)
```

## Key Data Schemas

### Parquet Trade Files (data/option/*.parquet)
| Column | Type | Description |
|---|---|---|
| instrument_name | String | e.g. "BTC-28MAR25-50000-C" |
| timestamp | Int64 | Trade time (ms UTC) |
| expiration_timestamp | Int64 | Expiry time (ms UTC) |
| strike | Float64 | Strike price USD |
| option_type | String | "C" or "P" |
| price | Float64 | Trade price (BTC) |
| iv | Float64 | Implied volatility (%) |
| index_price | Float64 | BTC spot at trade time |
| amount | Float64 | Trade size (BTC) |

### IV Surface DataFrame (from IVSurfaceBuilder.get_iv_surface)
| Column | Type | Description |
|---|---|---|
| strike | Float64 | Strike USD |
| expiration_timestamp | Int64 | Expiry ms |
| iv | Float64 | VWAP IV (%) |
| total_volume | Float64 | Aggregated volume |
| spot | Float64 | Average spot |
| ttm | Float64 | Time to maturity (years) |
| moneyness | Float64 | K/S |
| log_moneyness | Float64 | ln(K/S) |
| forward_log_moneyness | Float64 | ln(K/S)/√τ |

### rBergomi Calibration Input (export_for_rbergomi output)
```python
{"S": float, "K": ndarray, "tau": ndarray, "iv_market": ndarray,
 "moneyness": ndarray, "weights": ndarray, "n_points": int}
```

### CalibrationResult (from Calibrator.calibrate)
```python
CalibrationResult(H, eta, rho, xi, rmse, mae, n_points, n_evals,
                  elapsed_seconds, date, spot, success, method, details)
# details contains: residuals, xi0_curve, optimizer output
```

## CLI Commands (main_c.py)

| Command | Module | Description |
|---|---|---|
| `--download` | deribit_data | Fetch Deribit data |
| `--index` | iv_surface_builder | Build option index |
| `--vol-all` | btc_volatility_analysis | All Ch.1-2 analyses |
| `--inverse-all` | inverse_options | All inverse option figures |
| `--rbergomi-snapshot` | src/rbergomi | Snapshot calibration |
| `--rbergomi-timeseries` | src/rbergomi | Time series calibration |
| `--rbergomi-plots` | src/rbergomi | Generate calibration figures |
| `--scheme hybrid\|cholesky` | src/rbergomi | fBm scheme selection |
| `--pricing-method euler\|mixed` | src/rbergomi | Pricing method |
| `--clean-all` | data_cleaning | Cleanup |

## Dependencies Between Modules

- **pricer.py** imports from: utils.py, mixed_estimator.py
- **calibrator.py** imports from: pricer.py, utils.py
- **visualizer.py** imports from: utils.py, pricer.py, calibrator.py
- **main_c.py** imports from: all root scripts + src/rbergomi
- **iv_visualizer.py** imports from: iv_surface_builder.py (for animations)

## Global Config (main_c.py)
- Thesis period: 2022-01-01 to 2025-12-31
- MC defaults: n_paths=10,000, n_steps=50, seed=42
- Surface: window=4h, moneyness=[0.8, 1.2], TTM=[7d, 90d]
- Optimizer: Differential Evolution + Nelder-Mead refinement
- Bounds: H∈[0.01,0.49], η∈[0.5,2.0], ρ∈[-0.99,-0.10]
- Initial guess: H=0.07, η=1.5, ρ=-0.7
