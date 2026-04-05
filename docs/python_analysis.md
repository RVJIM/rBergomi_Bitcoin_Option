# Python Context: Analysis Scripts & Orchestrator

## btc_volatility_analysis.py (1008 lines)
Three analyzers + fBm path plotter for thesis Ch.1-2 figures.

### Global CONFIG
```python
CONFIG = {
    "data_dir": Path("data/option"),
    "results_dir": Path("Results"),
    "index_file": Path("data/option_index.parquet"),
    "kurtosis": {"output_dir": "fat_tails_kurtosis",
                 "frequencies": [('5-minute','5min'),('Hourly','1h'),('Daily','1d'),('Weekly','1W')]},
    "smile": {"output_dir": "implied_volatility_smile", "max_options": 1000,
              "min_volume": 2.0, "min_trades": 3, "days_to_expiry_range": (1,60),
              "moneyness_range": (0.75,1.35), "iv_bounds": (10,250)},
    "term_structure": {"output_dir": "volatility_term_structure", "sample_size": 2000},
    "dpi_save": 300
}
```

### Classes
```python
class KurtosisAnalyzer:
    def load_perpetual_data() -> pd.DataFrame  # BTC-PERPETUAL, resampled 1min
    def compute_returns(df, freq) -> pd.Series  # Log-returns
    def compute_statistics(returns) -> dict  # mean, std, skewness, kurtosis, JB test
    def analyze_tail_events(returns, thresholds=[2,3,4,5]) -> pd.DataFrame
    def plot_analysis(returns_dict)  # 2 figures: distribution + kurtosis analysis
    def run()  # Full pipeline

class SmileAnalyzer:
    def load_option_data(max_options, sample_seed=42) -> pd.DataFrame
        # Uses option_index.parquet, filters by volume/trades/DTE/moneyness
    def calculate_statistics(df) -> dict  # OTM puts/ATM/OTM calls IV breakdown
    def plot_2d_smile(df, stats)  # Main smile + by-strike + stats panel
    def plot_3d_surface(df)  # matplotlib 3D + Plotly interactive HTML
    def run()

class TermStructureAnalyzer:
    def load_option_data(sample_size=2000) -> pd.DataFrame
    def categorize_maturity(ttm) -> str  # Short/Medium/Long/Very Long buckets
    def compute_smile_metrics(df) -> pd.DataFrame  # ATM IV, skew, curvature per bucket
    def plot_analysis(df, metrics)  # 4-panel: smiles, skew coeff, skew, summary
    def run()

def plot_fbm_sample_paths()
    # H=0.5 vs H=0.1 side-by-side, Cholesky decomposition
    # Output: Results/fbm_paths/figures/fbm_sample_paths.png
```

### Public Entry Points (called from main_c.py)
```python
def run_kurtosis_analysis()
def run_smile_analysis()
def run_term_structure_analysis()
def run_fbm_paths()
def run_all_volatility_analysis()  # All four
```

---

## inverse_options.py (769 lines)
Direct vs Inverse Greeks comparison + didactic payoff analysis.

### CONFIG
```python
CONFIG = {
    "S": 50_000, "K": 50_000, "r": 0.0, "r_btc": 0.0, "sigma": 0.60,
    "comparison": {"S_fixed": 25_000, "sigma": 0.75,
                   "taus": [10/365, 30/365, 90/365], "K_range": (10_000, 40_000)},
    "didactic": {"S_range_call": (20_000, 100_000), "S_range_put": (10_000, 100_000),
                 "maturities": [0.02, 0.25, 1.0]}
}
```

### BS Functions
```python
def d1(S, K, r, sigma, tau) -> float
def d2(S, K, r, sigma, tau) -> float
# Direct Greeks:
def direct_call_delta(S, K, r, sigma, tau)   # Φ(d1)
def direct_put_delta(S, K, r, sigma, tau)    # Φ(d1) - 1
def direct_gamma(S, K, r, sigma, tau)        # φ(d1)/(S·σ·√τ)
def direct_put_price(S, K, T, r, sigma)
# Inverse Greeks (BTC-denominated):
def inverse_call_delta(S, K, r, r_btc, sigma, tau)  # (K/S²)e^{-r_btc·τ}Φ(-d2)
def inverse_put_delta(S, K, r, r_btc, sigma, tau)   # -(K/S²)e^{-r_btc·τ}Φ(d2)
def inverse_call_gamma(S, K, r, r_btc, sigma, tau)
def inverse_put_gamma(S, K, r, r_btc, sigma, tau)
def inverse_put_delta_from_price(S, K, T, r, sigma)  # Δ_USD/S - P_USD/S²
```

### Classes
```python
class ComparisonAnalyzer:
    # Generates 6 figures: call/put delta, call/put gamma, 2×2 panels (call/put)
    def plot_call_delta()   # Direct vs Inverse, 3 maturities
    def plot_put_delta()
    def plot_call_gamma()
    def plot_put_gamma()
    def plot_greeks_panel_put()  # 2×2: Delta + Gamma
    def plot_greeks_panel_call()
    def run()

class DidacticAnalyzer:
    # Payoff comparisons + decompositions
    def plot_call_payoff_comparison()  # Standard (USD) vs Inverse (BTC)
    def plot_put_payoff_comparison()   # Inverse put: unbounded as S→0
    def plot_call_decomposition()      # Digital - K×Reciprocal Put
    def plot_put_decomposition()       # K × Reciprocal Call on 1/S
    def plot_delta_maturity_comparison()  # Standard vs Inverse across maturities
    def run()
```

### Public Entry Points
```python
def run_comparison_analysis()
def run_didactic_analysis()
def run_all_inverse_analysis()  # Both
```

Output: Results/inverse_options/{comparison,didactic}/*.png

---

## main_c.py (1897 lines)
CLI orchestrator. All commands documented in docstring.

### Global CONFIG
```python
CONFIG = {
    "start_date": datetime(2022, 1, 1),
    "end_date": datetime(2025, 12, 31, 23, 59, 59),
    "data_folder": "data",
    "rbergomi": {
        "n_paths": 10_000, "n_paths_coarse": 2_000, "n_steps": 50, "seed": 42,
        "method": "differential_evolution", "maxiter": 25, "tol": 1.0,
        "window_hours": 4.0, "min_volume": 0.05,
        "moneyness_range": (0.8, 1.2), "min_ttm_days": 7, "max_ttm_days": 90,
        "scheme": "hybrid", "kappa": 6, "pricing_method": "euler",
        "output_dir": "Results/calibration"
        # Output saved to: Results/calibration/figures/{model}/ and tables/{model}/
        # where {model} = rbergomi | blp2017 | mp2018 | hybrid
    }
}
```

### Key Functions
```python
def run_rbergomi_snapshot(cfg, scheme, kappa, pricing_method, snapshot_category=None)
    # 15 curated dates split into: calm(5), high_vol(5), steep_skew(5)
    # Calibrates each, saves results + skew/residual plots

def run_rbergomi_timeseries(cfg, scheme, kappa, pricing_method)
    # Weekly calibration 2022-2025, warm-start from previous result

def run_rbergomi_timeseries_highvol(cfg, scheme, kappa, pricing_method)
    # Same but filtered to top 25% volume dates

def run_rbergomi_events(cfg, scheme, kappa, pricing_method)
    # Calibration around 10 market events (pre/event/post)

def run_rbergomi_compare(cfg)
    # Side-by-side Cholesky vs Hybrid comparison

def run_rbergomi_plots(cfg, scheme, pricing_method)
    # Generate all calibration figures from saved results
```

### Snapshot Dates (15 curated)
```python
SNAPSHOT_DATES = {
    "calm": [2023-06-15, 2023-09-15, 2024-03-15, 2024-07-15, 2024-09-15],
    "high_vol": [2022-06-15, 2022-11-15, 2024-01-15, 2024-11-15, 2025-01-15],
    "steep_skew": [2022-03-15, 2022-09-15, 2023-03-15, 2024-04-15, 2025-03-15]
}
```

### argparse Flags
--all, --post-download, --download, --index, --calibrate, --plots, --animation,
--vol-{kurtosis,smile,term,fbm,all}, --inverse-{comparison,didactic,all},
--rbergomi-{test,snapshot,timeseries,timeseries-hv,events,plots,all,compare,both},
--scheme {cholesky,hybrid}, --kappa N, --pricing-method {euler,mixed},
--snapshot-category {calm,high_vol,steep_skew},
--clean-{status,temp,results,all}, --btc-spy, --btc-spy-interactive
