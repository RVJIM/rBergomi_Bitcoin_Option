# Python Context: rBergomi Engine (src/rbergomi/)

## __init__.py
Exports: rBergomiPricer, Calibrator, CalibrationResult, ForwardVarianceCurve, MixedEstimator, CalibrationVisualizer, utils functions. Version 2.1.0. Optional Cython via rbergomi_core.pyx.

---

## pricer.py (1596 lines)
Monte Carlo pricer for inverse BTC options under rBergomi.

### Model
```
dS_t/S_t = √V_t dW_t
V_t = ξ(t) exp(η Z^H_t − 0.5 η² t^{2H})
```

### Two fBm Schemes
- **cholesky** (default): Coarse-grid Cholesky + linear interpolation. O(n_c³ + n_f·n_c).
- **hybrid**: Full Hybrid Scheme (Bennedsen-Lunde-Pakkanen 2017). Kernel decomposition into Distant(D) + Recent(R) with truncation κ. O(n·κ).

### Two Pricing Methods
- **euler**: Log-Euler spot path simulation + MC payoff averaging
- **mixed**: Mixed Estimator (Turbocharging) — conditional BS per variance path

### Numba-Optimized Core Functions
```python
@njit _fbm_covariance_matrix(times, H) -> ndarray  # Cov(Z^H_s, Z^H_t)
@njit _generate_fbm_paths_numba(n_paths, n_steps, chol_lower, random_normals) -> ndarray
@njit _generate_variance_paths_numba(fbm_paths, times, xi, eta, H) -> ndarray
@njit _generate_spot_paths_numba(var_paths, rho, dt, random_normals2) -> ndarray  # log(S/S0)
@njit _compute_inverse_payoffs_numba(log_spot_paths, K_over_S0, is_put) -> ndarray

# Hybrid Scheme specific:
@njit _hybrid_kernel_weights(times, H, kappa) -> (W_D, W_R)  # Distant + Recent
@njit _hybrid_generate_fbm(n_paths, n_steps, W_D, W_R, kappa, random_normals) -> ndarray
```

### Main Class
```python
class rBergomiPricer:
    def __init__(self, H=0.07, eta=1.9, rho=-0.7, xi=0.04,
                 n_paths=50_000, n_steps=100, seed=None,
                 scheme='cholesky', kappa=6, pricing_method='euler')
        # Pre-computes: covariance matrix, Cholesky factor (cached)
        # xi can be float or ForwardVarianceCurve (callable)

    def price_inverse_put(S0, K, T, r=0.0) -> Tuple[float, float]
        # Returns (price_btc, std_error)

    def price_inverse_call(S0, K, T, r=0.0) -> Tuple[float, float]

    def price_multiple_strikes(S0, strikes, T, option_type='put', r=0.0)
        -> Tuple[ndarray, ndarray]
        # Batch pricing: single path generation, multiple payoffs
        # Key optimization: shares variance paths across strikes

    def implied_vol(S0, K, T, option_type='put', r=0.0) -> float
        # Price → IV inversion via Brent's method

    def update_params(H=None, eta=None, rho=None, xi=None)
        # Update parameters without full re-initialization
        # Recomputes covariance only if H changes
```

### Performance Notes
- Numba JIT with fastmath=True, cache=True, parallel=True
- Fallback to pure NumPy if Numba unavailable
- Covariance matrix cached (recomputed only when H or n_steps change)
- Random normals pre-generated in single vectorized call

---

## calibrator.py (1654 lines)
Calibration engine: fits (H, η, ρ) to Deribit IV surface.

### Parameter Bounds & Defaults
```python
DEFAULT_BOUNDS = {"H": (0.01, 0.49), "eta": (0.5, 2.0), "rho": (-0.99, -0.10)}
DEFAULT_INITIAL_GUESS = {"H": 0.07, "eta": 1.5, "rho": -0.7}
```

### Classes
```python
class ForwardVarianceCurve:
    def __init__(self, maturities: ndarray, xi_values: ndarray)
    def __call__(self, t: float) -> float  # Piecewise-constant interpolation
    @property mean_xi -> float
    def to_dict() -> dict

class CalibrationResult:
    # Attributes: H, eta, rho, xi, rmse, mae, n_points, n_evals,
    #   elapsed_seconds, date, spot, success, method, details
    @property params -> Dict[str, float]
    @property sigma_atm -> float  # √xi
    def to_dict() -> dict
    def to_polars_row() -> dict

class Calibrator:
    def __init__(self, n_paths=10_000, n_steps=50, seed=42,
                 bounds=None, scheme='cholesky', kappa=6, pricing_method='euler')

    def calibrate(market_data: Dict, date=None, method='differential_evolution',
                  maxiter=25, tol=1.0, n_paths_coarse=None, verbose=True)
        -> CalibrationResult
        # Pipeline:
        #   1. Estimate xi_0(T) from ATM IVs (per-maturity σ_ATM²)
        #   2. Differential Evolution (global, coarse MC paths)
        #   3. Nelder-Mead refinement (local, full MC paths)
        #   4. Compute residuals (per-point model_iv vs market_iv)

    def calibrate_timeseries(builder, dates, ...) -> pl.DataFrame
        # Calibrate on multiple dates, warm-start from previous result

    def _estimate_xi0_curve(market_data) -> ForwardVarianceCurve
        # Groups by maturity → ATM IV → xi = σ_ATM²
        # ATM = closest to moneyness=1.0 per maturity bucket

    def _loss_function(params, market_data, xi_curve, pricer) -> float
        # RMSE in IV percentage points
        # Batch pricing per unique maturity
        # IV inversion via implied_vol_batch
```

### Loss Function Flow
```
params [H, η, ρ] → update pricer → for each unique τ:
  price_multiple_strikes(S0, K_group, τ) → prices (BTC)
  → implied_vol_batch(prices) → model_IVs
→ RMSE = √mean((model_IV - market_IV)² × 100²)
```

---

## mixed_estimator.py (467 lines)
Conditional BS approach from McCrickerd & Pakkanen (2018).

### Key Idea
Conditioned on variance path {V_t}:
```
S_T | {V_t} ~ LogNormal(μ_c, σ_c²)
μ_c = −½∫V_t dt + ρ∫√V_t dW^⊥
σ_c² = (1−ρ²)∫V_t dt
```
→ BS evaluation per path (no spot discretization noise).

### Numba Functions
```python
@njit _norm_cdf_scalar(x) -> float  # Abramowitz & Stegun approx, error < 1.5e-7
@njit _compute_integrated_quantities(var_paths, dW_driving, dt) -> (I_V, I_corr)
@njit _mixed_estimator_prices_multi_strike(S0, strikes, I_V, I_corr, rho, is_put)
      -> (prices, std_errs)
@njit _mixed_estimator_single_strike(S0, K, I_V, I_corr, rho, is_put)
      -> (price, std_err)
```

### Class
```python
class MixedEstimator:
    def price_inverse_options(S0, strikes, var_paths, dW_driving, dt, rho, option_type='put')
        -> (prices, std_errs)
    def price_single(S0, K, var_paths, dW_driving, dt, rho, option_type='put')
        -> (price, std_err)
```
Note: Requires `scheme='hybrid'` in pricer (needs dW_driving increments).

---

## utils.py (651 lines)
Helper functions for BS pricing, IV inversion, performance, formatting.

### BS Pricing (Inverse Options)
```python
def bs_d1(S, K, T, r, sigma) -> float
def bs_d2(S, K, T, r, sigma) -> float
def bs_price_call_usd(S, K, T, r, sigma) -> float
def bs_price_put_usd(S, K, T, r, sigma) -> float
def bs_price_inverse_call(S, K, T, r, sigma) -> float  # = call_usd / S
def bs_price_inverse_put(S, K, T, r, sigma) -> float   # = put_usd / S
def bs_vega_inverse(S, K, T, r, sigma) -> float         # √T · φ(d1)
```

### IV Inversion
```python
def implied_vol_from_price_inverse(price_btc, S, K, T, option_type='put',
    r=0.0, tol=1e-8) -> float
    # Brent's method, bracket [0.01, 5.0]

def implied_vol_newton(price_btc, S, K, T, ...) -> float
    # Newton-Raphson (faster, less robust)

def implied_vol_batch(prices, S, K_arr, tau_arr, option_type='put',
    r=0.0, sigma_lo=0.01, sigma_hi=5.0) -> ndarray
    # Sequential brentq over N points
```

### Performance
```python
@timer  # Decorator: logs ms, stores last_time, call_count, total_time

class PerformanceTracker:
    def measure(name) -> context_manager  # Time code blocks
    def summary() -> str                   # Formatted table
    def reset()
```

### Formatting
```python
def format_european(value, decimals=2) -> str  # 12345.67 → "12.345,67"
def format_percentage(value, decimals=2) -> str  # 0.6 → "60.00%"
def validate_params(H, eta, rho, xi) -> (bool, str)
def validate_option_params(S, K, T) -> (bool, str)
```

---

## visualizer.py (891 lines)
Publication-quality figures for thesis Chapter 3.

### Class
```python
class CalibrationVisualizer:
    def __init__(self, output_dir="Results/calibration/figures",
                 scheme="hybrid", pricing_method="euler")
        # Saves to output_dir/{model_subfolder}/ where model_subfolder is:
        #   cholesky → rbergomi/, hybrid → blp2017/, hybrid+euler → hybrid/
        # tag = "{scheme}_{pricing_method}" used in all filenames

    def plot_parameter_stability(results: pl.DataFrame, events=None) -> Figure
        # 2×2 grid: H, η, ρ, ξ over time with mean lines + event annotations

    def plot_parameter_with_spot(results, param='H', events=None) -> Figure
        # Single param + BTC spot on twin y-axis

    def plot_skew_reproduction(result, market_data, maturities_days=None) -> Figure
        # Market IV vs Model IV by moneyness, separate panels per maturity
        # Auto-detects maturities if not provided

    def plot_model_vs_market_scatter(result) -> Figure
        # Scatter with 45° perfect-fit line

    def plot_residuals_heatmap(result, n_moneyness_bins=10, n_ttm_bins=8) -> Figure
        # (moneyness × maturity) → mean error, RdBu_r diverging colormap

    def plot_residuals_histogram(result) -> Figure

    def generate_calibration_table(results, european_format=True) -> str
        # LaTeX table with European formatting, saves .tex + .csv

    def generate_summary_statistics(results) -> str
        # Mean, Std, Min, Max, Median per parameter

    def plot_model_comparison(bs_rmse, heston_rmse, rbergomi_rmse) -> Figure

    def create_full_report(timeseries_results, snapshot_result, market_data, events)
        # Generates all of the above
```

### Color Scheme
```python
COLORS = {'H':'#2E86AB', 'eta':'#A23B72', 'rho':'#F18F01', 'xi':'#C73E1D',
          'market':'#1f77b4', 'model':'#d62728', 'spot':'#2ca02c'}
```
