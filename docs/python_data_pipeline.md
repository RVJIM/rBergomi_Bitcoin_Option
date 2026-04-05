# Python Context: Data Pipeline

## deribit_data.py (349 lines)
Async Deribit API client for downloading BTC option trades.

### TRADE_SCHEMA
```python
{"instrument_name": pl.String, "timestamp": pl.Int64,
 "expiration_timestamp": pl.Int64, "strike": pl.Float64,
 "option_type": pl.String, "trade_seq": pl.Int64,
 "trade_id": pl.Int64, "tick_direction": pl.Int16,
 "price": pl.Float64, "iv": pl.Float64, "index_price": pl.Float64,
 "direction": pl.String, "amount": pl.Float64, "contracts": pl.Float64,
 "block_trade_id": pl.String, "combo_id": pl.String, "liquidation": pl.String}
```

### Classes & Functions
```python
class DeribitClient:
    BASE_URL = "https://history.deribit.com/api/v2/public"
    def __init__(self, max_concurrent: int = 15)
    async def get_instruments(currency="BTC", kind="option") -> pl.DataFrame
    async def get_trades_chunk(name, start_ts, end_ts, count=1000) -> List[Dict]

class DataManager:
    def __init__(self, base_dir: Path)  # trades_dir = base_dir/"option"
    def save_list(df, start_ts, limit_ts)  # Filter+save instrument list
    def load_list() -> List[Tuple[str, int, int]]
    def save_parquet(name, trades)  # Dedupe by trade_id, zstd compression

async def process_instrument(client, manager, instrument, user_start_ts, user_limit_ts)
    # Options: single file per instrument
    # BTC-PERPETUAL: monthly split (BTC-PERPETUAL_2024-01.parquet)

async def run_download_task(start_date, end_date, data_folder="data", max_concurrent=15)
    # Main entry point. Downloads instruments list + all trades

def parse_expiration_from_name(instrument_name) -> Optional[int]
    # "BTC-28MAR25-50000-C" → timestamp (08:00 UTC)

def postprocess_add_expiration(data_folder="data")
    # Add expiration_timestamp column to existing parquet files
```

---

## data_cleaning.py (296 lines)
File/folder cleanup utility. Run via `main_c.py --clean-*`.

### CONFIG
```python
{"temp_patterns": ["*.tmp","*.pyc","*.pyo","*~",".DS_Store","Thumbs.db"],
 "cache_dirs": ["__pycache__",".pytest_cache",".mypy_cache"],
 "valid_results_dirs": ["fat_tails_kurtosis","implied_volatility_smile",
                        "volatility_term_structure","inverse_options"]}
```

### Public Functions
```python
def run_clean_temp()       # Remove temp files + __pycache__
def run_clean_results()    # Remove timestamp/empty Results/ folders
def run_clean_all()        # Both
def run_show_status()      # Print current file/folder status
```

---

## iv_surface_builder.py (554 lines)
Builds IV surfaces from Deribit parquet data for rBergomi calibration.

### MARKET_EVENTS (10 events, 2020-2025)
COVID crash, halvings, China ban, ATH $69k, LUNA, FTX, ETF approval, 4th halving, $100k, Trump EO.

### Classes
```python
class IVSurfaceBuilder:
    def __init__(self, data_dir="data/option")
    def build_index(force_rebuild=False) -> pl.DataFrame
        # One-time scan: file→(expiry, strike, opt_type, min_ts, max_ts, n_trades, total_volume)
        # Saved to data/option_index.parquet

    def get_iv_surface(target_time, window_hours=2.0, min_volume=0.1,
                       moneyness_range=(0.5,2.0), min_ttm_days=1,
                       max_ttm_days=365, use_put_call_parity=True) -> pl.DataFrame
        # Returns: strike, expiration_timestamp, iv (VWAP %), total_volume,
        #          spot, ttm (years), moneyness, log_moneyness, forward_log_moneyness

    def get_calibration_dates(start, end, frequency="daily", hour=8) -> List[datetime]

    def get_high_volume_dates(dates, top_pct=0.25, window_hours=4.0) -> List[datetime]
        # Keep top 25% by active instrument count

    def export_for_rbergomi(surface) -> Dict[str, np.ndarray]
        # Returns: S (scalar), K, tau, iv_market (decimal), moneyness,
        #          log_moneyness, forward_log_moneyness, weights, n_points

class CalibrationRunner:
    def __init__(self, builder, output_dir="calibration_results")
    def run_daily_calibration(start, end, min_points=20) -> pl.DataFrame
        # Placeholder for rBergomi (H,eta,rho columns = None)
    def run_event_calibration() -> pl.DataFrame
        # Pre/event/post surfaces for 10 market events
```

---

## iv_visualizer.py (673 lines)
IV surface visualization and Delta comparison plots.

### Classes
```python
class IVSurfaceVisualizer:
    def __init__(self, output_dir="figures")
    def plot_iv_surface_3d(surface, title, save_name, show) -> Figure
    def plot_iv_smile(surface, ttm_target=0.0833, ...) -> Figure
    def plot_term_structure(surface, moneyness_target=1.0, ...) -> Figure
    def plot_calibration_timeseries(results, param, events, ...) -> Figure
    def plot_event_comparison(event_data, event_name, ...) -> Figure
    def create_surface_animation(builder, start_date, end_date, fps=2, ...) -> str

class DeltaComparisonPlotter:
    def __init__(self, output_dir="figures")
    # Static methods: _bs_put_price, _bs_call_price, _delta_traditional_put/call
    # Instance methods: _delta_inverse_put/call (BTC-denominated)
    def plot_delta_comparison(K, T, r, sigma, S_range, option_type, ...) -> Figure
    def plot_all_greeks_comparison(K, T, r, sigma, ...) -> Figure
```

### Plot Style
```python
plt.style.use('seaborn-v0_8-whitegrid')
font.family='serif', font.size=11, savefig.dpi=300
colors: call='#2E86AB', put='#A23B72', atm='#F18F01', spot='#C73E1D'
```
