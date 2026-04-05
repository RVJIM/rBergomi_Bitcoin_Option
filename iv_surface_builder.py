# iv_surface_builder.py
"""
IV Surface Builder for rBergomi calibration.
Builds implied volatility surfaces from Deribit data.

"""

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IVSurfaceBuilder")


# === KEY CRYPTO MARKET EVENTS 2020-2025 ===
# 10 events selected for complete coverage of the thesis period
MARKET_EVENTS = [
    {
        "date": datetime(2020, 3, 12),
        "name": "COVID Black Thursday",
        "description": "BTC crashes 50% in 24h, from $8k to $3.8k"
    },
    {
        "date": datetime(2020, 5, 11),
        "name": "Third Bitcoin Halving",
        "description": "Block reward reduced from 12.5 to 6.25 BTC"
    },
    {
        "date": datetime(2021, 5, 19),
        "name": "China Mining Ban Crash",
        "description": "BTC drops 30% on China crackdown news"
    },
    {
        "date": datetime(2021, 11, 10),
        "name": "BTC ATH $69k",
        "description": "Peak of 2021 bull market"
    },
    {
        "date": datetime(2022, 5, 9),
        "name": "LUNA/UST Collapse",
        "description": "Terra ecosystem implodes, BTC drops to $26k"
    },
    {
        "date": datetime(2022, 11, 8),
        "name": "FTX Bankruptcy",
        "description": "FTX collapse, BTC crashes to $15.5k"
    },
    {
        "date": datetime(2024, 1, 10),
        "name": "Bitcoin Spot ETF Approval",
        "description": "SEC approves first US spot Bitcoin ETFs"
    },
    {
        "date": datetime(2024, 4, 20),
        "name": "Fourth Bitcoin Halving",
        "description": "Block reward reduced from 6.25 to 3.125 BTC"
    },
    {
        "date": datetime(2024, 12, 5),
        "name": "BTC Breaks $100k",
        "description": "Bitcoin reaches six figures for first time"
    },
    {
        "date": datetime(2025, 1, 20),
        "name": "Trump Inauguration & Crypto EO",
        "description": "Pro-crypto executive orders, BTC rallies"
    }
]


class IVSurfaceBuilder:
    """
    Builds implied volatility surfaces from Deribit parquet files.
    Optimized for rBergomi calibration with put-call parity.
    """
    
    def __init__(self, data_dir: str = "data/option"):
        self.data_dir = Path(data_dir)
        self._index: Optional[pl.DataFrame] = None
        self._index_path = self.data_dir.parent / "option_index.parquet"
    
    def build_index(self, force_rebuild: bool = False) -> pl.DataFrame:
        """
        Build a lightweight index of all files for fast queries.
        """
        if self._index_path.exists() and not force_rebuild:
            logger.info(f"Loading existing index from {self._index_path}")
            self._index = pl.read_parquet(self._index_path)
            logger.info(f"Index loaded: {self._index.height} instruments")
            return self._index
        
        logger.info("Building option index (one-time operation)...")
        records = []
        files = list(self.data_dir.glob("BTC-*.parquet"))
        
        for i, f in enumerate(files):
            if i % 5000 == 0:
                logger.info(f"Scanning file {i}/{len(files)}...")
            
            if "PERPETUAL" in f.name:
                continue
            
            # Parse instrument name: BTC-28MAR25-100000-C
            parts = f.stem.split("-")
            if len(parts) != 4:
                continue
            
            try:
                expiry_str = parts[1]  # 28MAR25
                strike = float(parts[2])
                opt_type = parts[3]  # C or P
                
                # Parse expiry date (handles both 2-digit and 4-digit year)
                try:
                    expiry_dt = datetime.strptime(expiry_str, "%d%b%y")
                except:
                    try:
                        expiry_dt = datetime.strptime(expiry_str, "%d%b%Y")
                    except:
                        continue
                
                # Quick scan for timestamp range
                df_scan = pl.scan_parquet(f).select(["timestamp", "amount"]).collect()
                if df_scan.height == 0:
                    continue
                
                records.append({
                    "file": f.name,
                    "expiry": expiry_dt,
                    "expiry_ts": int(expiry_dt.timestamp() * 1000),
                    "strike": strike,
                    "option_type": opt_type,
                    "min_ts": df_scan["timestamp"].min(),
                    "max_ts": df_scan["timestamp"].max(),
                    "n_trades": df_scan.height,
                    "total_volume": df_scan["amount"].sum()
                })
            except Exception as e:
                continue
        
        self._index = pl.DataFrame(records)
        self._index.write_parquet(self._index_path)
        logger.info(f"Index built and saved: {len(records)} instruments")
        return self._index
    
    def get_iv_surface(
        self,
        target_time: datetime,
        window_hours: float = 2.0,
        min_volume: float = 0.1,  # Minimum volume in BTC
        moneyness_range: Tuple[float, float] = (0.5, 2.0),
        min_ttm_days: int = 1,
        max_ttm_days: int = 365,
        use_put_call_parity: bool = True
    ) -> pl.DataFrame:
        """
        Extract the IV surface for a given time instant.
        """
        if self._index is None:
            self.build_index()
        
        target_ts = int(target_time.timestamp() * 1000)
        window_ms = int(window_hours * 3600 * 1000)
        start_ts = target_ts - window_ms
        end_ts = target_ts + window_ms
        
        # Filter index for relevant files
        relevant = self._index.filter(
            (pl.col("max_ts") >= start_ts) &
            (pl.col("min_ts") <= end_ts) &
            (pl.col("expiry_ts") > target_ts) &
            (pl.col("total_volume") >= min_volume)
        )
        
        if relevant.height == 0:
            logger.warning(f"No data found for {target_time}")
            return pl.DataFrame()
        
        # Required columns in parquet files (strike and option_type come from index)
        REQUIRED_COLS = ["timestamp", "expiration_timestamp", "iv", "index_price", "amount"]

        # Load and normalize trades
        all_rows = []  # List of dicts instead of DataFrame for safety

        for row in relevant.iter_rows(named=True):
            try:
                df = pl.read_parquet(self.data_dir / row["file"])

                # Verify all required columns exist
                if not all(c in df.columns for c in REQUIRED_COLS):
                    continue

                # Filter before converting (more efficient)
                df_filtered = df.filter(
                    (pl.col("timestamp") >= start_ts) &
                    (pl.col("timestamp") <= end_ts) &
                    (pl.col("iv").is_not_null()) &
                    (pl.col("iv") > 0) &
                    (pl.col("iv") < 500) &
                    (pl.col("amount") > 0)
                )

                if df_filtered.height == 0:
                    continue

                # Strike and option_type from index (not present in trade files)
                strike_val = float(row["strike"])
                opt_type = row["option_type"]

                # Convert to list of dicts (guarantees consistent schema)
                for trade in df_filtered.iter_rows(named=True):
                    all_rows.append({
                        "timestamp": int(trade["timestamp"]),
                        "strike": strike_val,
                        "expiration_timestamp": int(trade["expiration_timestamp"]),
                        "iv": float(trade["iv"]),
                        "index_price": float(trade["index_price"]),
                        "amount": float(trade["amount"]),
                        "option_type": opt_type
                    })
                        
            except Exception as e:
                continue
        
        if not all_rows:
            logger.warning(f"No valid trades for {target_time}")
            return pl.DataFrame()
        
        # Create DataFrame from list of dicts (guaranteed identical schema)
        combined = pl.DataFrame(all_rows)
        
        # Aggregate by (strike, expiry, option_type) - VWAP IV
        surface = combined.group_by(["strike", "expiration_timestamp", "option_type"]).agg([
            (pl.col("iv") * pl.col("amount")).sum().alias("iv_x_vol"),
            pl.col("amount").sum().alias("volume"),
            pl.col("index_price").mean().alias("spot"),
            pl.len().alias("n_trades")
        ]).with_columns(
            (pl.col("iv_x_vol") / pl.col("volume")).alias("iv_vwap")
        ).drop("iv_x_vol")
        
        # Apply put-call parity: weighted average of call and put for same (K, τ)
        if use_put_call_parity:
            surface = surface.group_by(["strike", "expiration_timestamp"]).agg([
                (pl.col("iv_vwap") * pl.col("volume")).sum() / pl.col("volume").sum(),
                pl.col("volume").sum().alias("total_volume"),
                pl.col("spot").mean().alias("spot"),
                pl.col("n_trades").sum().alias("total_trades")
            ]).rename({"iv_vwap": "iv"})
        else:
            surface = surface.rename({"iv_vwap": "iv", "volume": "total_volume", "n_trades": "total_trades"})
        
        # Compute derived metrics
        surface = surface.with_columns([
            # TTM in years (365.25 for average leap year)
            ((pl.col("expiration_timestamp") - target_ts) / (365.25 * 24 * 3600 * 1000)).alias("ttm"),
            # Moneyness K/S
            (pl.col("strike") / pl.col("spot")).alias("moneyness"),
            # Log-moneyness for smile analysis
            (pl.col("strike") / pl.col("spot")).log().alias("log_moneyness"),
            # Forward log-moneyness (scaled by sqrt(ttm)) for rBergomi
            pl.lit(target_ts).alias("observation_ts")
        ])
        
        # Compute forward log-moneyness k = log(K/S) / sqrt(τ)
        surface = surface.with_columns(
            (pl.col("log_moneyness") / pl.col("ttm").sqrt()).alias("forward_log_moneyness")
        )
        
        # Apply final filters
        surface = surface.filter(
            (pl.col("moneyness") >= moneyness_range[0]) &
            (pl.col("moneyness") <= moneyness_range[1]) &
            (pl.col("ttm") >= min_ttm_days / 365.25) &
            (pl.col("ttm") <= max_ttm_days / 365.25) &
            (pl.col("total_volume") >= min_volume)
        ).sort(["ttm", "strike"])
        
        if surface.height > 0:
            logger.info(f"Surface @ {target_time}: {surface.height} points, "
                       f"TTM range [{surface['ttm'].min():.3f}, {surface['ttm'].max():.3f}] years")
        else:
            logger.warning(f"No data found for {target_time}")

        return surface
    
    def get_calibration_dates(
        self,
        start: datetime,
        end: datetime,
        frequency: str = "daily",
        hour: int = 8  # 08:00 UTC - globally active market
    ) -> List[datetime]:
        """
        Generate calibration dates for the period.
        """
        dates = []
        current = start
        
        delta = {
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30)
        }.get(frequency, timedelta(days=1))
        
        while current <= end:
            calibration_time = current.replace(hour=hour, minute=0, second=0, microsecond=0)
            dates.append(calibration_time)
            current += delta
        
        return dates

    def get_high_volume_dates(
        self,
        dates: List[datetime],
        top_pct: float = 0.25,
        window_hours: float = 4.0,
        min_volume: float = 0.05
    ) -> List[datetime]:
        """
        Filter dates to keep only the top `top_pct` fraction ranked by trading activity.

        Uses the number of active instruments (with trades overlapping the date window)
        as a proxy for daily volume. Dates with more active instruments have deeper,
        more liquid option chains, producing higher-quality calibration surfaces.

        Args:
            dates: candidate calibration dates
            top_pct: fraction to keep (default 0.25 = top 25%)
            window_hours: time window around each date for overlap check
            min_volume: minimum total_volume filter for instruments

        Returns:
            Filtered list of dates, sorted chronologically
        """
        if self._index is None:
            self.build_index()

        window_ms = int(window_hours * 3600 * 1000)
        date_scores = []

        for dt in dates:
            target_ts = int(dt.timestamp() * 1000)
            start_ts = target_ts - window_ms
            end_ts = target_ts + window_ms

            relevant = self._index.filter(
                (pl.col("max_ts") >= start_ts) &
                (pl.col("min_ts") <= end_ts) &
                (pl.col("expiry_ts") > target_ts) &
                (pl.col("total_volume") >= min_volume)
            )
            date_scores.append((dt, relevant.height))

        # Sort by score descending, keep top fraction
        date_scores.sort(key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(date_scores) * top_pct))
        selected = [dt for dt, _ in date_scores[:n_keep]]

        # Return in chronological order
        selected.sort()
        logger.info(f"High-volume filter: selected {len(selected)}/{len(dates)} dates "
                    f"(top {top_pct:.0%})")
        return selected

    def get_event_dates(self) -> List[Dict[str, Any]]:
        """Return the list of key market events."""
        return MARKET_EVENTS.copy()
    
    def export_for_rbergomi(self, surface: pl.DataFrame) -> Dict[str, np.ndarray]:
        """
        Format the surface as input for the rBergomi calibrator.

        Returns:
            dict with numpy arrays:
            - S: spot price (scalar)
            - K: strikes array
            - tau: time to maturity array (in years)
            - iv_market: implied volatilities (in decimals, e.g. 0.80 = 80%)
            - moneyness: K/S array
            - log_moneyness: log(K/S) array
            - forward_log_moneyness: log(K/S)/sqrt(tau) array
            - weights: calibration weights (volume-based)
        """
        if surface.height == 0:
            return {}
        
        # Normalize weights (volume-weighted)
        volumes = surface["total_volume"].to_numpy()
        weights = volumes / volumes.sum()
        
        # IV from percentage to decimal
        iv_decimal = surface["iv"].to_numpy() / 100.0
        
        return {
            "S": float(surface["spot"].mean()),
            "K": surface["strike"].to_numpy(),
            "tau": surface["ttm"].to_numpy(),
            "iv_market": iv_decimal,
            "moneyness": surface["moneyness"].to_numpy(),
            "log_moneyness": surface["log_moneyness"].to_numpy(),
            "forward_log_moneyness": surface["forward_log_moneyness"].to_numpy(),
            "weights": weights,
            "n_points": surface.height
        }


class CalibrationRunner:
    """
    Runs daily calibration and saves results.
    """
    
    def __init__(self, builder: IVSurfaceBuilder, output_dir: str = "calibration_results"):
        self.builder = builder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_daily_calibration(
        self,
        start: datetime,
        end: datetime,
        min_points: int = 20  # Minimum points for valid calibration
    ) -> pl.DataFrame:
        """
        Run daily calibration and save results.

        Returns:
            DataFrame with time series of calibrated parameters
        """
        dates = self.builder.get_calibration_dates(start, end, frequency="daily")
        results = []
        
        logger.info(f"Starting daily calibration: {len(dates)} dates")
        
        for i, date in enumerate(dates):
            if i % 30 == 0:
                logger.info(f"Processing {date.strftime('%Y-%m-%d')} ({i}/{len(dates)})")
            
            try:
                surface = self.builder.get_iv_surface(
                    target_time=date,
                    window_hours=2.0,
                    min_volume=0.1,
                    moneyness_range=(0.7, 1.4),  # Focus ATM ±40%
                    min_ttm_days=7,
                    max_ttm_days=180
                )
                
                if surface.height < min_points:
                    continue
                
                calib_input = self.builder.export_for_rbergomi(surface)
                
                # The actual rBergomi calibration will go here
                # For now, save the surface statistics
                results.append({
                    "date": date,
                    "spot": calib_input["S"],
                    "n_points": calib_input["n_points"],
                    "iv_atm_mean": float(np.mean(calib_input["iv_market"])),
                    "iv_atm_std": float(np.std(calib_input["iv_market"])),
                    "ttm_min": float(calib_input["tau"].min()),
                    "ttm_max": float(calib_input["tau"].max()),
                    "moneyness_min": float(calib_input["moneyness"].min()),
                    "moneyness_max": float(calib_input["moneyness"].max()),
                    # Placeholder for rBergomi parameters (to be implemented)
                    "H": None,  # Hurst exponent
                    "eta": None,  # Vol-of-vol
                    "rho": None,  # Correlation
                    "calibration_error": None  # RMSE
                })
                
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue
        
        df_results = pl.DataFrame(results)
        
        # Save results
        output_file = self.output_dir / "daily_calibration.parquet"
        df_results.write_parquet(output_file)
        logger.info(f"Saved {len(results)} calibration results to {output_file}")
        
        return df_results
    
    def run_event_calibration(self) -> pl.DataFrame:
        """
        Calibration for key market events.
        Generates data for static plots in the thesis.
        """
        events = self.builder.get_event_dates()
        results = []
        
        logger.info(f"Calibrating {len(events)} market events")
        
        for event in events:
            date = event["date"]
            
            # Pre-event (day before)
            pre_date = date - timedelta(days=1)
            # Post-event (day after)
            post_date = date + timedelta(days=1)
            
            for label, d in [("pre", pre_date), ("event", date), ("post", post_date)]:
                try:
                    surface = self.builder.get_iv_surface(
                        target_time=d.replace(hour=12),
                        window_hours=4.0,  # Wider window for events
                        min_volume=0.05,
                        moneyness_range=(0.6, 1.6),
                        min_ttm_days=3,
                        max_ttm_days=90
                    )
                    
                    if surface.height < 10:
                        continue
                    
                    calib_input = self.builder.export_for_rbergomi(surface)
                    
                    results.append({
                        "event_name": event["name"],
                        "event_date": date,
                        "observation": label,
                        "observation_date": d,
                        "spot": calib_input["S"],
                        "n_points": calib_input["n_points"],
                        "iv_mean": float(np.mean(calib_input["iv_market"])),
                        "iv_std": float(np.std(calib_input["iv_market"])),
                        # Full surface for plotting
                        "strikes": calib_input["K"].tolist(),
                        "ttms": calib_input["tau"].tolist(),
                        "ivs": calib_input["iv_market"].tolist()
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {event['name']} ({label}): {e}")
                    continue
        
        df_events = pl.DataFrame(results)
        
        output_file = self.output_dir / "event_calibration.parquet"
        df_events.write_parquet(output_file)
        logger.info(f"Saved {len(results)} event calibrations to {output_file}")
        
        return df_events


# NOTE: This module should be run ONLY via main_c.py
# Example: python main_c.py --index