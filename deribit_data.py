# deribit_loader.py

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

import aiohttp
import polars as pl
from tqdm.asyncio import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data/deribit_loader.log", mode="w"), logging.StreamHandler()]
)
logger = logging.getLogger("DeribitLoader")

# --- DATA SCHEMA (Float64 for rBergomi precision) ---
TRADE_SCHEMA = {
    "instrument_name": pl.String,
    "timestamp": pl.Int64,
    "expiration_timestamp": pl.Int64,
    "strike": pl.Float64,
    "option_type": pl.String,
    "trade_seq": pl.Int64,
    "trade_id": pl.Int64,
    "tick_direction": pl.Int16,
    "price": pl.Float64,
    "iv": pl.Float64,
    "index_price": pl.Float64,
    "direction": pl.String,
    "amount": pl.Float64,
    "contracts": pl.Float64,
    "block_trade_id": pl.String,
    "combo_id": pl.String,
    "liquidation": pl.String
}

class DeribitClient:
    BASE_URL = "https://history.deribit.com/api/v2/public"

    def __init__(self, max_concurrent: int = 15) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "DeribitClient":
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session: await self.session.close()
        await asyncio.sleep(0.250)

    @retry(stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, min=2, max=60), retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
    async def _fetch(self, endpoint: str, params: Dict[str, Any]) -> Any:
        url = f"{self.BASE_URL}/{endpoint}"
        async with self.semaphore:
            if not self.session or self.session.closed: raise RuntimeError("Session invalid")
            async with self.session.get(url, params=params) as response:
                if response.status == 429:
                    raise aiohttp.ClientResponseError(response.request_info, response.history, status=429, message="Rate Limit")
                response.raise_for_status()
                data = await response.json()
                return data.get("result", [])

    async def get_instruments(self, currency: str = "BTC", kind: str = "option") -> pl.DataFrame:
        tasks = [self._fetch("get_instruments", {"currency": currency, "kind": kind, "expired": e}) for e in ["true", "false"]]
        results = await asyncio.gather(*tasks)
        combined = results[0] + results[1]
        if not combined: return pl.DataFrame()
        
        df = pl.DataFrame(combined, infer_schema_length=None)
        cols = ["instrument_name", "creation_timestamp", "expiration_timestamp", "strike", "option_type", "kind", "contract_size", "settlement_period"]
        return df.select([c for c in cols if c in df.columns])

    async def get_trades_chunk(self, name: str, start_ts: int, end_ts: int, count: int = 1000) -> List[Dict[str, Any]]:
        params = {"instrument_name": name, "start_timestamp": start_ts, "end_timestamp": end_ts, "count": count, "sorting": "desc"}
        if name == "BTC-PERPETUAL" and start_ts == 0: params["start_timestamp"] = 1
        response = await self._fetch("get_last_trades_by_instrument_and_time", params)
        return response.get("trades", []) if isinstance(response, dict) else response

class DataManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.trades_dir = base_dir / "option"
        self.list_file = base_dir / "option-list.csv"
        self.trades_dir.mkdir(parents=True, exist_ok=True)

    def save_list(self, df: pl.DataFrame, start_ts: int, limit_ts: int) -> None:
        # Filter the saved list on disk for the thesis date range
        df.filter((pl.col("expiration_timestamp") >= start_ts) & (pl.col("expiration_timestamp") <= limit_ts)).write_csv(self.list_file)

    def load_list(self) -> List[Tuple[str, int, int]]:
        if not self.list_file.exists(): return []
        try:
            return pl.read_csv(self.list_file).select([
                pl.col("instrument_name"),
                pl.col("creation_timestamp").fill_null(0).cast(pl.Int64),
                pl.col("expiration_timestamp").fill_null(0).cast(pl.Int64)
            ]).rows()
        except Exception: return []

    def save_parquet(self, name: str, trades: List[Dict[str, Any]]) -> None:
        if not trades: return
        try:
            normalized = []
            for t in trades:
                nt = t.copy()
                try: nt["tick_direction"] = int(nt.get("tick_direction", 0))
                except: nt["tick_direction"] = 0
                for f in ["block_trade_id", "combo_id", "liquidation", "direction", "option_type"]:
                    if f in nt: nt[f] = str(nt[f]) if nt[f] is not None else None
                if "expiration_timestamp" not in nt: nt["expiration_timestamp"] = None
                normalized.append(nt)

            df = pl.from_dicts(normalized, schema_overrides=TRADE_SCHEMA, infer_schema_length=100)
            df = df.unique(subset=["trade_id"]).sort("timestamp")
            df.write_parquet(self.trades_dir / f"{name}.parquet", compression="zstd")
        except Exception as e: logger.error(f"Save error {name}: {e}")

async def process_instrument(client: DeribitClient, manager: DataManager, instrument: Tuple[str, int, int], user_start_ts: int, user_limit_ts: int) -> None:
    name, start_ts, end_ts = instrument
    
    # 1. Basic filters
    if end_ts > 0 and end_ts < user_start_ts: return
    
    # If NOT Perpetual, use simple logic (single file)
    if name != "BTC-PERPETUAL":
        # ... (Standard logic for options which are small)
        target_end = end_ts
        effective_start = user_start_ts if start_ts == 0 else max(start_ts, user_start_ts)
        if effective_start >= target_end: return
        if (manager.trades_dir / f"{name}.parquet").exists(): return

        # Download everything at once (legacy-style code for options)
        await _download_single_batch(client, manager, name, effective_start, target_end, expiration_ts=end_ts)
        return

    # 2. ADVANCED LOGIC FOR PERPETUAL (Monthly Split)
    logger.info(f"Starting Monthly Split Download for {name}...")
    
    # Calculate months between start and end
    current_date = datetime.fromtimestamp(user_start_ts / 1000)
    end_date_dt = datetime.fromtimestamp(user_limit_ts / 1000)
    
    # Iterate month by month
    while current_date <= end_date_dt:
        # Calculate start and end of current month
        # Current month start
        month_start = datetime(current_date.year, current_date.month, 1)
        month_start_ts = int(month_start.timestamp() * 1000)
        
        # Current month end (next month start - 1ms)
        if current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1)
        
        month_end_ts = int(next_month.timestamp() * 1000) - 1
        
        # Monthly filename: BTC-PERPETUAL_2024-01.parquet
        month_str = month_start.strftime("%Y-%m")
        filename = f"{name}_{month_str}"
        
        # If file already exists, skip to next month
        if not (manager.trades_dir / f"{filename}.parquet").exists():
            logger.info(f"Downloading {filename}...")
            # Download this month only
            await _download_single_batch(client, manager, filename, month_start_ts, month_end_ts, instrument_name_api=name)
        else:
            logger.info(f"Skipping {filename} (already exists)")
        
        # Go to next month
        current_date = next_month

# Helper function (extracts the actual download logic)
async def _download_single_batch(client, manager, save_name, start_ts, end_ts, instrument_name_api=None, expiration_ts=None):
    # If no different API name specified, use the file name (same for options)
    api_name = instrument_name_api if instrument_name_api else save_name

    all_trades = []
    current_end_ts = end_ts
    batch_size = 1000
    seen_ids = set()

    while True:
        try: 
            trades = await client.get_trades_chunk(api_name, start_ts, current_end_ts, batch_size)
        except Exception as e: 
            logger.error(f"Error {save_name}: {e}")
            break
            
        if not trades: break

        new_trades = [t for t in trades if t['trade_id'] not in seen_ids]
        for t in new_trades:
            seen_ids.add(t['trade_id'])
            if expiration_ts is not None:
                t['expiration_timestamp'] = expiration_ts
        all_trades.extend(new_trades)

        min_ts = trades[-1]["timestamp"]
        if min_ts <= start_ts: break
        
        if len(trades) < batch_size and not new_trades: current_end_ts = min_ts - 1
        elif len(trades) < batch_size: break 
        else: current_end_ts = min_ts

        # Limit resets per file (so 3M per month is OK)
        if len(all_trades) > 10_000_000: # Raised slightly for safety
             logger.warning(f"File {save_name} hit 4M limit (Month truncated?)")
             break

    if all_trades:
        all_trades.sort(key=lambda x: x['timestamp'])
        manager.save_parquet(save_name, all_trades)
        logger.info(f"Saved {save_name}: {len(all_trades)} trades")


# --- PUBLIC FUNCTION TO CALL FROM MAIN ---
async def run_download_task(start_date: datetime, end_date: datetime, data_folder: str = "data", max_concurrent: int = 15):
    """
    Main function to import and execute.
    """
    start_ts = int(start_date.timestamp() * 1000)
    limit_ts = int(end_date.timestamp() * 1000)
    base_dir = Path(data_folder)
    
    manager = DataManager(base_dir)
    
    async with DeribitClient(max_concurrent=max_concurrent) as client:
        # 1. Option List
        instruments = manager.load_list()
        if not instruments:
            logger.info("Fetching Option List...")
            df_full = await client.get_instruments(kind="option")
            manager.save_list(df_full, start_ts, limit_ts)
            instruments = manager.load_list()
        
        # 2. Add Perpetual (Thesis Ch. 3)
        all_jobs = [("BTC-PERPETUAL", 0, limit_ts)] + instruments
        
        logger.info(f"Starting download: {len(all_jobs)} instruments from {start_date} to {end_date}")
        
        # 3. Parallel Execution
        tasks = [process_instrument(client, manager, job, start_ts, limit_ts) for job in all_jobs]
        await tqdm.gather(*tasks, desc="Thesis Data Download")
    
    logger.info("All tasks completed.")


# --- POST-PROCESSING: Add expiration_timestamp to existing files ---
def parse_expiration_from_name(instrument_name: str) -> Optional[int]:
    """
    Extract expiration_timestamp from instrument name.
    Format: BTC-28MAR25-50000-C -> 28MAR25 -> timestamp
    Deribit uses 08:00 UTC as expiration time.
    """
    import re

    # Skip perpetuals
    if "PERPETUAL" in instrument_name:
        return None

    # Pattern: BTC-DDMMMYY-STRIKE-TYPE (es. BTC-28MAR25-50000-C)
    match = re.match(r'^[A-Z]+-(\d{1,2})([A-Z]{3})(\d{2})-', instrument_name)
    if not match:
        return None

    day, month_str, year_short = match.groups()

    # Month mapping
    months = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }

    month = months.get(month_str.upper())
    if not month:
        return None

    # Year: 25 -> 2025
    year = 2000 + int(year_short)

    # Deribit expiration: 08:00 UTC
    expiration_dt = datetime(year, month, int(day), 8, 0, 0)
    return int(expiration_dt.timestamp() * 1000)


def postprocess_add_expiration(data_folder: str = "data") -> None:
    """
    Post-processing to add expiration_timestamp to existing parquet files.
    Reads the instrument name and computes the expiration.
    """
    from tqdm import tqdm

    trades_dir = Path(data_folder) / "option"
    if not trades_dir.exists():
        logger.error(f"Directory {trades_dir} not found")
        return

    parquet_files = list(trades_dir.glob("*.parquet"))
    logger.info(f"Post-processing: {len(parquet_files)} files found")

    updated = 0
    skipped = 0

    for file_path in tqdm(parquet_files, desc="Adding expiration_timestamp"):
        try:
            df = pl.read_parquet(file_path)

            # Skip if expiration_timestamp is already populated correctly
            if "expiration_timestamp" in df.columns:
                non_null = df.filter(pl.col("expiration_timestamp").is_not_null())
                if len(non_null) == len(df) and len(df) > 0:
                    skipped += 1
                    continue

            # Get instrument name from file or first row
            instrument_name = file_path.stem  # Nome file senza .parquet

            # For perpetual monthly files (BTC-PERPETUAL_2024-01), skip
            if "PERPETUAL" in instrument_name:
                skipped += 1
                continue

            # Compute expiration from name
            exp_ts = parse_expiration_from_name(instrument_name)
            if exp_ts is None:
                logger.warning(f"Cannot parse expiration from: {instrument_name}")
                skipped += 1
                continue

            # Add/update column
            df = df.with_columns(pl.lit(exp_ts).alias("expiration_timestamp"))

            # Save the updated file
            df.write_parquet(file_path, compression="zstd")
            updated += 1

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Post-processing completed: {updated} files updated, {skipped} skipped")