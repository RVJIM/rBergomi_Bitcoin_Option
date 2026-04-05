"""
populate_tables.py
==================
Reads calibration CSVs from Results/calibration/tables/ and generates
ready-to-use LaTeX table files for Chapter 3.

Output: Results/calibration/latex_tables/
    event_luna.tex          LUNA event (day-1 / day / day+1) x 3 methods
    event_ftx.tex
    event_svb.tex
    event_etf.tex
    event_halving.tex
    event_btc100k.tex
    event_trump.tex
    baseline_low.tex        Low-IV baseline dates x 3 methods
    baseline_medium.tex
    baseline_high.tex
    cross_method.tex        Aggregate RMSE / speed comparison (all dates)
    speed_accuracy.tex      Speed-accuracy summary table

Each .tex file contains a self-contained tabular environment (no \begin{table}
wrapper) so it can be included with \input{} inside any table float.

Usage:
    python populate_tables.py
"""

import sys
import logging
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("Results/tables_population.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("PopulateTables")

# ── Paths ─────────────────────────────────────────────────────────────────────
TABLES_DIR = Path("Results/calibration/tables")
LATEX_DIR  = Path("Results/calibration/latex_tables")

# ── Date definitions (mirrors run_all_calibrations.py) ───────────────────────
EVENT_GROUPS = {
    "luna":    {"pre": "20220508", "event": "20220509", "post": "20220510",
                "name": "LUNA/UST collapse", "date_str": "9 May 2022"},
    "ftx":     {"pre": "20221107", "event": "20221108", "post": "20221109",
                "name": "FTX bankruptcy",    "date_str": "8 Nov 2022"},
    "svb":     {"pre": "20230309", "event": "20230310", "post": "20230311",
                "name": "SVB banking crisis","date_str": "10 Mar 2023"},
    "etf":     {"pre": "20240109", "event": "20240110", "post": "20240111",
                "name": "Bitcoin Spot ETF",  "date_str": "10 Jan 2024"},
    "halving": {"pre": "20240419", "event": "20240420", "post": "20240421",
                "name": "Fourth halving",    "date_str": "20 Apr 2024"},
    "btc100k": {"pre": "20241204", "event": "20241205", "post": "20241206",
                "name": "BTC \\$100k",       "date_str": "5 Dec 2024"},
    "trump":   {"pre": "20250119", "event": "20250120", "post": "20250121",
                "name": "Trump inauguration","date_str": "20 Jan 2025"},
}

BASELINE_GROUPS = {
    "low": {
        "20230812": "Calm summer 2023",
        "20230930": "Pre-rally 2023",
        "20240210": "Post-ETF rally",
    },
    "medium": {
        "20221217": "Crypto winter",
        "20220702": "Bear market 2022",
        "20240914": "Autumn 2024",
    },
    "high": {
        "20220618": "Post-LUNA recovery",
        "20221116": "Post-FTX recovery",
        "20250203": "Bull run 2025",
    },
}

from src.config.methods import METHOD_LABELS_SHORT as METHOD_LABELS

METHODS = ["cholesky_euler", "hybrid_euler", "hybrid_mixed"]


# =============================================================================
#  DATA LOADING
# =============================================================================

def load_all_results() -> pl.DataFrame:
    """Load all CSV results for all methods into a single DataFrame."""
    frames = []
    for method in METHODS:
        method_dir = TABLES_DIR / method
        if not method_dir.exists():
            logger.warning(f"Folder not found: {method_dir}")
            continue
        for f in sorted(method_dir.glob(f"calibration_*_{method}.csv")):
            try:
                df = pl.read_csv(f)
                if "method_name" not in df.columns:
                    df = df.with_columns(pl.lit(method).alias("method_name"))
                frames.append(df)
            except Exception as e:
                logger.warning(f"  Could not read {f.name}: {e}")
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="diagonal")
    # Normalise date column to YYYYMMDD string for consistent lookups
    combined = combined.with_columns(
        pl.col("date").cast(pl.Utf8).str.slice(0, 10).str.replace_all("-", "").alias("date_key")
    )
    return combined


def get_row(df: pl.DataFrame, date_str: str, method: str) -> Optional[Dict]:
    """Return the calibration result for a given date and method."""
    mask = (df["date_key"] == date_str) & (df["method_name"] == method)
    sub  = df.filter(mask)
    if sub.is_empty():
        return None
    return sub.row(0, named=True)


# =============================================================================
#  FORMATTING HELPERS
# =============================================================================

def fmt_eu(val: float, decimals: int = 4) -> str:
    """European decimal format (comma separator)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}".replace(".", ",")


def fmt_pct(val: float) -> str:
    """Format as percentage points with comma."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    return f"{val:.2f}".replace(".", ",") + "\\,pp"


def fmt_time(val: float) -> str:
    """Format seconds."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    if val >= 60:
        return f"{val/60:.1f}\\,min".replace(".", ",")
    return f"{val:.0f}\\,s"


def missing_row(method: str) -> str:
    """Placeholder row when calibration result is missing."""
    lbl = METHOD_LABELS[method]
    return f"  {lbl} & --- & --- & --- & --- & --- \\\\\n"


# =============================================================================
#  TABLE GENERATORS
# =============================================================================

def make_event_table(df: pl.DataFrame, event_key: str) -> str:
    """
    Build a LaTeX tabular for one event (day-1 / day / day+1 x 3 methods).
    Returns the full tabular block (without table float wrapper).
    """
    ev   = EVENT_GROUPS[event_key]
    rows = [("Day$-$1", ev["pre"]),
            ("Day",     ev["event"]),
            ("Day$+$1", ev["post"])]

    # Header
    lines = []
    lines.append("% " + "─" * 66)
    lines.append(f"% Event: {ev['name']}  ({ev['date_str']})")
    lines.append("% " + "─" * 66)
    lines.append("\\begin{tabular}{l l r r r r r}")
    lines.append("  \\toprule")
    lines.append("  \\textbf{Day} & \\textbf{Method} & \\textbf{$H$} & "
                 "\\textbf{$\\eta$} & \\textbf{$\\rho$} & "
                 "\\textbf{RMSE} & \\textbf{Time} \\\\")
    lines.append("  \\midrule")

    for day_label, date_str in rows:
        first = True
        for method in METHODS:
            r = get_row(df, date_str, method)
            day_col = day_label if first else ""
            first   = False
            if r is None:
                lines.append(f"  {day_col} & {METHOD_LABELS[method]} & "
                             f"--- & --- & --- & --- & --- \\\\")
            else:
                lines.append(
                    f"  {day_col} & {METHOD_LABELS[method]} & "
                    f"{fmt_eu(r['H'], 4)} & "
                    f"{fmt_eu(r['eta'], 3)} & "
                    f"{fmt_eu(r['rho'], 3)} & "
                    f"{fmt_pct(r['rmse_pp'])} & "
                    f"{fmt_time(r['time_seconds'])} \\\\"
                )
        if day_label != "Day$+$1":
            lines.append("  \\addlinespace[3pt]")

    lines.append("  \\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def make_baseline_table(df: pl.DataFrame, regime: str) -> str:
    """
    Build a LaTeX tabular for one IV-regime baseline group (4 dates x 3 methods).
    """
    group = BASELINE_GROUPS[regime]
    regime_labels = {"low": "Low IV ($\\lesssim 45\\%$)",
                     "medium": "Medium IV ($45$--$70\\%$)",
                     "high": "High IV ($\\gtrsim 70\\%$)"}

    lines = []
    lines.append("% " + "─" * 66)
    lines.append(f"% Baseline: {regime_labels[regime]}")
    lines.append("% " + "─" * 66)
    lines.append("\\begin{tabular}{l l r r r r r}")
    lines.append("  \\toprule")
    lines.append("  \\textbf{Date} & \\textbf{Method} & \\textbf{$H$} & "
                 "\\textbf{$\\eta$} & \\textbf{$\\rho$} & "
                 "\\textbf{RMSE} & \\textbf{Time} \\\\")
    lines.append("  \\midrule")

    for i, (date_str, date_label) in enumerate(group.items()):
        first = True
        for method in METHODS:
            r = get_row(df, date_str, method)
            lbl_col = date_label if first else ""
            first   = False
            if r is None:
                lines.append(f"  {lbl_col} & {METHOD_LABELS[method]} & "
                             f"--- & --- & --- & --- & --- \\\\")
            else:
                lines.append(
                    f"  {lbl_col} & {METHOD_LABELS[method]} & "
                    f"{fmt_eu(r['H'], 4)} & "
                    f"{fmt_eu(r['eta'], 3)} & "
                    f"{fmt_eu(r['rho'], 3)} & "
                    f"{fmt_pct(r['rmse_pp'])} & "
                    f"{fmt_time(r['time_seconds'])} \\\\"
                )
        if i < len(group) - 1:
            lines.append("  \\addlinespace[3pt]")

    lines.append("  \\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def make_cross_method_table(df: pl.DataFrame) -> str:
    """
    Aggregate cross-method comparison: mean/median RMSE and mean time
    split by crisis (event) vs calm (low-IV baseline).
    """
    # Collect event dates and calm dates
    event_dates = set()
    for ev in EVENT_GROUPS.values():
        event_dates.update([ev["pre"], ev["event"], ev["post"]])
    calm_dates = set(BASELINE_GROUPS["low"].keys())

    lines = []
    lines.append("% Cross-method aggregate comparison")
    lines.append("\\begin{tabular}{l r r r r r}")
    lines.append("  \\toprule")
    lines.append("  \\textbf{Method} & \\textbf{Mean RMSE} & \\textbf{Med RMSE} & "
                 "\\textbf{RMSE (event)} & \\textbf{RMSE (calm)} & \\textbf{Mean time} \\\\")
    lines.append("  \\midrule")

    for method in METHODS:
        sub = df.filter(pl.col("method_name") == method)
        if sub.is_empty():
            lines.append(f"  {METHOD_LABELS[method]} & --- & --- & --- & --- & --- \\\\")
            continue

        ev_sub    = sub.filter(pl.col("date_key").is_in(list(event_dates)))
        calm_sub  = sub.filter(pl.col("date_key").is_in(list(calm_dates)))

        mean_rmse = float(sub["rmse_pp"].mean()) if "rmse_pp" in sub.columns else float("nan")
        med_rmse  = float(sub["rmse_pp"].median()) if "rmse_pp" in sub.columns else float("nan")
        ev_rmse   = float(ev_sub["rmse_pp"].mean()) if not ev_sub.is_empty() else float("nan")
        calm_rmse = float(calm_sub["rmse_pp"].mean()) if not calm_sub.is_empty() else float("nan")
        mean_time = float(sub["time_seconds"].mean()) if "time_seconds" in sub.columns else float("nan")

        lines.append(
            f"  {METHOD_LABELS[method]} & "
            f"{fmt_pct(mean_rmse)} & "
            f"{fmt_pct(med_rmse)} & "
            f"{fmt_pct(ev_rmse)} & "
            f"{fmt_pct(calm_rmse)} & "
            f"{fmt_time(mean_time)} \\\\"
        )

    lines.append("  \\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def make_speed_accuracy_table(df: pl.DataFrame) -> str:
    """
    Speed-accuracy summary: method, mean time, mean RMSE, speedup vs slowest.
    """
    times = {}
    rmses = {}
    for method in METHODS:
        sub = df.filter(pl.col("method_name") == method)
        if not sub.is_empty():
            times[method] = float(sub["time_seconds"].mean()) if "time_seconds" in sub.columns else float("nan")
            rmses[method] = float(sub["rmse_pp"].mean()) if "rmse_pp" in sub.columns else float("nan")
        else:
            times[method] = float("nan")
            rmses[method] = float("nan")

    # Slowest method time for speedup reference
    max_time = max((t for t in times.values() if not np.isnan(t)), default=1.0)

    lines = []
    lines.append("% Speed-accuracy summary")
    lines.append("\\begin{tabular}{l r r r}")
    lines.append("  \\toprule")
    lines.append("  \\textbf{Method} & \\textbf{Mean time / snapshot} & "
                 "\\textbf{Mean RMSE} & \\textbf{Speedup} \\\\")
    lines.append("  \\midrule")

    for method in METHODS:
        t = times[method]
        speedup = max_time / t if not np.isnan(t) and t > 0 else float("nan")
        speedup_str = f"{speedup:.1f}$\\times$".replace(".", ",") if not np.isnan(speedup) else "---"
        lines.append(
            f"  {METHOD_LABELS[method]} & "
            f"{fmt_time(t)} & "
            f"{fmt_pct(rmses[method])} & "
            f"{speedup_str} \\\\"
        )

    lines.append("  \\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# =============================================================================
#  MAIN
# =============================================================================

def run_populate_tables():
    logger.info("=" * 70)
    logger.info("  TABLE POPULATION — rBergomi Chapter 3")
    logger.info("=" * 70)

    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load all results
    logger.info("\n[Step 1] Loading all calibration CSVs...")
    df = load_all_results()
    if df.is_empty():
        logger.error("No calibration results found. Run run_all_calibrations.py first.")
        return
    logger.info(f"  Loaded {len(df)} rows across {df['method_name'].n_unique()} methods.")

    # 1. Event tables
    logger.info("\n[Step 2] Generating event tables...")
    for event_key in EVENT_GROUPS:
        content = make_event_table(df, event_key)
        out = LATEX_DIR / f"event_{event_key}.tex"
        out.write_text(content, encoding="utf-8")
        logger.info(f"  -> {out.name}")

    # 2. Baseline tables
    logger.info("\n[Step 3] Generating baseline IV-regime tables...")
    for regime in ("low", "medium", "high"):
        content = make_baseline_table(df, regime)
        out = LATEX_DIR / f"baseline_{regime}.tex"
        out.write_text(content, encoding="utf-8")
        logger.info(f"  -> {out.name}")

    # 3. Cross-method comparison
    logger.info("\n[Step 4] Generating cross-method comparison table...")
    content = make_cross_method_table(df)
    out = LATEX_DIR / "cross_method.tex"
    out.write_text(content, encoding="utf-8")
    logger.info(f"  -> {out.name}")

    # 4. Speed-accuracy table
    logger.info("\n[Step 5] Generating speed-accuracy table...")
    content = make_speed_accuracy_table(df)
    out = LATEX_DIR / "speed_accuracy.tex"
    out.write_text(content, encoding="utf-8")
    logger.info(f"  -> {out.name}")

    logger.info("\n" + "=" * 70)
    logger.info("  Table population complete.")
    logger.info(f"  Output: {LATEX_DIR}/")
    logger.info("\n  To include in Chapter 3, use:")
    logger.info("    \\input{../Results/calibration/latex_tables/event_luna.tex}")
    logger.info("    \\input{../Results/calibration/latex_tables/cross_method.tex}")
    logger.info("    etc.")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_populate_tables()
