"""
compare_h_bounds.py
-------------------
Compares calibration results across three H lower bounds (0.01, 0.001, 1e-4)
for each simulation method, date by date.

Usage:
    python compare_h_bounds.py [--method hybrid_euler|hybrid_mixed|cholesky_euler]
    python compare_h_bounds.py --method hybrid_euler --save
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np

TABLES_DIR = os.path.join("Results", "calibration", "tables")

BOUND_SUFFIXES = {
    "0.01":  "",
    "0.001": "_h001",
    "1e-4":  "_hfree",
}

METHODS = ["hybrid_euler", "hybrid_mixed", "cholesky_euler"]

COLS_PARAMS  = ["H", "eta", "rho", "xi"]
COLS_FIT     = ["rmse_pp", "mae_pp", "quality_flag"]
COLS_EXTRA   = ["bias_pp"]  # only in h001 / hfree


def load_method(method: str, suffix: str) -> pd.DataFrame:
    """Load all CSV files for method+suffix into a single DataFrame."""
    folder = os.path.join(TABLES_DIR, method + suffix)
    if not os.path.isdir(folder):
        return pd.DataFrame()
    pattern = os.path.join(folder, f"calibration_*{method}{suffix}.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_comparison(method: str) -> pd.DataFrame:
    """Build a merged date-by-date comparison DataFrame for a single method."""
    frames = {}
    for bound_label, suffix in BOUND_SUFFIXES.items():
        df = load_method(method, suffix)
        if df.empty:
            print(f"  [WARN] No data found for {method}{suffix}")
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        frames[bound_label] = df

    if not frames:
        return pd.DataFrame()

    # Align on common dates
    dates = sorted(set.intersection(*[set(df["date"]) for df in frames.values()]))
    rows = []
    for date in dates:
        row = {"date": date}
        for bound_label, df in frames.items():
            sub = df[df["date"] == date]
            if sub.empty:
                continue
            r = sub.iloc[0]
            b = bound_label
            row[f"H_{b}"]           = r["H"]
            row[f"eta_{b}"]         = r["eta"]
            row[f"rho_{b}"]         = r["rho"]
            row[f"rmse_{b}"]        = r["rmse_pp"]
            row[f"mae_{b}"]         = r["mae_pp"]
            row[f"quality_{b}"]     = r["quality_flag"]
            if "bias_pp" in r.index:
                row[f"bias_{b}"]    = r["bias_pp"]
            row[f"label_{b}"]       = r.get("label", "")
        rows.append(row)

    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame, method: str):
    bounds = list(BOUND_SUFFIXES.keys())
    print(f"\n{'='*100}")
    print(f"  METHOD: {method.upper()}  —  {len(df)} dates")
    print(f"{'='*100}")

    # Header
    header = f"{'Date':<12}"
    for b in bounds:
        header += f"  {'H_' + b:<8} {'eta_' + b:<7} {'rho_' + b:<7} {'rmse_' + b:<9} {'q_' + b:<12}"
    print(header)
    print("-" * 100)

    for _, row in df.iterrows():
        line = f"{str(row['date']):<12}"
        for b in bounds:
            H    = row.get(f"H_{b}",    float("nan"))
            eta  = row.get(f"eta_{b}",  float("nan"))
            rho  = row.get(f"rho_{b}",  float("nan"))
            rmse = row.get(f"rmse_{b}", float("nan"))
            qual = row.get(f"quality_{b}", "")
            line += f"  {H:<8.4f} {eta:<7.3f} {rho:<7.3f} {rmse:<9.2f} {qual:<12}"
        print(line)

    print()
    # Summary: where does H differ across bounds?
    print("  --- Delta H analysis (H_0.001 - H_0.01  |  H_1e-4 - H_0.01) ---")
    header2 = f"{'Date':<12}  {'dH(0.001-0.01)':>16}  {'dH(1e-4-0.01)':>14}  {'dRMSE(0.001-0.01)':>20}  {'dRMSE(1e-4-0.01)':>18}"
    print(header2)
    print("-" * 85)
    for _, row in df.iterrows():
        H_def  = row.get("H_0.01",  float("nan"))
        H_h001 = row.get("H_0.001", float("nan"))
        H_hfr  = row.get("H_1e-4",  float("nan"))
        r_def  = row.get("rmse_0.01",  float("nan"))
        r_h001 = row.get("rmse_0.001", float("nan"))
        r_hfr  = row.get("rmse_1e-4",  float("nan"))

        dH1   = H_h001 - H_def   if not np.isnan(H_h001)  else float("nan")
        dH2   = H_hfr  - H_def   if not np.isnan(H_hfr)   else float("nan")
        dR1   = r_h001 - r_def   if not np.isnan(r_h001)  else float("nan")
        dR2   = r_hfr  - r_def   if not np.isnan(r_hfr)   else float("nan")

        # Flag if H hit the lower bound (within 10% of the bound)
        flag = ""
        if not np.isnan(H_def)  and H_def  < 0.015: flag += "[H@lb:0.01] "
        if not np.isnan(H_h001) and H_h001 < 0.0015: flag += "[H@lb:0.001] "
        if not np.isnan(H_hfr)  and H_hfr  < 0.0002: flag += "[H@lb:1e-4] "

        print(f"{str(row['date']):<12}  {dH1:>+16.4f}  {dH2:>+14.4f}  {dR1:>+20.2f}  {dR2:>+18.2f}  {flag}")


def save_comparison(df: pd.DataFrame, method: str):
    out_dir = os.path.join("Results", "calibration", "h_bound_comparison")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"h_bound_comparison_{method}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare calibration results across H lower bounds.")
    parser.add_argument("--method", type=str, default=None,
                        choices=METHODS, help="Method to compare (default: all)")
    parser.add_argument("--save", action="store_true", help="Save comparison CSV")
    args = parser.parse_args()

    methods = [args.method] if args.method else METHODS

    for method in methods:
        df = build_comparison(method)
        if df.empty:
            print(f"[SKIP] {method}: no data found")
            continue
        print_comparison(df, method)
        if args.save:
            save_comparison(df, method)


if __name__ == "__main__":
    main()
