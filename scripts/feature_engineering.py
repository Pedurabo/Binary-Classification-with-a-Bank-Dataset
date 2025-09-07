#!/usr/bin/env python3
"""
Feature engineering on integrated dataset (combined.parquet):
- Optional log1p(balance)
- Optional capping for duration/campaign
- Rare-category grouping for low-cardinality categorical-like columns
- Quantile/binning for heavy-tailed numerics
- Interactions: duration_per_campaign, duration_x_contact, balance_over_age

Usage:
  python scripts/feature_engineering.py --in data/combined.parquet --out data/combined_fe.parquet --log1p-balance --cap-pct 0.99
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering for combined dataset")
    parser.add_argument("--in", dest="in_path", type=str, default="data/combined.parquet")
    parser.add_argument("--out", dest="out_path", type=str, default="data/combined_fe.parquet")
    parser.add_argument("--log1p-balance", action="store_true")
    parser.add_argument("--cap-pct", type=float, default=None)
    parser.add_argument("--rare-threshold", type=float, default=None, help="Group categories appearing <= threshold fraction into a rare code")
    parser.add_argument("--rare-max-unique", type=int, default=50, help="Treat columns with <= this many uniques as categorical-like for rare grouping")
    parser.add_argument("--rare-code", type=float, default=-1.0, help="Numeric code to assign to rare categories")
    parser.add_argument("--quantile-bins", type=int, default=None, help="Number of quantile bins to create (qcut) for specified columns")
    parser.add_argument("--bin-cols", type=str, nargs='*', default=["balance", "duration", "campaign"], help="Columns to quantile-bin when --quantile-bins is set")
    parser.add_argument("--add-interactions", action="store_true", help="Add pairwise interaction features")
    return parser.parse_args()


def main() -> None:
    logger = setup_logging(__name__)
    args = parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = pd.read_parquet(in_path) if in_path.suffix.lower() == ".parquet" else pd.read_csv(in_path)

    # Capping
    if isinstance(args.cap_pct, float) and 0.0 < args.cap_pct < 1.0:
        for col in ["duration", "campaign"]:
            if col in df.columns:
                cap_val = df[col].quantile(args.cap_pct)
                df[col] = np.clip(df[col], None, cap_val)

    # Log1p on balance
    if args.log1p_balance and "balance" in df.columns:
        df["balance"] = np.log1p(df["balance"]) 

    # Rare-category grouping for low-cardinality numeric-coded categoricals
    if isinstance(args.rare_threshold, float) and 0.0 < args.rare_threshold < 1.0:
        n = len(df)
        for col in df.columns:
            if col in {"y", "split"}:
                continue
            # consider columns that look categorical (few unique values)
            uniques = df[col].nunique(dropna=True)
            if uniques <= args.rare_max_unique:
                freq = df[col].value_counts(dropna=True) / n
                rare_vals = set(freq[freq <= args.rare_threshold].index.tolist())
                if rare_vals:
                    df[col] = df[col].apply(lambda v: args.rare_code if v in rare_vals else v)

    # Quantile/binning for heavy-tailed numerics (adds new columns with _qbin suffix)
    if isinstance(args.quantile_bins, int) and args.quantile_bins > 1:
        for col in args.bin_cols:
            if col in df.columns:
                try:
                    df[f"{col}_qbin"] = pd.qcut(df[col], q=args.quantile_bins, duplicates='drop', labels=False)
                except Exception:
                    # skip if constant or invalid for qcut
                    pass

    # Interactions
    # duration_per_campaign
    if "duration" in df.columns and "campaign" in df.columns:
        denom = df["campaign"].replace(0, 1)
        df["duration_per_campaign"] = df["duration"] / denom
    # duration_x_contact (if contact exists)
    if "duration" in df.columns and "contact" in df.columns and args.add_interactions:
        df["duration_x_contact"] = df["duration"] * df["contact"]
    # balance_over_age
    if "balance" in df.columns and "age" in df.columns and args.add_interactions:
        denom = df["age"].replace(0, np.nan)
        df["balance_over_age"] = df["balance"] / denom

    # Write
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    logger.info("Feature engineered dataset written to %s (rows=%d, cols=%d)", out_path, len(df), df.shape[1])
    print(f"Feature engineered dataset written to {out_path} with shape {df.shape}")


if __name__ == "__main__":
    main()


