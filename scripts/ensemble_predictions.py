#!/usr/bin/env python3
"""
Simple ensembling of multiple test prediction CSVs by arithmetic mean.

Usage:
  python scripts/ensemble_predictions.py --inputs data/test_pred_mean.csv data/test_pred_mean_lgbm.csv data/test_pred_mean_cat.csv --out data/test_pred_mean_ens.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average multiple test prediction CSVs (column 'y')")
    parser.add_argument("--inputs", nargs="+", type=str, required=True, help="List of CSVs with a 'y' column")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        if "y" not in df.columns:
            if df.shape[1] == 1:
                df.columns = ["y"]
            else:
                raise ValueError(f"File {p} missing 'y' column")
        frames.append(df[["y"]].rename(columns={"y": f"y_{len(frames)}"}))

    ens = pd.concat(frames, axis=1)
    ens["y"] = ens.mean(axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ens[["y"]].to_csv(args.out, index=False)
    print(f"Ensembled predictions written to {args.out}")


if __name__ == "__main__":
    main()


