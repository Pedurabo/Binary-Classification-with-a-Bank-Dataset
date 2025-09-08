#!/usr/bin/env python3
"""
Export top-K feature names from reports into a features file for training.

Example:
  python scripts/export_top_features.py --source shap --k 60 --out reports/features_top60.txt
  python scripts/export_top_features.py --source model --k 80 --out reports/features_top80.txt
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export top-K features from reports")
    p.add_argument("--source", type=str, choices=["shap", "model", "auc", "mi", "corr"], required=True)
    p.add_argument("--k", type=int, default=60)
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rep = Path(args.reports_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    mapping = {
        "shap": (rep / "shap_importance.csv", "feature", "shap_mean_abs", True),
        "model": (rep / "model_importance.csv", "feature", "importance", True),
        "auc": (rep / "univariate_auc.csv", "feature", "auc", True),
        "mi": (rep / "mutual_info.csv", "feature", "mi", True),
        "corr": (rep / "target_correlations.csv", "feature", "corr_abs", True),
    }

    path, feat_col, score_col, desc = mapping[args.source]
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if score_col in df.columns:
        df = df.sort_values(score_col, ascending=not desc)
    features = df[feat_col].dropna().astype(str).head(args.k).tolist()

    with out.open("w", encoding="utf-8") as f:
        for name in features:
            f.write(f"{name}\n")

    print(f"Wrote {len(features)} features to {out}")


if __name__ == "__main__":
    main()


