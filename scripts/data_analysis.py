#!/usr/bin/env python3
"""
Data analysis utility:
- Descriptive stats and missingness summary (CSV)
- Correlation matrix (CSV) and heatmap (PNG)
- Distributions for numeric features (PNG)
- Target-wise distributions for top features by MI/AUC if available (PNGs)

Outputs: reports/analysis/

Usage:
  python scripts/data_analysis.py --use-combined data/combined_fe.parquet
  python scripts/data_analysis.py  # uses data/train_processed.csv by default
"""

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data analysis and diagnostics")
    p.add_argument("--use-combined", type=str, default=None, help="Path to combined(.parquet/.csv) with split column")
    p.add_argument("--max-plots", type=int, default=30, help="Max number of feature plots")
    return p.parse_args()


def load_train_df(args: argparse.Namespace) -> pd.DataFrame:
    if args.use_combined:
        path = Path(args.use_combined)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        if "split" not in df.columns:
            raise ValueError("combined dataset must have 'split'")
        train_df = df[df["split"] == "train"].drop(columns=["split"]).copy()
        if "y" not in train_df.columns:
            raise ValueError("train rows must include 'y'")
        return train_df

    cfg = Config()
    train_path = Path(cfg.DATA_DIR) / "train_processed.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    return pd.read_csv(train_path)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_descriptives(train_df: pd.DataFrame, out_dir: Path) -> None:
    desc = train_df.describe(include="all").transpose()
    desc.to_csv(out_dir / "descriptive_stats.csv")

    miss = train_df.isnull().sum().rename("missing_count").to_frame()
    miss["missing_pct"] = miss["missing_count"] / len(train_df)
    miss.sort_values("missing_pct", ascending=False).to_csv(out_dir / "missingness.csv")


def save_correlations(train_df: pd.DataFrame, out_dir: Path) -> None:
    num = train_df.select_dtypes(include=[np.number]).copy()
    if num.shape[1] == 0:
        return
    corr = num.corr()
    corr.to_csv(out_dir / "correlation_matrix.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, square=False, cbar=True)
    plt.title("Correlation heatmap (numeric)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def save_distributions(train_df: pd.DataFrame, out_dir: Path, max_plots: int) -> None:
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "y"][:max_plots]
    if not num_cols:
        return
    fig_dir = out_dir / "distributions"
    ensure_dir(fig_dir)
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        s = train_df[col].replace([np.inf, -np.inf], np.nan)
        sns.histplot(s, kde=True, bins=50, color="#4C78A8")
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"dist_{col}.png", dpi=150)
        plt.close()


def save_target_wise(train_df: pd.DataFrame, out_dir: Path, max_plots: int) -> None:
    if "y" not in train_df.columns:
        return
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "y"]

    # Try to prioritize from existing reports if available
    reports_dir = Path("reports")
    prior_list: List[str] = []
    shap_path = reports_dir / "shap_importance.csv"
    auc_path = reports_dir / "univariate_auc.csv"
    try:
        if shap_path.exists():
            df = pd.read_csv(shap_path).sort_values("shap_mean_abs", ascending=False)
            prior_list = df["feature"].astype(str).tolist()
        elif auc_path.exists():
            df = pd.read_csv(auc_path).sort_values("auc", ascending=False)
            prior_list = df["feature"].astype(str).tolist()
    except Exception:
        prior_list = []

    ordered = [c for c in prior_list if c in num_cols] + [c for c in num_cols if c not in prior_list]
    top = ordered[:max_plots]
    if not top:
        return

    fig_dir = out_dir / "target_plots"
    ensure_dir(fig_dir)
    for col in top:
        try:
            plt.figure(figsize=(6, 4))
            sns.kdeplot(data=train_df, x=col, hue="y", common_norm=False)
            plt.title(f"Target-wise KDE: {col}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"kde_{col}.png", dpi=150)
            plt.close()
        except Exception:
            continue


def main() -> None:
    args = parse_args()
    out_dir = Path("reports") / "analysis"
    ensure_dir(out_dir)

    train_df = load_train_df(args)

    save_descriptives(train_df, out_dir)
    save_correlations(train_df, out_dir)
    save_distributions(train_df, out_dir, max_plots=int(args.max_plots))
    save_target_wise(train_df, out_dir, max_plots=int(args.max_plots))

    print(f"Data analysis outputs written to: {out_dir}")


if __name__ == "__main__":
    main()


