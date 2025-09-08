#!/usr/bin/env python3
"""
Generate charts from mining and CV reports into reports/figures/.

Inputs (if present):
  reports/univariate_auc.csv
  reports/mutual_info.csv
  reports/target_correlations.csv
  reports/model_importance.csv
  reports/shap_importance.csv

Outputs:
  reports/figures/
    top_univariate_auc.png
    top_mutual_info.png
    target_correlation_top.png
    model_importance_top.png
    shap_importance_top.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_barplot(df: pd.DataFrame, feature_col: str, value_col: str, title: str, out_path: Path, top_n: int = 30) -> None:
    if df is None or df.empty:
        return
    d = df.head(top_n).copy()
    plt.figure(figsize=(10, max(4, 0.3 * len(d))))
    sns.barplot(data=d, y=feature_col, x=value_col, orient="h", palette="Blues_r")
    plt.title(title)
    plt.xlabel(value_col)
    plt.ylabel("")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize mining reports")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--fig-dir", type=str, default="reports/figures")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    rep = Path(args.reports_dir)
    fig = Path(args.fig_dir)
    n = args.top_n

    # Univariate AUC
    p = rep / "univariate_auc.csv"
    if p.exists():
        df = pd.read_csv(p).sort_values("auc", ascending=False)
        save_barplot(df, "feature", "auc", f"Top {n} Univariate AUC", fig / "top_univariate_auc.png", n)

    # MI
    p = rep / "mutual_info.csv"
    if p.exists():
        df = pd.read_csv(p).sort_values("mi", ascending=False)
        save_barplot(df, "feature", "mi", f"Top {n} Mutual Information", fig / "top_mutual_info.png", n)

    # Target correlations (abs)
    p = rep / "target_correlations.csv"
    if p.exists():
        df = pd.read_csv(p).sort_values("corr_abs", ascending=False)
        save_barplot(df, "feature", "corr_abs", f"Top {n} |Correlation| with Target", fig / "target_correlation_top.png", n)

    # Model importance
    p = rep / "model_importance.csv"
    if p.exists():
        df = pd.read_csv(p).sort_values("importance", ascending=False)
        save_barplot(df, "feature", "importance", f"Top {n} Model Importance", fig / "model_importance_top.png", n)

    # SHAP importance
    p = rep / "shap_importance.csv"
    if p.exists():
        df = pd.read_csv(p).sort_values("shap_mean_abs", ascending=False)
        save_barplot(df, "feature", "shap_mean_abs", f"Top {n} SHAP |mean|", fig / "shap_importance_top.png", n)

    print(f"Figures written to: {fig}")


if __name__ == "__main__":
    main()


