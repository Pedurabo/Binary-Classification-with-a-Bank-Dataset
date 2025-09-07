#!/usr/bin/env python3
"""
Data mining utility to profile features:
- Univariate ROC AUC per feature (numeric only)
- Mutual information scores
- Pearson correlation with target (numeric)
- Quick XGBoost feature importance
- Optional KMeans clustering summary on selected features

Outputs written under reports/:
- reports/univariate_auc.csv
- reports/mutual_info.csv
- reports/target_correlations.csv
- reports/model_importance.csv
- reports/kmeans_summary.csv (optional)

Usage:
  python scripts/data_mining.py --use-combined data/combined_fe.parquet --kmeans 0
  python scripts/data_mining.py --train data/train_processed.csv --test data/test_processed.csv --kmeans 5 --kmeans-cols duration balance
"""

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data mining and feature profiling")
    p.add_argument("--use-combined", type=str, default=None, help="Path to combined or combined_fe with split column")
    p.add_argument("--train", type=str, default=None, help="Path to train_processed.csv if not using combined")
    p.add_argument("--test", type=str, default=None, help="Path to test_processed.csv if not using combined")
    p.add_argument("--kmeans", type=int, default=0, help="If >0, run KMeans with K clusters on selected columns")
    p.add_argument("--kmeans-cols", nargs='*', default=None, help="Columns to use for KMeans (default: all numeric)")
    p.add_argument("--shap", action="store_true", help="Compute SHAP-based feature importance using a quick XGBoost model")
    p.add_argument("--shap-n-samples", type=int, default=5000, help="Max training rows to sample for SHAP (speed)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.use_combined:
        path = Path(args.use_combined)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        if "split" not in df.columns:
            raise ValueError("combined dataset must have 'split' column")
        train_df = df[df["split"] == "train"].drop(columns=["split"]).copy()
        test_df = df[df["split"] == "test"].drop(columns=["split"]).copy()
        if "y" not in train_df.columns:
            raise ValueError("train rows must include 'y'")
        return train_df, test_df

    cfg = Config()
    train_path = Path(args.train or (Path(cfg.DATA_DIR) / "train_processed.csv"))
    test_path = Path(args.test or (Path(cfg.DATA_DIR) / "test_processed.csv"))
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "y" not in train_df.columns:
        raise ValueError("Column 'y' not in training data")
    return train_df, test_df


def compute_univariate_auc(train_df: pd.DataFrame) -> pd.DataFrame:
    y = train_df["y"].astype(int)
    auc_rows: list[dict] = []
    for col in train_df.columns:
        if col == "y":
            continue
        s = train_df[col]
        if not np.issubdtype(s.dtype, np.number):
            continue
        try:
            auc = roc_auc_score(y, s.fillna(s.median()))
            auc_rows.append({"feature": col, "auc": float(auc)})
        except Exception:
            continue
    return pd.DataFrame(auc_rows).sort_values("auc", ascending=False)


def compute_mutual_information(train_df: pd.DataFrame) -> pd.DataFrame:
    y = train_df["y"].astype(int)
    X = train_df.drop(columns=["y"]).select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "mi"])    
    scores = mutual_info_classif(X.fillna(X.median()), y, random_state=0)
    return pd.DataFrame({"feature": X.columns, "mi": scores}).sort_values("mi", ascending=False)


def compute_target_correlations(train_df: pd.DataFrame) -> pd.DataFrame:
    y = train_df["y"].astype(int)
    X = train_df.drop(columns=["y"]).select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "corr_abs", "corr"])
    corrs = []
    for col in X.columns:
        try:
            s = X[col].replace([np.inf, -np.inf], np.nan)
            s = s.fillna(s.median())
            # Guard against constant columns (zero std -> NaN correlation)
            if np.isclose(float(np.nanstd(s)), 0.0) or np.isclose(float(np.nanstd(y)), 0.0):
                c = 0.0
            else:
                with np.errstate(invalid='ignore', divide='ignore'):
                    c = float(pd.Series(s).corr(pd.Series(y), method='pearson'))
                if np.isnan(c):
                    c = 0.0
            corrs.append({"feature": col, "corr_abs": abs(c), "corr": c})
        except Exception:
            continue
    return pd.DataFrame(corrs).sort_values("corr_abs", ascending=False)


def compute_model_importance(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    from xgboost import XGBClassifier

    y = train_df["y"].astype(int)
    X = train_df.drop(columns=["y"]).select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "importance"])    
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X, y, verbose=False)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])    
    return pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)


def kmeans_summary(df: pd.DataFrame, k: int, cols: List[str] | None, seed: int) -> pd.DataFrame:
    if k <= 0:
        return pd.DataFrame()
    use_cols = cols or df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[use_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(X)
    out = pd.DataFrame({"cluster": labels})
    df_reset = df.reset_index(drop=True)
    # Concatenate side-by-side by row order to avoid ambiguous join
    out = pd.concat([out, df_reset], axis=1)
    summary = out.groupby("cluster").agg(**{**{f"mean_{c}": (c, "mean") for c in use_cols}})
    summary = summary.reset_index()
    return summary


def compute_shap_importance(train_df: pd.DataFrame, seed: int, max_samples: int) -> pd.DataFrame:
    """Train a quick XGBoost model and compute mean(|SHAP|) per feature.

    To keep it fast, subsample up to max_samples rows.
    """
    try:
        import shap  # type: ignore
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError("SHAP or xgboost is not installed. Please `pip install shap xgboost`." ) from exc

    y = train_df["y"].astype(int)
    X = train_df.drop(columns=["y"]).select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "shap_mean_abs"])    

    # Subsample for speed
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X.iloc[idx].copy()
        y = y.iloc[idx].copy()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X, y, verbose=False)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values can be (n_samples, n_features)
    if isinstance(shap_values, list):
        # For multiclass models returns list; we have binary so take last
        sv = shap_values[-1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    return pd.DataFrame({"feature": X.columns, "shap_mean_abs": mean_abs}).sort_values("shap_mean_abs", ascending=False)


def main() -> None:
    args = parse_args()
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(args)

    # Univariate AUC
    uni_auc = compute_univariate_auc(train_df)
    uni_auc.to_csv(reports_dir / "univariate_auc.csv", index=False)

    # Mutual information
    mi = compute_mutual_information(train_df)
    mi.to_csv(reports_dir / "mutual_info.csv", index=False)

    # Correlation with target
    corrs = compute_target_correlations(train_df)
    corrs.to_csv(reports_dir / "target_correlations.csv", index=False)

    # Model importance
    imp = compute_model_importance(train_df, seed=args.seed)
    imp.to_csv(reports_dir / "model_importance.csv", index=False)

    # Optional KMeans
    if args.kmeans and args.kmeans > 0:
        cols = args.kmeans_cols if args.kmeans_cols else None
        km = kmeans_summary(train_df.drop(columns=["y"]), k=args.kmeans, cols=cols, seed=args.seed)
        km.to_csv(reports_dir / "kmeans_summary.csv", index=False)

    # Optional SHAP
    if args.shap:
        shap_imp = compute_shap_importance(train_df, seed=args.seed, max_samples=int(args.shap_n_samples))
        shap_imp.to_csv(reports_dir / "shap_importance.csv", index=False)

    print("Data mining reports written to:")
    for f in ["univariate_auc.csv", "mutual_info.csv", "target_correlations.csv", "model_importance.csv"]:
        print(reports_dir / f)
    if args.kmeans and args.kmeans > 0:
        print(reports_dir / "kmeans_summary.csv")
    if args.shap:
        print(reports_dir / "shap_importance.csv")


if __name__ == "__main__":
    main()


