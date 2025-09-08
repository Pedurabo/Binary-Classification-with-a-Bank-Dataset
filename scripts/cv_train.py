#!/usr/bin/env python3
"""
Cross-validated training with optional GPU acceleration (XGBoost).

Outputs:
- data/oof.csv                         (OOF probabilities with fold indices)
- data/test_pred_mean.csv              (mean test probabilities across folds)
- reports/cv_metrics.csv               (per-fold and overall ROC AUC)
- models/xgb_fold_<k>.json             (saved XGBoost models per fold)

Usage:
  python scripts/cv_train.py --folds 5 --seed 42 --n-jobs -1
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import mutual_info_classif

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging


def correlation_prune_columns(df: pd.DataFrame, threshold: float) -> List[str]:
    """Return list of columns to keep after pruning highly correlated features.

    Works on numeric DataFrame `df`. Builds absolute Pearson correlation matrix
    and drops one of each pair with |corr| >= threshold (keeps earlier columns).
    """
    if df.shape[1] <= 1:
        return df.columns.tolist()

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        high_corr = upper[col][upper[col] >= threshold].index.tolist()
        to_drop.update(high_corr)

    keep_cols = [c for c in df.columns if c not in to_drop]
    return keep_cols


def variance_prune_columns(df: pd.DataFrame, threshold: float) -> List[str]:
    """Return list of columns to keep after removing near-zero variance features."""
    if df.shape[1] == 0:
        return []
    variances = df.var(axis=0)
    keep_cols = variances[variances > threshold].index.tolist()
    return keep_cols


def select_features_mutual_information(x_num: pd.DataFrame, y: pd.Series, k_top: Optional[int]) -> List[str]:
    """Select features by mutual information (higher is better). Returns kept column names."""
    if x_num.shape[1] == 0:
        return []
    if k_top is None or k_top >= x_num.shape[1]:
        return x_num.columns.tolist()
    scores = mutual_info_classif(x_num.fillna(x_num.median()), y, random_state=0)
    order = np.argsort(scores)[::-1]
    keep_idx = order[:k_top]
    return [x_num.columns[i] for i in keep_idx]


def select_features_model_importance(
    x_num: pd.DataFrame,
    y: pd.Series,
    k_top: Optional[int],
    importance_quantile: Optional[float],
    seed: int,
    n_jobs: int,
    use_gpu: bool,
) -> List[str]:
    """Select features via quick XGBoost importance on the train fold."""
    from xgboost import XGBClassifier  # local import

    if x_num.shape[1] == 0:
        return []

    quick_params = build_xgb_params(n_jobs=n_jobs, seed=seed, use_gpu=use_gpu)
    quick_params = {**quick_params, "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}
    model = XGBClassifier(**quick_params)
    model.fit(x_num, y, verbose=False)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return x_num.columns.tolist()

    order = np.argsort(importances)[::-1]
    cols = x_num.columns.to_list()

    if k_top is not None and k_top > 0:
        keep = [cols[i] for i in order[:min(k_top, len(cols))]]
        return keep

    if importance_quantile is not None and 0.0 <= importance_quantile <= 1.0:
        thr = np.quantile(importances, importance_quantile)
        keep = [c for c, w in zip(cols, importances) if w >= thr]
        return keep if keep else cols

    return cols

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV training with XGBoost (GPU if available)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--log1p-balance", action="store_true", help="Apply log1p to balance feature")
    parser.add_argument(
        "--cap-pct",
        type=float,
        default=None,
        help="Cap duration and campaign at this percentile (e.g., 0.99)",
    )
    parser.add_argument(
        "--cap-within-fold",
        action="store_true",
        help="Compute cap thresholds on train fold only (prevents leakage)",
    )
    parser.add_argument(
        "--use-combined",
        type=str,
        default=None,
        help="Path to combined or combined_fe parquet/csv with a 'split' column",
    )
    parser.add_argument(
        "--auto-drop-leakers",
        action="store_true",
        help="Auto-drop features with near-perfect single-feature AUC (>=0.999 or <=0.001)",
    )
    parser.add_argument(
        "--reduce",
        type=str,
        default=None,
        choices=[None, "corr", "var"],
        help="Optional feature reduction: 'corr' for correlation pruning, 'var' for variance threshold",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.98,
        help="Absolute correlation threshold for pruning when --reduce=corr",
    )
    parser.add_argument(
        "--var-threshold",
        type=float,
        default=0.0,
        help="Variance threshold for pruning when --reduce=var",
    )
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        choices=[None, "mi", "model"],
        help="Optional feature selection: 'mi' for mutual information, 'model' for model-based importance",
    )
    parser.add_argument(
        "--k-top",
        type=int,
        default=None,
        help="Keep top-K features after selection (applies to --select)",
    )
    parser.add_argument(
        "--importance-quantile",
        type=float,
        default=None,
        help="For --select=model: keep features with importance >= given quantile (0-1). Ignored if --k-top is set.",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="Optional path to a newline-delimited list of feature names to keep (applied within each fold)",
    )
    return parser.parse_args()


def gpu_params_available() -> dict:
    # Prefer GPU; if it fails at fit time, we'll catch and retry on CPU.
    return {
        "tree_method": "hist",
        "device": "cuda",
    }


def cpu_params() -> dict:
    return {
        "tree_method": "hist",
    }


def build_xgb_params(n_jobs: int, seed: int, use_gpu: bool) -> dict:
    base = {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": n_jobs,
        "eval_metric": "auc",
    }
    base.update(gpu_params_available() if use_gpu else cpu_params())
    return base


def try_import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception as exc:
        raise RuntimeError("xgboost is required. Please `pip install xgboost`.") from exc


def main() -> None:
    logger = setup_logging(__name__)
    args = parse_args()

    cfg = Config()
    data_dir = Path(cfg.DATA_DIR)
    reports_dir = Path(cfg.REPORTS_DIR)
    models_dir = Path(cfg.MODELS_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.use_combined:
        comb_path = Path(args.use_combined)
        if not comb_path.exists():
            raise FileNotFoundError(f"Combined dataset not found at {comb_path}")
        if comb_path.suffix.lower() == ".parquet":
            combined = pd.read_parquet(comb_path)
        else:
            combined = pd.read_csv(comb_path)
        if "split" not in combined.columns:
            raise ValueError("Combined dataset must contain a 'split' column with values 'train'/'test'")
        train_df = combined[combined["split"] == "train"].drop(columns=["split"]).copy()
        test_df = combined[combined["split"] == "test"].drop(columns=["split"]).copy()
        if "y" not in train_df.columns:
            raise ValueError("Combined train rows must include 'y' column")
        # Align columns
        feature_cols = sorted(list(set(train_df.columns) | set(test_df.columns) - {"y"}))
        X = train_df.reindex(columns=feature_cols).copy()
        y = train_df["y"].astype(int)
        X_test = test_df.reindex(columns=feature_cols).copy()
    else:
        train_path = data_dir / "train_processed.csv"
        test_path = data_dir / "test_processed.csv"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Processed data not found. Expected {train_path} and {test_path}. Run data_preprocessing first."
            )

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        if "y" not in train_df.columns:
            raise ValueError("Column 'y' not found in processed training data.")

        X = train_df.drop(columns=["y"])
        y = train_df["y"].astype(int)
        # Drop non-informative identifiers if present
        if "id" in X.columns:
            X = X.drop(columns=["id"]) 
        if "id" in test_df.columns:
            X_test = test_df.drop(columns=["id"]).copy()
        else:
            X_test = test_df.copy()
    
    # Optional feature engineering (sensitivity tests)
    if args.cap_pct is not None and not args.cap_within_fold:
        for col in ["duration", "campaign"]:
            if col in X.columns and col in X_test.columns:
                cap_val = X[col].quantile(args.cap_pct)
                X[col] = np.clip(X[col], None, cap_val)
                X_test[col] = np.clip(X_test[col], None, cap_val)
    if args.log1p_balance and "balance" in X.columns and "balance" in X_test.columns:
        X["balance"] = np.log1p(X["balance"]) 
        X_test["balance"] = np.log1p(X_test["balance"]) 

    # Handle class imbalance for XGBoost
    num_pos = int((y == 1).sum())
    num_neg = int((y == 0).sum())
    scale_pos_weight = (num_neg / num_pos) if num_pos > 0 else 1.0

    xgb_classifier_cls = try_import_xgb()

    # Try GPU first; if it errors on first fit, fall back to CPU for all folds.
    use_gpu = True
    params = build_xgb_params(args.n_jobs, args.seed, use_gpu=True)
    params["scale_pos_weight"] = scale_pos_weight

    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(X), dtype=float)
    test_preds = []
    fold_metrics = []

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        x_train_fold, x_valid_fold = X.iloc[trn_idx].copy(), X.iloc[val_idx].copy()
        y_train_fold, y_valid_fold = y.iloc[trn_idx], y.iloc[val_idx]
        x_test_fold = X_test.copy()

        # Within-fold transforms to avoid leakage
        if args.cap_pct is not None and args.cap_within_fold:
            for col in ["duration", "campaign"]:
                if col in x_train_fold.columns:
                    cap_val = x_train_fold[col].quantile(args.cap_pct)
                    x_train_fold[col] = np.clip(x_train_fold[col], None, cap_val)
                    if col in x_valid_fold.columns:
                        x_valid_fold[col] = np.clip(x_valid_fold[col], None, cap_val)
                    if col in x_test_fold.columns:
                        x_test_fold[col] = np.clip(x_test_fold[col], None, cap_val)

        # Auto-drop leaker features evaluated on train fold
        if args.auto_drop_leakers:
            from sklearn.metrics import roc_auc_score  # local import
            leaker_cols: list[str] = []
            for col in x_train_fold.columns:
                try:
                    s = x_train_fold[col]
                    # Skip non-numeric columns just in case
                    if not np.issubdtype(s.dtype, np.number):
                        continue
                    # Single-feature AUC
                    auc = roc_auc_score(y_train_fold, s.fillna(s.median()))
                    if auc >= 0.999 or auc <= 0.001:
                        leaker_cols.append(col)
                except Exception:
                    continue
            if leaker_cols:
                # Drop from all splits consistently
                x_train_fold = x_train_fold.drop(columns=leaker_cols)
                x_valid_fold = x_valid_fold.drop(columns=[c for c in leaker_cols if c in x_valid_fold.columns])
                x_test_fold = x_test_fold.drop(columns=[c for c in leaker_cols if c in x_test_fold.columns])
                logger.info("Dropped %d suspected leaker columns", len(leaker_cols))

        # Feature reduction within fold (fit on train, apply to valid/test)
        if args.reduce is not None:
            numeric_train = x_train_fold.select_dtypes(include=[np.number])
            non_numeric_cols = [c for c in x_train_fold.columns if c not in numeric_train.columns]

            if args.reduce == "corr":
                keep_numeric = correlation_prune_columns(numeric_train, threshold=float(args.corr_threshold))
            elif args.reduce == "var":
                keep_numeric = variance_prune_columns(numeric_train, threshold=float(args.var_threshold))
            else:
                keep_numeric = numeric_train.columns.tolist()

            cols_fold = keep_numeric + [c for c in non_numeric_cols if c in x_valid_fold.columns and c in x_test_fold.columns]
            x_train_fold = x_train_fold.reindex(columns=cols_fold)
            x_valid_fold = x_valid_fold.reindex(columns=cols_fold)
            x_test_fold = x_test_fold.reindex(columns=cols_fold)

            # Save a summary of kept columns for this fold
            try:
                kept_summary_path = models_dir / f"reduction_fold_{fold}.txt"
                with kept_summary_path.open("w", encoding="utf-8") as f:
                    f.write(f"Method: {args.reduce}\n")
                    if args.reduce == "corr":
                        f.write(f"corr_threshold: {args.corr_threshold}\n")
                    if args.reduce == "var":
                        f.write(f"var_threshold: {args.var_threshold}\n")
                    f.write(f"num_features_kept: {len(cols_fold)}\n")
                    f.write("columns:\n")
                    for c in cols_fold:
                        f.write(f"{c}\n")
            except Exception:
                pass

        # Apply explicit feature list if provided (last step before modeling)
        if args.features_file:
            try:
                with open(args.features_file, "r", encoding="utf-8") as f:
                    keep_list = [line.strip() for line in f if line.strip()]
                keep_cols = [c for c in x_train_fold.columns if c in keep_list]
                if keep_cols:
                    x_train_fold = x_train_fold.reindex(columns=keep_cols)
                    x_valid_fold = x_valid_fold.reindex(columns=keep_cols)
                    x_test_fold = x_test_fold.reindex(columns=keep_cols)
            except Exception:
                pass

        # Feature selection within fold (fit on train, apply to valid/test)
        if args.select is not None:
            numeric_train = x_train_fold.select_dtypes(include=[np.number])
            non_numeric_cols = [c for c in x_train_fold.columns if c not in numeric_train.columns]

            if args.select == "mi":
                keep_numeric = select_features_mutual_information(numeric_train, y_train_fold, args.k_top)
            elif args.select == "model":
                keep_numeric = select_features_model_importance(
                    numeric_train,
                    y_train_fold,
                    args.k_top,
                    args.importance_quantile,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                    use_gpu=use_gpu,
                )
            else:
                keep_numeric = numeric_train.columns.tolist()

            cols_fold = keep_numeric + [c for c in non_numeric_cols if c in x_valid_fold.columns and c in x_test_fold.columns]
            x_train_fold = x_train_fold.reindex(columns=cols_fold)
            x_valid_fold = x_valid_fold.reindex(columns=cols_fold)
            x_test_fold = x_test_fold.reindex(columns=cols_fold)

            # Save selection summary
            try:
                kept_summary_path = models_dir / f"selection_fold_{fold}.txt"
                with kept_summary_path.open("w", encoding="utf-8") as f:
                    f.write(f"Method: {args.select}\n")
                    if args.k_top is not None:
                        f.write(f"k_top: {args.k_top}\n")
                    if args.importance_quantile is not None:
                        f.write(f"importance_quantile: {args.importance_quantile}\n")
                    f.write(f"num_features_kept: {len(cols_fold)}\n")
                    f.write("columns:\n")
                    for c in cols_fold:
                        f.write(f"{c}\n")
            except Exception:
                pass

        model = xgb_classifier_cls(**params)
        try:
            model.fit(
                x_train_fold,
                y_train_fold,
                eval_set=[(x_valid_fold, y_valid_fold)],
                verbose=False,
            )
        except Exception as exc:
            if use_gpu:
                logger.warning("GPU training failed on fold %s. Falling back to CPU for remaining folds.", fold)
                use_gpu = False
                params = build_xgb_params(args.n_jobs, args.seed, use_gpu=False)
                model = xgb_classifier_cls(**params)
                model.fit(
                    x_train_fold,
                    y_train_fold,
                    eval_set=[(x_valid_fold, y_valid_fold)],
                    verbose=False,
                )
            else:
                raise

        val_pred = model.predict_proba(x_valid_fold)[:, 1]
        oof[val_idx] = val_pred
        auc = roc_auc_score(y_valid_fold, val_pred)
        fold_metrics.append({"fold": fold, "roc_auc": float(auc)})
        logger.info("Fold %s ROC AUC: %.5f", fold, auc)

        # Save model
        model_path = models_dir / f"xgb_fold_{fold}.json"
        model.save_model(str(model_path))

        # Test predictions for this fold
        test_pred = model.predict_proba(x_test_fold)[:, 1]
        test_preds.append(test_pred)

    # OOF and overall metric
    overall_auc = roc_auc_score(y, oof)
    fold_metrics.append({"fold": 0, "roc_auc": float(overall_auc)})
    logger.info("Overall OOF ROC AUC: %.5f", overall_auc)

    # Save metrics
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(reports_dir / "cv_metrics.csv", index=False)

    # Save OOF
    oof_df = pd.DataFrame({"oof": oof, "fold": 0})
    oof_df.to_csv(data_dir / "oof.csv", index=False)

    # Save mean test predictions
    test_mean = np.mean(np.vstack(test_preds), axis=0)
    pd.DataFrame({"y": test_mean}).to_csv(data_dir / "test_pred_mean.csv", index=False)

    print("CV training completed. Overall OOF ROC AUC:", overall_auc)


if __name__ == "__main__":
    main()


