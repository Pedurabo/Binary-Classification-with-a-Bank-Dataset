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

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging


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
    if args.cap_pct is not None:
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
        x_train_fold, x_valid_fold = X.iloc[trn_idx], X.iloc[val_idx]
        y_train_fold, y_valid_fold = y.iloc[trn_idx], y.iloc[val_idx]

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
        test_pred = model.predict_proba(X_test)[:, 1]
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


