#!/usr/bin/env python3
"""
LightGBM GPU cross-validated training with optional combined dataset input.

Outputs:
- data/oof_lgbm.csv
- data/test_pred_mean_lgbm.csv
- reports/cv_metrics_lgbm.csv
- models/lgbm_fold_<k>.txt (model dump)

Usage:
  python scripts/cv_train_lgbm.py --folds 5 --seed 42 --n-jobs -1 [--use-combined data/combined.parquet]
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
    parser = argparse.ArgumentParser(description="CV training with LightGBM (GPU if available)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--use-combined", type=str, default=None)
    return parser.parse_args()


def try_import_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except Exception as exc:
        raise RuntimeError("lightgbm is required. Please `pip install lightgbm`." ) from exc


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

    # Load data
    if args.use_combined:
        comb_path = Path(args.use_combined)
        if not comb_path.exists():
            raise FileNotFoundError(f"Combined dataset not found at {comb_path}")
        combined = pd.read_parquet(comb_path) if comb_path.suffix.lower() == ".parquet" else pd.read_csv(comb_path)
        if "split" not in combined.columns:
            raise ValueError("Combined dataset must contain a 'split' column")
        train_df = combined[combined["split"] == "train"].drop(columns=["split"]).copy()
        test_df = combined[combined["split"] == "test"].drop(columns=["split"]).copy()
    else:
        train_df = pd.read_csv(data_dir / "train_processed.csv")
        test_df = pd.read_csv(data_dir / "test_processed.csv")

    if "y" not in train_df.columns:
        raise ValueError("Column 'y' not found in training data.")

    X = train_df.drop(columns=["y"]).copy()
    y = train_df["y"].astype(int).copy()
    if "id" in X.columns:
        X = X.drop(columns=["id"]) 
    if "id" in test_df.columns:
        X_test = test_df.drop(columns=["id"]).copy()
    else:
        X_test = test_df.copy()

    # Class imbalance weight
    num_pos = int((y == 1).sum())
    num_neg = int((y == 0).sum())
    scale_pos_weight = (num_neg / num_pos) if num_pos > 0 else 1.0

    lgb = try_import_lgbm()
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": args.seed,
        "num_threads": args.n_jobs,
        "scale_pos_weight": scale_pos_weight,
        # GPU settings
        "device": "gpu",
        "gpu_platform_id": -1,
        "gpu_device_id": -1,
    }

    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(X), dtype=float)
    test_preds = []
    fold_metrics = []

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        x_tr, x_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1200,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[lgb.log_evaluation(period=0)],
        )

        val_pred = model.predict(x_val)
        oof[val_idx] = val_pred
        auc = roc_auc_score(y_val, val_pred)
        fold_metrics.append({"fold": fold, "roc_auc": float(auc)})
        logger.info("Fold %s ROC AUC: %.5f", fold, auc)

        model_path = models_dir / f"lgbm_fold_{fold}.txt"
        model.save_model(str(model_path))
        test_preds.append(model.predict(X_test))

    overall_auc = roc_auc_score(y, oof)
    fold_metrics.append({"fold": 0, "roc_auc": float(overall_auc)})
    logger.info("Overall OOF ROC AUC: %.5f", overall_auc)

    pd.DataFrame(fold_metrics).to_csv(reports_dir / "cv_metrics_lgbm.csv", index=False)
    pd.DataFrame({"oof": oof, "fold": 0}).to_csv(data_dir / "oof_lgbm.csv", index=False)
    test_mean = np.mean(np.vstack(test_preds), axis=0)
    pd.DataFrame({"y": test_mean}).to_csv(data_dir / "test_pred_mean_lgbm.csv", index=False)

    print("LightGBM CV completed. Overall OOF ROC AUC:", overall_auc)


if __name__ == "__main__":
    main()


