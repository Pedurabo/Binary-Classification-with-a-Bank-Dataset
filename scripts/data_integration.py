#!/usr/bin/env python3
"""
Data integration script:
- Load processed train/test
- Align schemas and concatenate with split flag (train/test)
- Optional: left-join any CSVs found under data/external/ by 'id' (if present)
- Save unified dataset for downstream feature engineering

Usage:
  python scripts/data_integration.py --out data/combined.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate processed train/test into one dataset")
    parser.add_argument("--out", type=str, default=None, help="Output path (.parquet or .csv)")
    parser.add_argument("--external-dir", type=str, default=None, help="Directory of external CSVs to join by 'id'")
    return parser.parse_args()


def load_processed(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(cfg.DATA_DIR)
    train_p = data_dir / "train_processed.csv"
    test_p = data_dir / "test_processed.csv"
    if not train_p.exists() or not test_p.exists():
        raise FileNotFoundError("Processed files not found. Run data_preprocessing first.")
    train = pd.read_csv(train_p)
    test = pd.read_csv(test_p)
    return train, test


def align_and_concat(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # Ensure both have same feature columns (y only in train)
    feature_cols = sorted(list(set(train.columns) | set(test.columns) - {"y"}))
    train_feat = train.reindex(columns=feature_cols).copy()
    test_feat = test.reindex(columns=feature_cols).copy()
    train_feat["y"] = train["y"].values if "y" in train.columns else None
    train_feat["split"] = "train"
    test_feat["y"] = None
    test_feat["split"] = "test"
    combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
    return combined


def join_external(combined: pd.DataFrame, external_dir: Path) -> pd.DataFrame:
    if not external_dir.exists():
        return combined
    csvs = list(external_dir.glob("*.csv"))
    if not csvs:
        return combined
    logger = setup_logging(__name__)
    if "id" not in combined.columns:
        logger.warning("'id' column not found; skipping external joins.")
        return combined
    for csv_path in csvs:
        try:
            ext = pd.read_csv(csv_path)
            if "id" not in ext.columns:
                logger.warning("External file %s has no 'id'; skipping", csv_path)
                continue
            before_cols = set(combined.columns)
            combined = combined.merge(ext, on="id", how="left")
            new_cols = sorted(list(set(combined.columns) - before_cols))
            logger.info("Joined %s; added columns: %s", csv_path.name, new_cols[:10])
        except Exception as exc:
            logger.warning("Failed to join %s: %s", csv_path, exc)
    return combined


def main() -> None:
    logger = setup_logging(__name__)
    cfg = Config()
    args = parse_args()

    train, test = load_processed(cfg)
    combined = align_and_concat(train, test)

    if args.external_dir:
        combined = join_external(combined, Path(args.external_dir))

    out_path = Path(args.out) if args.out else Path(cfg.DATA_DIR) / "combined.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".csv":
        combined.to_csv(out_path, index=False)
    else:
        combined.to_parquet(out_path, index=False)

    logger.info("Integrated dataset written to %s (rows=%d, cols=%d)", out_path, len(combined), combined.shape[1])
    print(f"Integrated dataset written to {out_path} with shape {combined.shape}")


if __name__ == "__main__":
    main()


