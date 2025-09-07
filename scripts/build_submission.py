#!/usr/bin/env python3
"""
Build Kaggle submission from test predictions and sample submission.

Usage:
  python scripts/build_submission.py --test-preds data/test_pred_mean.csv --sample data/sample_submission.csv --out submissions
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kaggle submission CSV")
    parser.add_argument("--test-preds", type=str, default=None, help="Path to test predictions CSV (with column 'y')")
    parser.add_argument("--sample", type=str, default=None, help="Path to sample_submission.csv")
    parser.add_argument("--out", type=str, default=None, help="Output directory for submission CSVs")
    return parser.parse_args()


def main() -> None:
    logger = setup_logging(__name__)
    cfg = Config()
    args = parse_args()

    preds_path = Path(args.test_preds) if args.test_preds else Path(cfg.DATA_DIR) / "test_pred_mean.csv"
    sample_path = Path(args.sample) if args.sample else Path(cfg.DATA_DIR) / "sample_submission.csv"
    out_dir = Path(args.out) if args.out else (Path(cfg.PROJECT_ROOT) / "submissions")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing test predictions at {preds_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample submission at {sample_path}")

    preds = pd.read_csv(preds_path)
    sample = pd.read_csv(sample_path)

    if "y" not in preds.columns:
        # allow a single column that is unnamed
        if preds.shape[1] == 1:
            preds.columns = ["y"]
        else:
            raise ValueError("Predictions file must have a 'y' column or a single unnamed column")

    if len(sample) != len(preds):
        raise ValueError(f"Row mismatch: sample has {len(sample)}, preds has {len(preds)}")

    submission = sample.copy()
    submission["y"] = preds["y"].clip(0.0, 1.0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"submission_{ts}.csv"
    submission.to_csv(out_path, index=False)
    logger.info(f"Submission written to {out_path}")
    print(f"Submission written to {out_path}")


if __name__ == "__main__":
    main()


