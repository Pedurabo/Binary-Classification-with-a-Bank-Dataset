#!/usr/bin/env python3
"""
End-to-end data pipeline runner:
- Download Kaggle data
- Run data validation
- Run preprocessing

Usage:
  python scripts/run_data_pipeline.py --competition playground-series-s5e8 --force-download
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data pipeline: download → validate → preprocess")
    parser.add_argument("--competition", type=str, default="playground-series-s5e8")
    parser.add_argument("--force-download", action="store_true", help="Re-download data even if files exist")
    return parser.parse_args()


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def run_cmd(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def main() -> None:
    logger = setup_logging(__name__)
    args = parse_args()

    cfg = Config()
    data_dir = Path(cfg.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    sample_csv = data_dir / "sample_submission.csv"

    # 1) Download
    needs_download = (
        args.force_download
        or not file_exists(train_csv)
        or not file_exists(test_csv)
        or not file_exists(sample_csv)
    )
    if needs_download:
        logger.info("Downloading Kaggle data...")
        code = run_cmd([
            sys.executable,
            str(Path(__file__).parent / "kaggle_download.py"),
            "--competition",
            args.competition,
            "--out",
            str(data_dir),
        ])
        if code != 0:
            raise SystemExit(code)
    else:
        logger.info("Kaggle data already present. Skipping download.")

    # 2) Validation
    logger.info("Running data validation...")
    code = run_cmd([sys.executable, str(Path(__file__).parent / "data_validation.py")])
    if code != 0:
        raise SystemExit(code)

    # 3) Preprocessing
    logger.info("Running data preprocessing...")
    code = run_cmd([sys.executable, str(Path(__file__).parent / "data_preprocessing.py")])
    if code != 0:
        raise SystemExit(code)

    logger.info("Data pipeline completed successfully.")
    print("Data pipeline completed successfully.")


if __name__ == "__main__":
    main()


