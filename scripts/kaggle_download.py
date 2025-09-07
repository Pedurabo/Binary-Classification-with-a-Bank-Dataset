#!/usr/bin/env python3
"""
Download Kaggle competition data for Playground Series S5E8.

Requirements:
- Kaggle API credentials at:
  Windows: %USERPROFILE%\.kaggle\kaggle.json (read-only)

Usage:
  python scripts/kaggle_download.py --competition playground-series-s5e8 --out data
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle competition data")
    parser.add_argument(
        "--competition",
        type=str,
        default="playground-series-s5e8",
        help="Kaggle competition slug",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for downloaded files (defaults to Config.DATA_DIR)",
    )
    return parser.parse_args()


def main() -> None:
    logger = setup_logging(__name__)
    args = parse_args()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Kaggle package not installed. Please `pip install kaggle`."
        ) from exc

    config = Config()
    out_dir = Path(args.out) if args.out else Path(config.DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Authenticating with Kaggle API ...")
    api = KaggleApi()
    api.authenticate()

    logger.info(
        f"Downloading competition files for '{args.competition}' into {out_dir} ..."
    )
    api.competition_download_files(args.competition, path=str(out_dir), quiet=False)

    # If Kaggle provides a single zip, extract it
    # Try to find the downloaded zip named as competition.zip
    zip_path = out_dir / f"{args.competition}.zip"
    if zip_path.exists():
        import zipfile

        logger.info(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        zip_path.unlink(missing_ok=True)

    expected = ["train.csv", "test.csv", "sample_submission.csv"]
    missing = [f for f in expected if not (out_dir / f).exists()]
    if missing:
        logger.warning(
            f"Download completed but missing expected files: {missing}. Contents: {list(out_dir.iterdir())}"
        )
    else:
        logger.info("All expected files present: train.csv, test.csv, sample_submission.csv")

    print(f"Data ready at: {out_dir}")


if __name__ == "__main__":
    main()


