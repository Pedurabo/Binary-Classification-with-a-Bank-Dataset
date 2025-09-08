#!/usr/bin/env python3
"""
Archive reports and figures into a timestamped zip for reproducibility.

Usage:
  python scripts/archive_reports.py --out-dir submissions
"""

import argparse
from datetime import datetime
from pathlib import Path
import zipfile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Archive reports and figures")
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--out-dir", type=str, default="submissions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rep = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_zip = out_dir / f"reports_{ts}.zip"

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in rep.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(rep.parent))

    print(f"Archived reports to {out_zip}")


if __name__ == "__main__":
    main()


