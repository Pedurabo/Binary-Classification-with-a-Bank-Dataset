#!/usr/bin/env python3
"""
Sweep correlation pruning thresholds for cv_train.py and record best OOF AUC.

Examples:
  python scripts/sweep_corr_threshold.py --thresholds 0.995 0.99 0.985 0.98 --folds 5 --n-jobs -1
  python scripts/sweep_corr_threshold.py --use-combined data/combined_fe.parquet --thresholds 0.999 0.997 0.995 --folds 5

Outputs:
  - reports/corr_sweep_results.csv (threshold, oof_auc)
  - prints the best threshold and AUC
"""

import argparse
import subprocess
import sys
from pathlib import Path
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep correlation thresholds for feature reduction")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True, help="List of corr thresholds to test")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--use-combined", type=str, default=None, help="Path to combined(.parquet/.csv) with split column")
    parser.add_argument("--extra-args", type=str, nargs=argparse.REMAINDER, help="Extra args passed to cv_train.py after '--'")
    return parser.parse_args()


def run_once(threshold: float, folds: int, seed: int, n_jobs: int, use_combined: str | None, extra_args: list[str] | None) -> float:
    cmd = [sys.executable, "scripts/cv_train.py", "--folds", str(folds), "--seed", str(seed), "--n-jobs", str(n_jobs), "--reduce", "corr", "--corr-threshold", str(threshold), "--auto-drop-leakers"]
    if use_combined:
        cmd.extend(["--use-combined", use_combined])
    if extra_args:
        # Pass through any additional args after '--'
        cmd.extend(extra_args)

    # Capture stdout for parsing final AUC line
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    text = stdout + "\n" + stderr

    # Look for the printed overall line
    auc = None
    needle = "CV training completed. Overall OOF ROC AUC:".lower()
    for line in text.splitlines()[::-1]:
        if needle in line.lower():
            # The number may be at end of line; split and take last float
            parts = line.strip().split()
            for token in reversed(parts):
                try:
                    auc = float(token)
                    break
                except Exception:
                    continue
            break

    if auc is None:
        # Fallback: try reading reports/cv_metrics.csv last row
        try:
            rep_path = Path("reports") / "cv_metrics.csv"
            if rep_path.exists():
                import pandas as pd  # type: ignore
                df = pd.read_csv(rep_path)
                # overall row has fold == 0
                if "fold" in df.columns and "roc_auc" in df.columns:
                    overall = df[df["fold"] == 0]["roc_auc"]
                    if len(overall) > 0:
                        auc = float(overall.iloc[-1])
        except Exception:
            pass

    if auc is None:
        raise RuntimeError(f"Failed to parse AUC for threshold {threshold}.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")

    return auc


def main() -> None:
    args = parse_args()
    results: list[tuple[float, float]] = []

    for thr in args.thresholds:
        auc = run_once(
            threshold=float(thr),
            folds=int(args.folds),
            seed=int(args.seed),
            n_jobs=int(args.n_jobs),
            use_combined=args.use_combined,
            extra_args=args.extra_args,
        )
        print(f"threshold={thr} -> OOF AUC={auc:.6f}")
        results.append((float(thr), float(auc)))

    # Write results CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "corr_sweep_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["corr_threshold", "oof_auc"])
        writer.writerows(results)

    # Best
    best_thr, best_auc = max(results, key=lambda t: t[1])
    print(f"Best: threshold={best_thr} OOF AUC={best_auc:.6f}")


if __name__ == "__main__":
    main()


