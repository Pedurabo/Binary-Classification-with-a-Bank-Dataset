#!/usr/bin/env python3
"""
Sweep SHAP top-K feature counts by generating a features file and running cv_train.py.

Examples:
  python scripts/sweep_shap_k.py --k-list 40 60 80 --reports-dir reports --features-out reports/features_topK.txt \
    --cv-args "--folds 5 --n-jobs -1 --reduce corr --corr-threshold 0.99 --auto-drop-leakers"
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep SHAP top-K for CV training")
    p.add_argument("--k-list", type=int, nargs="+", required=True)
    p.add_argument("--reports-dir", type=str, default="reports")
    p.add_argument("--features-out", type=str, default="reports/features_topK.txt")
    p.add_argument("--cv-script", type=str, default="scripts/cv_train.py")
    p.add_argument("--cv-args", type=str, default="--folds 5 --n-jobs -1")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout or "", p.stderr or ""


def extract_auc(text: str) -> float | None:
    needle = "CV training completed. Overall OOF ROC AUC:".lower()
    for line in text.splitlines()[::-1]:
        if needle in line.lower():
            parts = line.strip().split()
            for tok in reversed(parts):
                try:
                    return float(tok)
                except Exception:
                    continue
    return None


def main() -> None:
    args = parse_args()
    results: list[tuple[int, float]] = []

    for k in args.k_list:
        # Export features
        code, out, err = run_cmd([sys.executable, "scripts/export_top_features.py", "--source", "shap", "--k", str(k), "--reports-dir", args.reports_dir, "--out", args.features_out])
        if code != 0:
            print(out + "\n" + err)
            raise SystemExit(code)

        # Run CV with features-file
        cv_cmd = [sys.executable, args.cv_script]
        cv_cmd.extend(shlex.split(args.cv_args))
        cv_cmd.extend(["--features-file", args.features_out])

        code, out, err = run_cmd(cv_cmd)
        text = out + "\n" + err
        auc = extract_auc(text)
        if auc is None:
            print(text)
            raise RuntimeError("Failed to parse AUC")
        print(f"K={k} -> OOF AUC={auc:.6f}")
        results.append((k, auc))

    # Write results
    import csv
    rep = Path(args.reports_dir)
    rep.mkdir(parents=True, exist_ok=True)
    out_csv = rep / "sweep_shap_k_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "oof_auc"])
        w.writerows(results)

    best_k, best_auc = max(results, key=lambda t: t[1])
    print(f"Best: K={best_k} OOF AUC={best_auc:.6f}")


if __name__ == "__main__":
    main()


