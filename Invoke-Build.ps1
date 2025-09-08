<#
Requires -Version 5.1

Orchestrate end-to-end pipeline:
  preprocess -> feature_engineering -> cv_train -> export features -> cv_train with features -> build submission -> (optional) submit

Usage:
  .\Invoke-Build.ps1 -Submit:$false -Competition playground-series-s5e8
#>

[CmdletBinding()]
param(
  [switch]$Submit = $false,
  [string]$Competition = "playground-series-s5e8",
  [int]$Folds = 5,
  [int]$TopK = 60,
  [string]$CorrThreshold = "0.99"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "[1/7] Preprocess" -ForegroundColor Cyan
python scripts/data_preprocessing.py

Write-Host "[2/7] Feature engineering" -ForegroundColor Cyan
python scripts/feature_engineering.py --in data/combined.parquet --out data/combined_fe.parquet --log1p-balance --cap-pct 0.99 --add-interactions

Write-Host "[3/7] CV train (baseline)" -ForegroundColor Cyan
python scripts/cv_train.py --folds $Folds --n-jobs -1 --auto-drop-leakers --reduce corr --corr-threshold $CorrThreshold

Write-Host "[4/7] Export top-$TopK features (SHAP)" -ForegroundColor Cyan
python scripts/data_mining.py --shap --shap-n-samples 5000
python scripts/export_top_features.py --source shap --k $TopK --out reports/features_top$TopK.txt

Write-Host "[5/7] CV train (features locked)" -ForegroundColor Cyan
python scripts/cv_train.py --folds $Folds --n-jobs -1 --auto-drop-leakers --reduce corr --corr-threshold $CorrThreshold --features-file reports/features_top$TopK.txt

Write-Host "[6/7] Build submission" -ForegroundColor Cyan
python scripts/build_submission.py

Write-Host "[7/7] Archive reports" -ForegroundColor Cyan
python scripts/archive_reports.py --out-dir submissions

if ($Submit) {
  Write-Host "[Submit] Kaggle" -ForegroundColor Yellow
  $sub = (Get-ChildItem submissions\submission_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
  kaggle competitions submit -c $Competition -f "$sub" -m "Pipeline submit: $(Split-Path $sub -Leaf)"
}

Write-Host "Done." -ForegroundColor Green


