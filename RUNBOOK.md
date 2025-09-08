## Binary Classification with a Bank Dataset — Run Book

This run book provides end-to-end, copy/pasteable commands for Windows PowerShell. Adjust paths if needed.

### 1) Prerequisites
- Python 3.11+ with `pip`
- Git, GitHub access to repo
- Kaggle API token configured at `%USERPROFILE%\.kaggle\kaggle.json`

Install core dependencies:
```powershell
python -m pip install -r requirements.txt
```

### 2) Data preprocessing and feature engineering
```powershell
# Preprocess (outputs data/train_processed.csv, data/test_processed.csv)
python scripts/data_preprocessing.py

# Feature engineering (optional, creates data/combined_fe.parquet)
python scripts/feature_engineering.py --in data/combined.parquet --out data/combined_fe.parquet --log1p-balance --cap-pct 0.99 --add-interactions
```

### 3) Cross-validated training (XGBoost, GPU if available)
```powershell
# Base CV with correlation pruning and auto leaker drop
python scripts/cv_train.py --folds 5 --n-jobs -1 --reduce corr --corr-threshold 0.99 --auto-drop-leakers
```

### 4) Data mining and reports (incl. SHAP)
```powershell
python scripts/data_mining.py --shap --shap-n-samples 5000
python scripts/visualize_reports.py --top-n 30
```
Outputs: `reports/*.csv`, `reports/figures/*.png`.

### 5) Feature selection and sweeps
Export top-K features:
```powershell
python scripts/export_top_features.py --source shap --k 60 --out reports/features_top60.txt
```
Retrain with features locked:
```powershell
python scripts/cv_train.py --folds 5 --n-jobs -1 --reduce corr --corr-threshold 0.99 --auto-drop-leakers --features-file reports/features_top60.txt
```
Sweep SHAP top-K:
```powershell
python scripts/sweep_shap_k.py --k-list 40 60 80 --features-out reports/features_topK.txt --cv-args "--folds 5 --n-jobs -1 --reduce corr --corr-threshold 0.99 --auto-drop-leakers"
```
Sweep correlation threshold:
```powershell
python scripts/sweep_corr_threshold.py --thresholds 0.995 0.99 0.985 0.98 --folds 5 --n-jobs -1
```

### 6) Build submission and submit to Kaggle
```powershell
python scripts/build_submission.py
$sub = (Get-ChildItem submissions\submission_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$COMP = "playground-series-s5e8"  # change if different
kaggle competitions submit -c $COMP -f "$sub" -m "Auto-submit: $(Split-Path $sub -Leaf)"
```
Check status:
```powershell
kaggle competitions submissions -c $COMP --csv | ConvertFrom-Csv | Select-Object fileName,description,date,publicScore,privateScore | Format-Table -AutoSize
```

Optional reusable submitter: `Submit-Kaggle.ps1` (already added). Run:
```powershell
.\Submit-Kaggle.ps1 -Competition "playground-series-s5e8" -Message "Locked features"
```

### 7) Archive reports for reproducibility
```powershell
python scripts/archive_reports.py --out-dir submissions
```
Outputs a timestamped zip under `submissions/`.

### 8) Dashboard (local Streamlit) and Cloud
Local:
```powershell
python -m streamlit run scripts\dashboard.py --server.address 127.0.0.1 --server.port 8625 --server.headless true
```
Cloud: Deploy repo `Pedurabo/Binary-Classification-with-a-Bank-Dataset`, app file `scripts/dashboard.py`.

### 9) CI workflow (GitHub Actions)
Workflow: `.github/workflows/train_and_archive.yml`.
Trigger via GitHub → Actions → "Train and Archive Reports" → Run workflow.

### 10) BI exports (Tableau/Power BI)
Create exports folder and copy key CSVs:
```powershell
New-Item -ItemType Directory -Force -Path bi\exports | Out-Null
Copy-Item data\train_processed.csv bi\exports\train.csv -ErrorAction SilentlyContinue
Copy-Item data\test_processed.csv  bi\exports\test.csv  -ErrorAction SilentlyContinue
Copy-Item reports\univariate_auc.csv           bi\exports\univariate_auc.csv      -ErrorAction SilentlyContinue
Copy-Item reports\mutual_info.csv              bi\exports\mutual_info.csv         -ErrorAction SilentlyContinue
Copy-Item reports\target_correlations.csv      bi\exports\target_correlations.csv -ErrorAction SilentlyContinue
Copy-Item reports\model_importance.csv         bi\exports\model_importance.csv    -ErrorAction SilentlyContinue
Copy-Item reports\shap_importance.csv          bi\exports\shap_importance.csv     -ErrorAction SilentlyContinue
```
Optional: zip for handoff
```powershell
New-Item -ItemType Directory -Force -Path submissions | Out-Null
Compress-Archive -Path bi\exports\* -DestinationPath submissions\bi_exports.zip -Force
```

### 11) DuckDB warehouse (optional local warehouse)
Create/refresh:
```powershell
python -m pip install duckdb
$code = @'
import duckdb, pandas as pd, pathlib as p
wd = duckdb.connect("warehouse.duckdb")
train = pd.read_csv("data/train_processed.csv"); test = pd.read_csv("data/test_processed.csv")
wd.execute("CREATE SCHEMA IF NOT EXISTS bank")
wd.register("train_view", train); wd.execute("CREATE OR REPLACE TABLE bank.fact_train AS SELECT * FROM train_view")
wd.register("test_view",  test);  wd.execute("CREATE OR REPLACE TABLE bank.fact_test  AS SELECT * FROM test_view")
print("DuckDB refreshed: warehouse.duckdb")
'@
Set-Content -Encoding UTF8 -Path .\_duckdb_refresh.py -Value $code
python .\_duckdb_refresh.py
Remove-Item .\_duckdb_refresh.py
```

### 12) One-command pipeline (local)
Use `Invoke-Build.ps1`:
```powershell
.\Invoke-Build.ps1 -Submit:$false -Competition "playground-series-s5e8" -Folds 5 -TopK 60 -CorrThreshold 0.99
```

---
Notes
- Prefer GPU if available; CPU fallback is automatic in `cv_train.py`.
- Large artifacts (e.g., `warehouse.duckdb`) should be ignored or tracked via Git LFS.
