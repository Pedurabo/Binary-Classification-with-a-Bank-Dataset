param(
  [string]$Competition = "playground-series-s5e8",
  [string]$SubPattern = "submissions\submission_*.csv",
  [string]$Message = $null
)

# UTF-8 console to avoid encoding issues
chcp 65001 > $null
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Ensure Kaggle CLI and config
python -m pip install -U kaggle | Out-Null
if (-not $env:KAGGLE_CONFIG_DIR) {
  $env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle"
}

# Find latest submission file
$latest = Get-ChildItem $SubPattern -ErrorAction Stop | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latest) { Write-Error "No submission files found matching: $SubPattern"; exit 1 }

$subPath = $latest.FullName
$subName = Split-Path $subPath -Leaf
if (-not $Message) { $Message = "Auto-submit: $subName" }

Write-Host "Submitting $subName to competition '$Competition'..." -ForegroundColor Cyan
kaggle competitions submit -c $Competition -f "$subPath" -m "$Message"
if ($LASTEXITCODE -ne 0) { Write-Error "Kaggle submit failed."; exit $LASTEXITCODE }

Start-Sleep -Seconds 5
$csv = kaggle competitions submissions -c $Competition --csv
$rows = $csv | ConvertFrom-Csv
$mine = $rows | Where-Object { $_.fileName -eq $subName }
if ($mine) {
  $mine | Select-Object fileName, description, date, publicScore, privateScore | Format-Table -AutoSize
} else {
  $rows | Select-Object fileName, description, date, publicScore, privateScore -First 5 | Format-Table -AutoSize
}
