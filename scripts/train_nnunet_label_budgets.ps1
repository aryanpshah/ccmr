# Train nnU-Net v2 (3d_fullres, fold 0) for all label budgets.
# Run from repo root: pwsh .\scripts\train_nnunet_label_budgets.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Move to repo root (parent of the scripts directory)
Set-Location -Path (Resolve-Path "$PSScriptRoot/..")

# nnU-Net environment (sets nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
. .\scripts\nnunet_env.ps1

# Constrain CPU threading / dataloader workers to ease RAM and Windows handle pressure.
$env:nnUNet_n_proc_DA = "1"
$env:nnUNet_n_proc_val = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_MAX_THREADS = "8"   # raise cap to avoid nthreads errors
$env:NUMEXPR_NUM_THREADS = "1"   # actual threads used by numexpr
$env:PYTORCH_NUM_THREADS = "2"

# Dataset ID -> label tag
$configs = @{
    905 = "L5"
    910 = "L10"
    920 = "L20"
    940 = "L40"
}

$trainerExe = ".\.venv\Scripts\nnUNetv2_train.exe"
$logsDir = "logs\nnunet"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$summary = @()

foreach ($id in $configs.Keys | Sort-Object) {
    $label = $configs[$id]
    $datasetName = "Dataset${id}_HVSMR_${label}"
    $trainerDir = Join-Path $env:nnUNet_results "$datasetName\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0"
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logPath = Join-Path $logsDir ("nnunet_{0}_fold0_{1}.log" -f $label, $timestamp)

    Write-Host ""
    Write-Host "==== Training nnU-Net for Dataset $id ($label) ===="

    if (Test-Path $trainerDir) {
        Write-Host "Skipping $id ($label), fold 0 already trained at $trainerDir"
        $summary += "Dataset $id ($label): skipped (already exists)"
        continue
    }

    Write-Host "Starting training -> $logPath"
    # Stream stdout/stderr to console and log file simultaneously to see epoch updates.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $trainerExe $id 3d_fullres 0 -device cpu 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed for $id ($label) with exit code $LASTEXITCODE. See $logPath"
        $summary += "Dataset $id ($label): FAILED (exit $LASTEXITCODE)"
        $ErrorActionPreference = $prevEAP
        continue
    }
    $ErrorActionPreference = $prevEAP
    $summary += "Dataset $id ($label): trained this run"
}

Write-Host ""
Write-Host "==== Summary ===="
$summary | ForEach-Object { Write-Host $_ }
