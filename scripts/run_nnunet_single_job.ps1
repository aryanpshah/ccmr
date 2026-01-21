param(
  [Parameter(Mandatory=$true)][string]$Budget,      # L10/L20/L40
  [Parameter(Mandatory=$true)][int]$DatasetID,      # 910/920/940
  [Parameter(Mandatory=$true)][string]$SeedsCsv,    # "0" or "0,1"
  [string]$ResultsTag = "gpuX",
  [int]$Fold = 0,
  [string]$Config3D = "3d_fullres",
  [string]$Trainer = "nnUNetTrainer",
  [string]$Plans = "nnUNetPlans"
)

$ErrorActionPreference = "Stop"
function Timestamp { (Get-Date -Format "yyyyMMdd_HHmmss") }

# Parse seeds in a bash-safe way (no @(...) needed)
$Seeds = @()
foreach ($s in $SeedsCsv.Split(",")) {
  $t = $s.Trim()
  if ($t -ne "") { $Seeds += [int]$t }
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Shared raw/preprocessed
$env:nnUNet_raw          = Join-Path $ProjectRoot "data/nnunet/nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $ProjectRoot "data/nnunet/nnUNet_preprocessed"

# Unique results root per GPU/job (prevents overwrite)
$env:nnUNet_results      = Join-Path $ProjectRoot ("data/nnunet/nnUNet_results_{0}" -f $ResultsTag)

New-Item -ItemType Directory -Force -Path $env:nnUNet_raw          | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results      | Out-Null

# Allow non-L5 setup in this job (for your guard)
$env:ALLOW_NNUNET_NONL5_SETUP = "1"

# Make python output unbuffered so logs stream live
$env:PYTHONUNBUFFERED = "1"

Write-Host "ProjectRoot: $ProjectRoot" -ForegroundColor Green
Write-Host "nnUNet_results: $env:nnUNet_results" -ForegroundColor Yellow
Write-Host "Budget=$Budget DatasetID=$DatasetID Seeds=$($Seeds -join ',')" -ForegroundColor White
Write-Host ""

$LogDir = Join-Path $ProjectRoot ("logs/nnunet_{0}" -f $ResultsTag)
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Invoke-LoggedExe {
  param(
    [string]$Exe,
    [Alias("Args")][string[]]$ArgList,
    [string]$LogPath
  )
  $pretty = "$Exe " + ($ArgList -join " ")
  Write-Host "    CMD: $pretty" -ForegroundColor Gray

  if ($DryRun) {
    Write-Host "    [DRY RUN] skipping execution" -ForegroundColor Yellow
    return 0
  }

  & $Exe @ArgList 2>&1 | Tee-Object -FilePath $LogPath -Append

  if ($LASTEXITCODE -ne 0) {
    throw "Command failed (exit $LASTEXITCODE): $pretty"
  }
  return 0
}

$BudgetNum   = [int]($Budget.TrimStart("L"))
$DatasetName = "Dataset${DatasetID}_HVSMR_${Budget}"

foreach ($Seed in $Seeds) {

  Write-Host "  Setup: $DatasetName (seed $Seed)" -ForegroundColor Cyan
  $setupLog = Join-Path $LogDir ("setup_{0}_seed{1}_{2}.log" -f $DatasetName, $Seed, (Timestamp))
  $setupArgs = @(
    "-u", "scripts/setup_nnunet_dataset.py",
    "--dataset-id", "$DatasetID",
    "--label-budget", "$BudgetNum",
    "--seed", "$Seed"
  )
  Invoke-LoggedExe -Exe "python" -ArgList $setupArgs -LogPath $setupLog

  # Preprocess guard: only run if plans missing
  $plansPath = Join-Path $env:nnUNet_preprocessed (Join-Path $DatasetName "nnUNetPlans.json")
  if (-not (Test-Path $plansPath)) {
    Write-Host "  Plan+preprocess: $DatasetName (DatasetID=$DatasetID)" -ForegroundColor Cyan
    $ppLog  = Join-Path $LogDir ("preprocess_{0}_{1}.log" -f $DatasetName, (Timestamp))
    $ppArgs = @("-d", "$DatasetID", "--verify_dataset_integrity")
    Invoke-LoggedExe -Exe "nnUNetv2_plan_and_preprocess" -ArgList $ppArgs -LogPath $ppLog
  } else {
    Write-Host "  Preprocess exists: $plansPath" -ForegroundColor DarkGray
  }

  # Train (default nnU-Net schedule/settings)
  Write-Host "  Train: $DatasetName (seed $Seed)" -ForegroundColor Magenta
  $trainLog = Join-Path $LogDir ("train_{0}_seed{1}_{2}.log" -f $DatasetName, $Seed, (Timestamp))
  $trainArgs = @(
    $DatasetName, $Config3D, "$Fold",
    "-tr", $Trainer,
    "-p", $Plans
  )
  Invoke-LoggedExe -Exe "nnUNetv2_train" -ArgList $trainArgs -LogPath $trainLog

  Write-Host ""
}

Write-Host "Done: $ResultsTag" -ForegroundColor Green
