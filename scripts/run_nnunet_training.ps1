param(
    [string[]]$Budgets = @("L5","L10","L20","L40"),
    [int]$Fold = 0,
    [string]$Config3D = "3d_fullres",
    [string]$Trainer = "nnUNetTrainer",
    [string]$Plans = "nnUNetPlans",
    [switch]$DryRun = $false,

    # If true: if training is interrupted (Ctrl+C) or exits non-zero, skip to the next run instead of stopping the whole script.
    [switch]$ContinueOnTrainFailure = $true,

    # If true: archive any existing Dataset<ID>_* dirs out of nnUNet_raw/preprocessed/results before each seed run
    [switch]$ArchiveBeforeEachSeed = $true
)

$ErrorActionPreference = "Stop"

# ProjectRoot is the repo root (parent of /scripts), resolved relative to this script
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Write-Host "ProjectRoot: $ProjectRoot" -ForegroundColor Green

# Export nnU-Net paths (equivalent to bash export nnUNet_raw=..., etc.)
$env:nnUNet_raw          = Join-Path $ProjectRoot "data/nnunet/nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $ProjectRoot "data/nnunet/nnUNet_preprocessed"
$env:nnUNet_results      = Join-Path $ProjectRoot "data/nnunet/nnUNet_results"

New-Item -ItemType Directory -Force -Path $env:nnUNet_raw          | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results      | Out-Null

Write-Host "nnUNet_raw:          $env:nnUNet_raw" -ForegroundColor White
Write-Host "nnUNet_preprocessed: $env:nnUNet_preprocessed" -ForegroundColor White
Write-Host "nnUNet_results:      $env:nnUNet_results" -ForegroundColor White
Write-Host ""

# Logs
$LogDir = Join-Path $ProjectRoot "logs/nnunet"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Timestamp {
    return (Get-Date -Format "yyyyMMdd_HHmmss")
}

function Invoke-LoggedExe {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [Parameter(Mandatory=$true)][string[]]$Args,
        [Parameter(Mandatory=$true)][string]$LogPath,
        [switch]$AllowFailure = $false
    )

    $pretty = $Exe + " " + ($Args -join " ")
    Write-Host "    CMD: $pretty" -ForegroundColor Gray
    if ($DryRun) {
        Write-Host "    [DRY RUN] skipping execution" -ForegroundColor Yellow
        return $true
    }

    try {
        # Stream stdout+stderr to console AND write to log
        & $Exe @Args 2>&1 | Tee-Object -FilePath $LogPath -Append

        if ($LASTEXITCODE -ne 0) {
            if ($AllowFailure) {
                Write-Host "    [WARN] Exit code $LASTEXITCODE (continuing)." -ForegroundColor Yellow
                return $false
            }
            throw "Command failed (exit $LASTEXITCODE): $pretty"
        }

        return $true
    }
    catch [System.Management.Automation.PipelineStoppedException] {
        # This commonly happens when you hit Ctrl+C during a piped command
        if ($AllowFailure) {
            Write-Host "    [WARN] Interrupted (Ctrl+C) (continuing)." -ForegroundColor Yellow
            return $false
        }
        throw
    }
    catch {
        if ($AllowFailure) {
            Write-Host "    [WARN] Failed/interrupted (continuing)." -ForegroundColor Yellow
            return $false
        }
        throw
    }
}

# --- AUTO-ARCHIVE (prevent seed overwrites + avoid DatasetID collisions) ---
function Archive-NNUNetDataset {
    param(
        [int]$DatasetID,
        [string]$Tag
    )

    $ArchiveRoot = Join-Path $ProjectRoot "data/nnunet/_archives"
    $StampDir = Join-Path $ArchiveRoot $Tag

    $Targets = @(
        @{ Name = "nnUNet_results"; Path = $env:nnUNet_results }
    )
foreach ($t in $Targets) {
        $src = $t.Path
        if (-not (Test-Path $src)) { continue }

        $dst = Join-Path $StampDir $t.Name
        New-Item -ItemType Directory -Force -Path $dst | Out-Null

        $pattern = "Dataset$DatasetID" + "_*"
        $found = Get-ChildItem -Path $src -Directory -Filter $pattern -ErrorAction SilentlyContinue

        foreach ($d in $found) {
            $to = Join-Path $dst $d.Name
            Write-Host "    [ARCHIVE] Moving $($d.FullName) -> $to" -ForegroundColor Yellow
            Move-Item -Force -Path $d.FullName -Destination $to
        }
    }
}
# --- END AUTO-ARCHIVE ---

# Define training configs
$CONFIGS = @(
    @{Budget='L5';  DatasetID=905; Seeds=@(0, 1)},
    @{Budget='L10'; DatasetID=910; Seeds=@(0, 1)},
    @{Budget='L20'; DatasetID=920; Seeds=@(0)},
    @{Budget='L40'; DatasetID=940; Seeds=@(0)}
)

Write-Host "Training Configuration:" -ForegroundColor Green
Write-Host "  Budgets: $($Budgets -join ', ')" -ForegroundColor White
Write-Host "  Fold:    $Fold" -ForegroundColor White
Write-Host "  Trainer: $Trainer" -ForegroundColor White
Write-Host "  Plans:   $Plans" -ForegroundColor White
Write-Host "  Config:  $Config3D" -ForegroundColor White
Write-Host "  ContinueOnTrainFailure: $ContinueOnTrainFailure" -ForegroundColor White
Write-Host "  ArchiveBeforeEachSeed:  $ArchiveBeforeEachSeed" -ForegroundColor White
Write-Host ""

function Setup-NNUNetDataset {
    param(
        [string]$Budget,
        [int]$DatasetID,
        [int]$Seed
    )

    $BudgetNum = [int]($Budget.TrimStart("L"))   # L5 -> 5
    $DatasetName = "Dataset${DatasetID}_HVSMR_${Budget}"

    if ($ArchiveBeforeEachSeed) {
        $tag = ("{0}_seed{1}_{2}" -f $DatasetName, $Seed, (Timestamp))
        Archive-NNUNetDataset -DatasetID $DatasetID -Tag $tag
    }

    Write-Host "  Setting up dataset: $DatasetName (seed $Seed)" -ForegroundColor Cyan

    $setupArgs = @(
        "scripts/setup_nnunet_dataset.py",
        "--dataset-id", "$DatasetID",
        "--label-budget", "$BudgetNum",
        "--seed", "$Seed"
    )
    $setupLog = Join-Path $LogDir ("setup_{0}_seed{1}_{2}.log" -f $DatasetName, $Seed, (Timestamp))

    # Setup failures should stop the script (do not allow failure here)
    [void](Invoke-LoggedExe -Exe "python" -Args $setupArgs -LogPath $setupLog)


    # Ensure preprocessing exists (plans file + preprocessed data). Run ONCE per DatasetID (guarded by plans file).
    $plansPath = Join-Path $env:nnUNet_preprocessed (Join-Path $DatasetName "nnUNetPlans.json")
    if (-not (Test-Path $plansPath)) {
        Write-Host "  Planning+preprocessing: $DatasetName (DatasetID=$DatasetID)" -ForegroundColor Cyan

        $ppArgs = @(
            "-d", "$DatasetID",
            "--verify_dataset_integrity"
        )
        $ppLog = Join-Path $LogDir ("preprocess_{0}_{1}.log" -f $DatasetName, (Timestamp))
        [void](Invoke-LoggedExe -Exe "nnUNetv2_plan_and_preprocess" -Args $ppArgs -LogPath $ppLog)
    } else {
        Write-Host "  Preprocess already exists: $plansPath" -ForegroundColor DarkGray
    }

    return $DatasetName
}

function Train-NNUNet {
    param(
        [string]$DatasetName,
        [int]$Seed
    )

    Write-Host "  Training: $DatasetName (seed $Seed)" -ForegroundColor Magenta

    $trainArgs = @(
        $DatasetName,
        $Config3D,
        "$Fold",
        "-tr", $Trainer,
        "-p", $Plans
    )
    $trainLog = Join-Path $LogDir ("train_{0}_seed{1}_{2}.log" -f $DatasetName, $Seed, (Timestamp))

    $ok = Invoke-LoggedExe -Exe "nnUNetv2_train" -Args $trainArgs -LogPath $trainLog -AllowFailure:$ContinueOnTrainFailure
    if (-not $ok) {
        Write-Host "  [SKIP] Moving to next config/seed." -ForegroundColor Yellow
    }
}

foreach ($cfg in $CONFIGS) {
    if ($Budgets -notcontains $cfg.Budget) { continue }

    foreach ($seed in $cfg.Seeds) {
        $ds = Setup-NNUNetDataset -Budget $cfg.Budget -DatasetID $cfg.DatasetID -Seed $seed
        Train-NNUNet -DatasetName $ds -Seed $seed
        Write-Host ""
    }
}

Write-Host "Done." -ForegroundColor Green
