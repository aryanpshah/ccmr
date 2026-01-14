# nnU-Net Training Script for Label Budget Experiments
# Runs training for L5/L10/L20/L40 with specified seeds
# Maintains consistent hyperparameters across all runs

param(
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "nnU-Net Label Budget Training Pipeline" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$PROJECT_ROOT = $PSScriptRoot
if ($PROJECT_ROOT -eq "") {
    $PROJECT_ROOT = Get-Location
}

$DATA_ROOT = Join-Path $PROJECT_ROOT "data"
$LOG_DIR = Join-Path $PROJECT_ROOT "logs\nnunet"
$NNUNET_ROOT = Join-Path $DATA_ROOT "nnunet"

# nnU-Net environment variables
$env:nnUNet_raw = Join-Path $NNUNET_ROOT "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $NNUNET_ROOT "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $NNUNET_ROOT "nnUNet_results"

Write-Host "Project root: $PROJECT_ROOT" -ForegroundColor Yellow
Write-Host "nnUNet_raw: $env:nnUNet_raw" -ForegroundColor Yellow
Write-Host "nnUNet_results: $env:nnUNet_results" -ForegroundColor Yellow
Write-Host ""

# Create directories
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_raw | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_preprocessed | Out-Null
New-Item -ItemType Directory -Force -Path $env:nnUNet_results | Out-Null

# Define training configurations
# Format: @{Budget='L5'; DatasetID=905; NumSeeds=3}
$CONFIGS = @(
    @{Budget='L5'; DatasetID=905; Seeds=@(0, 1, 2)},
    @{Budget='L10'; DatasetID=910; Seeds=@(0, 1, 2)},
    @{Budget='L20'; DatasetID=920; Seeds=@(0, 1)},
    @{Budget='L40'; DatasetID=940; Seeds=@(0, 1)}
)

# Hyperparameters (kept consistent across all runs)
$FOLD = 0  # Using fold 0 only
$CONFIG_3D = "3d_fullres"
$TRAINER = "nnUNetTrainer"
$PLANS = "nnUNetPlans"

Write-Host "Training Configuration:" -ForegroundColor Green
Write-Host "  Fold: $FOLD" -ForegroundColor White
Write-Host "  Trainer: $TRAINER" -ForegroundColor White
Write-Host "  Plans: $PLANS" -ForegroundColor White
Write-Host "  Config: $CONFIG_3D" -ForegroundColor White
Write-Host ""

# Function to setup dataset with specific seed
function Setup-NNUNetDataset {
    param(
        [string]$Budget,
        [int]$DatasetID,
        [int]$Seed
    )
    
    $DatasetName = "Dataset${DatasetID}_HVSMR_${Budget}"
    
    Write-Host "  Setting up dataset: $DatasetName (seed $Seed)" -ForegroundColor Cyan
    
    # Run setup script
    $SetupCmd = "python scripts\setup_nnunet_dataset.py --dataset_id $DatasetID --budget $Budget --seed $Seed"
    
    if ($DryRun) {
        Write-Host "    [DRY RUN] Would execute: $SetupCmd" -ForegroundColor Yellow
    } else {
        Write-Host "    Executing: $SetupCmd" -ForegroundColor Gray
        Invoke-Expression $SetupCmd
        
        if ($LASTEXITCODE -ne 0) {
            throw "Dataset setup failed for $DatasetName"
        }
    }
}

# Function to run preprocessing
function Run-NNUNetPreprocessing {
    param(
        [int]$DatasetID
    )
    
    Write-Host "  Running preprocessing for dataset $DatasetID..." -ForegroundColor Cyan
    
    $PreprocCmd = "nnUNetv2_plan_and_preprocess -d $DatasetID --verify_dataset_integrity"
    
    if ($DryRun) {
        Write-Host "    [DRY RUN] Would execute: $PreprocCmd" -ForegroundColor Yellow
    } else {
        Write-Host "    Executing: $PreprocCmd" -ForegroundColor Gray
        Invoke-Expression $PreprocCmd
        
        if ($LASTEXITCODE -ne 0) {
            throw "Preprocessing failed for dataset $DatasetID"
        }
    }
}

# Function to run training
function Run-NNUNetTraining {
    param(
        [string]$Budget,
        [int]$DatasetID,
        [int]$Seed
    )
    
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LOG_DIR "nnunet_${Budget}_fold${FOLD}_seed${Seed}_${Timestamp}.log"
    
    Write-Host "  Training nnU-Net: Budget=$Budget, Seed=$Seed" -ForegroundColor Green
    Write-Host "    Log file: $LogFile" -ForegroundColor Gray
    
    # Set seed for reproducibility
    $env:PYTHONHASHSEED = $Seed
    
    $TrainCmd = "nnUNetv2_train $DatasetID $CONFIG_3D $FOLD --npz -tr $TRAINER -p $PLANS"
    
    if ($DryRun) {
        Write-Host "    [DRY RUN] Would execute: $TrainCmd" -ForegroundColor Yellow
        Write-Host "    [DRY RUN] Log would be saved to: $LogFile" -ForegroundColor Yellow
    } else {
        Write-Host "    Executing: $TrainCmd" -ForegroundColor Gray
        Write-Host "    Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
        
        # Run with output to both console and log file
        $TrainCmd | Tee-Object -FilePath $LogFile | Invoke-Expression
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    WARNING: Training returned non-zero exit code" -ForegroundColor Yellow
        }
        
        Write-Host "    Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    }
    
    return $LogFile
}

# Main training loop
$TotalRuns = ($CONFIGS | ForEach-Object { $_.Seeds.Count }) | Measure-Object -Sum | Select-Object -ExpandProperty Sum

Write-Host "Starting training for $TotalRuns total runs..." -ForegroundColor Green
Write-Host ""

$RunNumber = 0

foreach ($Config in $CONFIGS) {
    $Budget = $Config.Budget
    $DatasetID = $Config.DatasetID
    $Seeds = $Config.Seeds
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Budget: $Budget (Dataset $DatasetID)" -ForegroundColor Cyan
    Write-Host "Seeds: $($Seeds -join ', ')" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check if preprocessing is needed (only once per dataset)
    $PreprocessedDir = Join-Path $env:nnUNet_preprocessed "Dataset${DatasetID}_HVSMR_${Budget}"
    
    if (-not (Test-Path $PreprocessedDir)) {
        Write-Host "Preprocessing not found. Running preprocessing..." -ForegroundColor Yellow
        
        # Setup dataset with first seed for preprocessing
        Setup-NNUNetDataset -Budget $Budget -DatasetID $DatasetID -Seed $Seeds[0]
        
        # Run preprocessing
        Run-NNUNetPreprocessing -DatasetID $DatasetID
        
        Write-Host ""
    } else {
        Write-Host "Preprocessing already exists: $PreprocessedDir" -ForegroundColor Green
        Write-Host ""
    }
    
    # Train for each seed
    foreach ($Seed in $Seeds) {
        $RunNumber++
        
        Write-Host "----------------------------------------" -ForegroundColor White
        Write-Host "Run $RunNumber/$TotalRuns : $Budget, Seed $Seed" -ForegroundColor White
        Write-Host "----------------------------------------" -ForegroundColor White
        
        # Setup dataset with specific seed (for different train splits)
        Setup-NNUNetDataset -Budget $Budget -DatasetID $DatasetID -Seed $Seed
        
        # Train
        $LogFile = Run-NNUNetTraining -Budget $Budget -DatasetID $DatasetID -Seed $Seed
        
        Write-Host "  Completed: $Budget, Seed $Seed" -ForegroundColor Green
        Write-Host ""
    }
}

Write-Host "================================================" -ForegroundColor Green
Write-Host "All training runs completed!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run inference and evaluation: .\scripts\run_evaluation_pipeline.ps1" -ForegroundColor White
Write-Host "  2. Check logs in: $LOG_DIR" -ForegroundColor White
Write-Host ""
