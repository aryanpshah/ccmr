# nnU-Net Evaluation Pipeline
# Runs inference, computes metrics, aggregates results, and generates all figures

param(
    [switch]$SkipInference = $false,
    [switch]$SkipMetrics = $false,
    [switch]$SkipPlots = $false,
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "nnU-Net Evaluation Pipeline" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$PROJECT_ROOT = $PSScriptRoot
if ($PROJECT_ROOT -eq "") {
    $PROJECT_ROOT = Get-Location
}

$DATA_ROOT = Join-Path $PROJECT_ROOT "data"
$LOG_DIR = Join-Path $PROJECT_ROOT "logs\nnunet"
$RESULTS_DIR = Join-Path $PROJECT_ROOT "results"
$FIGURES_DIR = Join-Path $PROJECT_ROOT "figures"
$NNUNET_ROOT = Join-Path $DATA_ROOT "nnunet"

# nnU-Net environment variables
$env:nnUNet_raw = Join-Path $NNUNET_ROOT "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $NNUNET_ROOT "nnUNet_preprocessed"
$env:nnUNet_results = Join-Path $NNUNET_ROOT "nnUNet_results"

$TEST_IDS_FILE = Join-Path $DATA_ROOT "splits\test_ids.txt"
$SEVERITY_CSV = Join-Path $DATA_ROOT "raw\HVSMR2\hvsmr_clinical.csv"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Project root: $PROJECT_ROOT" -ForegroundColor White
Write-Host "  Results dir: $RESULTS_DIR" -ForegroundColor White
Write-Host "  Figures dir: $FIGURES_DIR" -ForegroundColor White
Write-Host "  Test IDs: $TEST_IDS_FILE" -ForegroundColor White
Write-Host ""

# Create directories
New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RESULTS_DIR "tables") | Out-Null
New-Item -ItemType Directory -Force -Path $FIGURES_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $FIGURES_DIR "qualitative") | Out-Null

# Define configurations
$CONFIGS = @(
    @{Budget='L5'; DatasetID=905; Seeds=@(0, 1)},
    @{Budget='L10'; DatasetID=910; Seeds=@(0, 1)},
    @{Budget='L20'; DatasetID=920; Seeds=@(0)},
    @{Budget='L40'; DatasetID=940; Seeds=@(0)}
)

$FOLD = 0
$CONFIG_3D = "3d_fullres"
$TRAINER = "nnUNetTrainer"
$PLANS = "nnUNetPlans"

# ============================================================================
# Step 1: Run Inference
# ============================================================================
if (-not $SkipInference) {
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Step 1: Running Inference" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    
    foreach ($Config in $CONFIGS) {
        $Budget = $Config.Budget
        $DatasetID = $Config.DatasetID
        $Seeds = $Config.Seeds
        
        $DatasetName = "Dataset${DatasetID}_HVSMR_${Budget}"
        $RawDatasetDir = Join-Path $env:nnUNet_raw $DatasetName
        $ImagesTs = Join-Path $RawDatasetDir "imagesTs"
        
        foreach ($Seed in $Seeds) {
            Write-Host "Inference: $Budget, Seed $Seed" -ForegroundColor Cyan
            
            $PredDir = Join-Path $NNUNET_ROOT "predictions\${Budget}\seed${Seed}"
            New-Item -ItemType Directory -Force -Path $PredDir | Out-Null
            
            $ResultsPath = Join-Path $env:nnUNet_results "$DatasetName\${TRAINER}__${PLANS}__${CONFIG_3D}"
            $CheckpointFile = Join-Path $ResultsPath "fold_${FOLD}\checkpoint_best.pth"
            
            if (-not (Test-Path $CheckpointFile)) {
                Write-Host "  WARNING: Checkpoint not found: $CheckpointFile" -ForegroundColor Yellow
                Write-Host "  Skipping inference for $Budget seed $Seed" -ForegroundColor Yellow
                continue
            }
            
            $InferCmd = "nnUNetv2_predict -i `"$ImagesTs`" -o `"$PredDir`" -d $DatasetID -c $CONFIG_3D -f $FOLD -chk checkpoint_best.pth"
            
            if ($DryRun) {
                Write-Host "  [DRY RUN] $InferCmd" -ForegroundColor Yellow
            } else {
                Write-Host "  Executing: $InferCmd" -ForegroundColor Gray
                Invoke-Expression $InferCmd
                
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "  WARNING: Inference failed for $Budget seed $Seed" -ForegroundColor Yellow
                } else {
                    Write-Host "  Success: Predictions saved to $PredDir" -ForegroundColor Green
                }
            }
            
            Write-Host ""
        }
    }
} else {
    Write-Host "Skipping inference (--SkipInference)" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# Step 2: Compute Per-Case Metrics
# ============================================================================
if (-not $SkipMetrics) {
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Step 2: Computing Metrics" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    
    $AllWideCsvs = @()
    $AllLongCsvs = @()
    
    foreach ($Config in $CONFIGS) {
        $Budget = $Config.Budget
        $DatasetID = $Config.DatasetID
        $Seeds = $Config.Seeds
        
        $DatasetName = "Dataset${DatasetID}_HVSMR_${Budget}"
        $GtDir = Join-Path $env:nnUNet_raw "$DatasetName\labelsTs"
        
        foreach ($Seed in $Seeds) {
            Write-Host "Computing metrics: $Budget, Seed $Seed" -ForegroundColor Cyan
            
            $PredDir = Join-Path $NNUNET_ROOT "predictions\${Budget}\seed${Seed}"
            $WideCsv = Join-Path $RESULTS_DIR "${Budget}_seed${Seed}_per_case.csv"
            $LongCsv = Join-Path $RESULTS_DIR "${Budget}_seed${Seed}_long.csv"
            
            if (-not (Test-Path $PredDir)) {
                Write-Host "  WARNING: Prediction directory not found: $PredDir" -ForegroundColor Yellow
                continue
            }
            
            # Compute per-case metrics
            $MetricsCmd = "python scripts\compute_per_case_metrics.py --gt_dir `"$GtDir`" --pred_dir `"$PredDir`" --test_ids `"$TEST_IDS_FILE`" --out_csv `"$WideCsv`" --budget $Budget --seed $Seed"
            
            if ($DryRun) {
                Write-Host "  [DRY RUN] $MetricsCmd" -ForegroundColor Yellow
            } else {
                Write-Host "  Computing per-case metrics..." -ForegroundColor Gray
                Invoke-Expression $MetricsCmd
                
                if ($LASTEXITCODE -eq 0) {
                    $AllWideCsvs += $WideCsv
                    Write-Host "  Success: $WideCsv" -ForegroundColor Green
                } else {
                    Write-Host "  ERROR: Metrics computation failed" -ForegroundColor Red
                }
            }
            
            # Convert to long format
            if (Test-Path $WideCsv) {
                $LongCmd = "python scripts\wide_to_long_csv.py --wide_csv `"$WideCsv`" --severity_csv `"$SEVERITY_CSV`" --out_csv `"$LongCsv`""
                
                if ($DryRun) {
                    Write-Host "  [DRY RUN] $LongCmd" -ForegroundColor Yellow
                } else {
                    Write-Host "  Converting to long format..." -ForegroundColor Gray
                    Invoke-Expression $LongCmd
                    
                    if ($LASTEXITCODE -eq 0) {
                        $AllLongCsvs += $LongCsv
                        Write-Host "  Success: $LongCsv" -ForegroundColor Green
                    }
                }
            }
            
            Write-Host ""
        }
    }
    
    # ========================================================================
    # Step 2b: Aggregate Metrics
    # ========================================================================
    Write-Host "Aggregating metrics across all runs..." -ForegroundColor Cyan
    
    if ($AllLongCsvs.Count -gt 0) {
        $LongCsvsStr = ($AllLongCsvs | ForEach-Object { "`"$_`"" }) -join " "
        $AggCmd = "python scripts\aggregate_metrics.py --input_csvs $LongCsvsStr --out_dir `"$RESULTS_DIR\tables`""
        
        if ($DryRun) {
            Write-Host "  [DRY RUN] $AggCmd" -ForegroundColor Yellow
        } else {
            Write-Host "  Executing: $AggCmd" -ForegroundColor Gray
            Invoke-Expression $AggCmd
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  Success: Tables saved to $RESULTS_DIR\tables" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "  No long CSV files found to aggregate" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # ========================================================================
    # Step 2c: Extract Compute Stats
    # ========================================================================
    Write-Host "Extracting compute statistics..." -ForegroundColor Cyan
    
    $ComputeStatsCSV = Join-Path $RESULTS_DIR "compute_stats.csv"
    
    foreach ($Config in $CONFIGS) {
        $Budget = $Config.Budget
        $DatasetID = $Config.DatasetID
        $Seeds = $Config.Seeds
        
        $DatasetName = "Dataset${DatasetID}_HVSMR_${Budget}"
        
        foreach ($Seed in $Seeds) {
            # Find most recent log file for this config
            $LogPattern = "nnunet_${Budget}_fold${FOLD}_seed${Seed}_*.log"
            $LogFiles = Get-ChildItem -Path $LOG_DIR -Filter $LogPattern | Sort-Object LastWriteTime -Descending
            
            if ($LogFiles.Count -eq 0) {
                Write-Host "  WARNING: No log file found for $Budget seed $Seed" -ForegroundColor Yellow
                continue
            }
            
            $LogFile = $LogFiles[0].FullName
            $ResultsPath = Join-Path $env:nnUNet_results "$DatasetName\${TRAINER}__${PLANS}__${CONFIG_3D}\fold_${FOLD}"
            $PredDir = Join-Path $NNUNET_ROOT "predictions\${Budget}\seed${Seed}"
            
            $StatsCmd = "python scripts\extract_compute_stats.py --log_file `"$LogFile`" --results_dir `"$ResultsPath`" --budget $Budget --seed $Seed --out_csv `"$ComputeStatsCSV`" --pred_dir `"$PredDir`""
            
            if ($DryRun) {
                Write-Host "  [DRY RUN] $StatsCmd" -ForegroundColor Yellow
            } else {
                Write-Host "  Extracting stats for $Budget seed $Seed..." -ForegroundColor Gray
                Invoke-Expression $StatsCmd
            }
        }
    }
    
    Write-Host "Compute stats saved to: $ComputeStatsCSV" -ForegroundColor Green
    Write-Host ""
    
} else {
    Write-Host "Skipping metrics computation (--SkipMetrics)" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# Step 3: Generate Plots
# ============================================================================
if (-not $SkipPlots) {
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "Step 3: Generating Plots" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    
    # Find all long CSV files
    $LongCsvs = Get-ChildItem -Path $RESULTS_DIR -Filter "*_long.csv" | ForEach-Object { $_.FullName }
    
    if ($LongCsvs.Count -eq 0) {
        Write-Host "No long CSV files found in $RESULTS_DIR" -ForegroundColor Yellow
        Write-Host "Please run metrics computation first" -ForegroundColor Yellow
    } else {
        Write-Host "Found $($LongCsvs.Count) long CSV files" -ForegroundColor Cyan
        
        $LongCsvsStr = ($LongCsvs | ForEach-Object { "`"$_`"" }) -join " "
        
        # Use L40 for qualitative examples (best performance)
        $GtDir = Join-Path $env:nnUNet_raw "Dataset940_HVSMR_L40\labelsTs"
        $PredDirsRoot = Join-Path $NNUNET_ROOT "predictions"
        
        $PlotCmd = "python scripts\generate_plots.py --long_csvs $LongCsvsStr --pred_dirs `"$PredDirsRoot`" --gt_dir `"$GtDir`" --out_dir `"$FIGURES_DIR`""
        
        if ($DryRun) {
            Write-Host "[DRY RUN] $PlotCmd" -ForegroundColor Yellow
        } else {
            Write-Host "Generating all figures..." -ForegroundColor Cyan
            Invoke-Expression $PlotCmd
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Success: Figures saved to $FIGURES_DIR" -ForegroundColor Green
            } else {
                Write-Host "WARNING: Some plots may have failed" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host ""
} else {
    Write-Host "Skipping plots (--SkipPlots)" -ForegroundColor Yellow
    Write-Host ""
}

# ============================================================================
# Summary
# ============================================================================
Write-Host "================================================" -ForegroundColor Green
Write-Host "Evaluation Pipeline Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:" -ForegroundColor Yellow
Write-Host "  Per-case metrics: $RESULTS_DIR\*_per_case.csv" -ForegroundColor White
Write-Host "  Long format: $RESULTS_DIR\*_long.csv" -ForegroundColor White
Write-Host "  Summary tables: $RESULTS_DIR\tables\" -ForegroundColor White
Write-Host "  Compute stats: $RESULTS_DIR\compute_stats.csv" -ForegroundColor White
Write-Host "  Figures: $FIGURES_DIR\" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review figures in: $FIGURES_DIR" -ForegroundColor White
Write-Host "  2. Check summary tables in: $RESULTS_DIR\tables" -ForegroundColor White
Write-Host "  3. Review compute stats: $RESULTS_DIR\compute_stats.csv" -ForegroundColor White
Write-Host ""
