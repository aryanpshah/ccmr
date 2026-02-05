# Quick Start Guide - nnU-Net Label Budget Experiments

## One-Command Execution

### Full Pipeline (Training + Evaluation)

```powershell
# Step 1: Train all models (L5/L10/L20/L40 with multiple seeds)
.\scripts\run_nnunet_training.ps1

# Step 2: Run inference, compute metrics, and generate figures
.\scripts\run_evaluation_pipeline.ps1
```

**Total Time:** ~150-200 GPU hours for training + ~2-4 hours for evaluation

---

## Outputs You'll Get

### 📊 Metrics & Tables

- `results/*_per_case.csv` - Detailed per-case, per-structure metrics
- `results/tables/` - Summary statistics with confidence intervals
- `results/compute_stats.csv` - Training time, GPU usage, etc.

### 📈 Figures

- Label-budget scaling curves (Dice & HD95)
- Per-structure learning curves
- Structure learning rankings (which structures learn fastest?)
- Severity-stratified performance
- Qualitative overlays (best/median/worst cases)

### 📝 Logs

- `logs/nnunet/` - Complete training logs for all runs

---

## What Gets Trained

| Budget | Dataset ID | Training Cases | Seeds | Total Runs |
| ------ | ---------- | -------------- | ----- | ---------- |
| L5     | 905        | 5              | 3     | 3          |
| L10    | 910        | 10             | 3     | 3          |
| L20    | 920        | 20             | 2     | 2          |
| L40    | 940        | 40             | 2     | 2          |

**Total:** 10 training runs, all evaluated on same 10-case test set

---

## Directory Structure After Completion

```
your-project/
├── data/
│   ├── nnunet/
│   │   ├── nnUNet_raw/           # Datasets
│   │   ├── nnUNet_preprocessed/  # Preprocessed data
│   │   ├── nnUNet_results/       # Trained models
│   │   └── predictions/          # Test predictions
│   └── splits/                   # Train/val/test splits
│
├── results/
│   ├── L5_seed0_per_case.csv
│   ├── L5_seed0_long.csv
│   ├── ... (all budgets/seeds)
│   ├── compute_stats.csv
│   └── tables/
│       ├── dice_by_structure.csv
│       ├── hd95_by_structure.csv
│       └── summary_table.tex
│
├── figures/
│   ├── scaling_curve_dice_overall.png
│   ├── scaling_curve_hd95_overall.png
│   ├── structure_learning_rankings.png
│   ├── severity_stratified_performance.png
│   └── overlay_*.png
│
├── logs/
│   └── nnunet/
│       └── nnunet_L5_fold0_seed0_*.log
│
└── scripts/
    ├── run_nnunet_training.ps1
    └── run_evaluation_pipeline.ps1
```

---

## Command Options

### Training Script

```powershell
# Dry run (see what would be executed without running)
.\scripts\run_nnunet_training.ps1 -DryRun

# Normal run
.\scripts\run_nnunet_training.ps1
```

### Evaluation Script

```powershell
# Full pipeline
.\scripts\run_evaluation_pipeline.ps1

# Skip inference (use existing predictions)
.\scripts\run_evaluation_pipeline.ps1 -SkipInference

# Skip metrics (use existing CSVs)
.\scripts\run_evaluation_pipeline.ps1 -SkipMetrics

# Only generate plots from existing data
.\scripts\run_evaluation_pipeline.ps1 -SkipInference -SkipMetrics

# Dry run
.\scripts\run_evaluation_pipeline.ps1 -DryRun
```

---

## Key Metrics

### Dice Coefficient

- **Range:** 0-1 (higher = better)
- **Meaning:** Overlap between prediction and ground truth
- **Good Performance:** > 0.85

### Hausdorff Distance 95 (HD95)

- **Units:** millimeters
- **Meaning:** 95th percentile of surface distances
- **Good Performance:** < 5mm

### AULC (Area Under Learning Curve)

- **Meaning:** How quickly a structure learns
- **Higher = learns faster with fewer labels**

---

## Troubleshooting

### "CUDA out of memory"

```powershell
# Reduce batch size in nnU-Net plans
# Or use smaller GPU-compatible configuration
```

### "Missing checkpoint_best.pth"

```powershell
# Training incomplete - check logs
Get-Content logs\nnunet\*.log | Select-String "error"
```

### "No CSV files found"

```powershell
# Run evaluation pipeline in order:
# 1. Inference
# 2. Metrics computation
# 3. Plotting
.\scripts\run_evaluation_pipeline.ps1
```

---

## What to Send to Client

### Complete Package

```
📦 Send these folders:
  ├── results/          # All metrics and tables
  ├── figures/          # All generated plots
  ├── scripts/          # Reproduction scripts
  └── CLIENT_DELIVERABLES.md
```

### Results Only

```
📦 Minimal package:
  ├── results/
  ├── figures/
  └── CLIENT_DELIVERABLES.md
```

---

## Next Steps After Running

1. **Review Figures**: Check `figures/` directory
2. **Examine Tables**: Look at `results/tables/` for summary stats
3. **Check Compute Stats**: Review `results/compute_stats.csv` for training times
4. **Validate Results**: Ensure metrics look reasonable (Dice > 0.6, HD95 < 20mm)
5. **Package for Client**: Copy results + figures + documentation

---

## Time Estimates

| Task                  | Duration           |
| --------------------- | ------------------ |
| Dataset setup         | ~10 min per budget |
| Preprocessing         | ~30 min per budget |
| Training L5 (1 seed)  | ~6-8 hours         |
| Training L10 (1 seed) | ~8-12 hours        |
| Training L20 (1 seed) | ~12-18 hours       |
| Training L40 (1 seed) | ~20-30 hours       |
| Inference (all runs)  | ~30 min            |
| Metrics computation   | ~15 min            |
| Figure generation     | ~5 min             |

**Total Pipeline:** ~150-200 GPU hours + ~1 hour CPU

---

## Support

- **Documentation:** See `CLIENT_DELIVERABLES.md` for detailed info
- **Logs:** Check `logs/nnunet/` for error messages
- **nnU-Net Help:** See nnU-Net documentation

---

_Last updated: January 2026_
