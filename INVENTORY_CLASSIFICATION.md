# Supplemental Release Inventory

## A) KEEP (necessary for reproducing results/figures)

| File | Purpose |
|------|---------|
| scripts/preprocess_hvsmr.py | Split generation logic (seed 1337, budgets) |
| scripts/compute_per_case_metrics.py | Dice + HD95 per structure |
| scripts/aggregate_metrics.py | Summary tables, bootstrap CI |
| scripts/wide_to_long_csv.py | Format conversion for aggregation |
| scripts/recompute_metrics_from_preds.py | Metrics from predictions (nnU-Net layout) |
| scripts/stat_tests_fg_dice.py | Paired t-test, Wilcoxon |
| scripts/plot_budget_curves.py | Macro + per-structure curves |
| scripts/plot_budget_curves_pretty_from_metrics.py | HD95 curve, boxplot |
| scripts/plot_per_class_dice_pretty.py | Per-structure Dice curve |
| scripts/make_per_structure_hd95_median_iqr_table.py | HD95 table |
| scripts/render_multi_budget_overlays.py | Qualitative overlays |
| scripts/visualize_predictions.py | Overlay generation |
| models/lora_utils.py | (excluded - Swin-UNETR, not paper scope) |

## B) KEEP BUT EDIT/SCRUB

| File | Issue |
|------|-------|
| scripts/make_qual_montage.py | Hardcoded /workspace/ccmr/ |
| scripts/eval_testset_metrics.py | Default /workspace/ccmr/data/nnunet/ |
| scripts/run_nnunet_training.ps1 | /workspace/ccmr |
| run_scratch_all.sh | cd /workspace/ccmr |
| README.md | RunPod section, git clone URLs |

## C) EXCLUDE

- data/, outputs/, figures/, runs/, logs/, pretrained/
- nnUNet_raw/, nnUNet_preprocessed/, nnUNet_results/
- *.nii.gz, *.pt, *.pth, *.ckpt, *.npz, *.pkl
- .venv/, __pycache__/, wandb/, tensorboard/
- _archives/, *.zip (except final supplemental)
- FIXES_IMPLEMENTED.md, IMPLEMENTATION_COMPLETE.txt, CURRENT_PROBLEMS_*.txt
- PROBLEM_SUMMARY.md, QUICK_REFERENCE.md, QUICK_START.md (internal notes)
- All Swin-UNETR training scripts (paper is nnU-Net only)
- install_btcv_model.sh, install_ngc.sh, nnunet_env.ps1 (env-specific)

## D) UNKNOWN (default exclude)

- analyze_current_problems.py, diagnose_data.py, validate_fixes.py
- Shell launchers (launch_*.sh, run_*.sh) - environment-specific
