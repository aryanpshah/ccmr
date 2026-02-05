# Label-Budget Scaling Benchmark for Whole-Heart CMR Segmentation (HVSMR-2.0)

Supplemental code for the paper: "Label-Budget Scaling in Whole-Heart Cardiac MRI Segmentation: A Structure-Wise Benchmark on HVSMR-2.0 with nnU-Net v2."

## Summary

This repository provides fixed splits, evaluation scripts, and figure-generation scripts sufficient to reproduce the paper's reported metrics and figures. The benchmark uses a fixed test set (n=20), a training pool of 40 cases, and nested label budgets L5, L10, L20, L40. Budgets are derived from a single shuffle with seed 1337 (prefix subsets). Metrics include per-structure Dice and HD95, macro-averaged across the 8 cardiac structures.

**Outputs:** Scripts reproduce Table 1 (macro metrics), Table 2 (per-structure Dice), Figure 3 (macro Dice curve), Figure 4 (boxplot of per-case macro Dice), Figure 6 (per-structure Dice and HD95 curves), and qualitative overlay figures when predictions are available.

## Data Note

HVSMR-2.0 is a public dataset but must be obtained separately. This supplemental does not redistribute the dataset. Users must download HVSMR-2.0 and place it in a directory of their choice.

## Contents

- **splits/** — Split generation script and seed documentation. Run `generate_splits.py` with `--data_root` pointing to your HVSMR-2.0 data to create `train_pool_ids.txt`, `test_ids.txt`, and `budgets/L5_ids.txt`, `L10_ids.txt`, `L20_ids.txt`, `L40_ids.txt`.
- **evaluation/** — Scripts to compute per-case Dice and HD95, aggregate into tables, and run paired statistical tests.
- **figures/** — Scripts to generate macro curve, boxplot, per-structure curves, and qualitative overlays.

## Expected Input Layout

To run evaluation and figure generation, the following layout is expected:

1. **Ground truth labels:** NIfTI files (e.g., `{case_id}.nii.gz`) in a directory such as `labelsTs` or `labels`.
2. **Predictions:** One directory per budget (L5, L10, L20, L40), each containing `{case_id}.nii.gz` prediction files.
3. **Test IDs:** Text file with one case ID per line (e.g., `splits/test_ids.txt`).

Example structure:

```
data_root/
  labelsTs/           # Ground truth
    pat6.nii.gz
    pat11.nii.gz
    ...
  predictions/
    L5/
      pat6.nii.gz
      ...
    L10/
      ...
    L20/
      ...
    L40/
      ...

splits/
  test_ids.txt
```

## Quickstart

### 1. Create environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate splits (requires HVSMR-2.0 data)

```bash
python splits/generate_splits.py --data_root /path/to/HVSMR-2.0 --splits_dir splits --out_dir .
```

### 3. Compute per-case metrics (per budget)

```bash
python evaluation/compute_metrics.py --gt_dir /path/to/labelsTs --pred_dir /path/to/predictions/L5 --test_ids splits/test_ids.txt --budget L5 --seed 0 --out_csv artifacts/L5_per_case.csv
# Repeat for L10, L20, L40.
```

### 4. Aggregate tables

```bash
python evaluation/aggregate_tables.py --input_csvs artifacts/L5_per_case.csv artifacts/L10_per_case.csv artifacts/L20_per_case.csv artifacts/L40_per_case.csv --out_dir artifacts
```

### 5. Run paired statistical tests

```bash
python evaluation/paired_tests.py --input_csvs artifacts/L5_per_case.csv artifacts/L10_per_case.csv artifacts/L20_per_case.csv artifacts/L40_per_case.csv --out_txt artifacts/stat_tests_fg_dice.txt
```

### 6. Generate figures

```bash
python figures/make_fig_macro_curve.py --metrics_csv artifacts/budget_macro_metrics.csv --out_dir artifacts
python figures/make_fig_boxplot.py --input_csvs artifacts/L5_per_case.csv artifacts/L10_per_case.csv artifacts/L20_per_case.csv artifacts/L40_per_case.csv --out_dir artifacts
python figures/make_fig_per_structure.py --dice_csv artifacts/per_structure_dice_by_budget.csv --hd95_csv artifacts/per_structure_hd95_by_budget.csv --out_dir artifacts
```

For qualitative overlays (requires image, GT, and predictions):

```bash
python figures/make_fig_qual_overlays.py --case pat6 --gt_path /path/to/labelsTs/pat6.nii.gz --img_path /path/to/imagesTs/pat6_0000.nii.gz --pred_l5 /path/to/L5 --pred_l10 /path/to/L10 --pred_l20 /path/to/L20 --pred_l40 /path/to/L40 --out_dir artifacts
```

## Reproducibility

Scripts reproduce the reported metrics and figures given identical predictions and ground truth labels. The split generation uses a fixed seed (1337) and nested prefix budgets. All outputs are written to `artifacts/` by default.

## Anonymity

This repository is anonymized for double-blind peer review.
