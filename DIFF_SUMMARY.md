# Diff Summary: Anonymous Supplemental Release

## Changes for CHIL Supplemental

### New Files

- **supplemental_release/** — Minimal reproducibility bundle
  - README.md — Title, summary, data note, quickstart, anonymity statement
  - LICENSE — MIT
  - requirements.txt — numpy, scipy, matplotlib, pandas, nibabel, SimpleITK
  - splits/seed.txt — Seed 1337 and explanation
  - splits/generate_splits.py — Split generation (train 40, test 20, budgets L5/L10/L20/L40)
  - splits/README.txt — Instructions for generating split files
  - evaluation/compute_metrics.py — Per-case Dice and HD95
  - evaluation/aggregate_tables.py — Table 1 (macro), Table 2 (per-structure)
  - evaluation/paired_tests.py — Paired t-test and Wilcoxon
  - figures/make_fig_macro_curve.py — Figure 3 (macro Dice curve)
  - figures/make_fig_boxplot.py — Figure 4 (boxplot)
  - figures/make_fig_per_structure.py — Figure 6 (per-structure Dice/HD95)
  - figures/make_fig_qual_overlays.py — Qualitative overlays
  - configs/nnunet_protocol.md — nnU-Net v2 protocol summary
  - configs/env_notes.md — Environment notes
  - utils/io.py, metrics.py, plotting.py — Shared utilities
  - artifacts/dummy_*.csv — Dummy CSVs for validation

### Modified Files (Identity Scrubbing)

- **.gitignore** — Expanded to cover outputs, data, envs, archives, denylist
- **README.md** — Removed RunPod section and workspace paths
- **scripts/make_qual_montage.py** — Replaced /workspace/ccmr with env-based path
- **scripts/eval_testset_metrics.py** — Replaced /workspace/ccmr with /path/to/nnUNet_raw
- **scripts/run_nnunet_training.ps1** — Removed /workspace/ccmr from comment
- **run_scratch_all.sh** — Replaced cd /workspace/ccmr with cd "$(dirname "$0")"
- **QUICK_START.md** — Removed github.com/MIC-DKFZ/nnUNet link

### Excluded from Supplemental

- data/, outputs/, figures/, runs/, logs/, pretrained/
- nnUNet_raw/, nnUNet_preprocessed/, nnUNet_results/
- All Swin-UNETR scripts (paper is nnU-Net only)
- Training scripts, install scripts, internal notes

### Branch

- Created branch: anonymous-release

### Deliverable

- **chil_supplemental_code.zip** — Contains only supplemental_release/ contents
- Command: `Compress-Archive -Path supplemental_release\* -DestinationPath chil_supplemental_code.zip -Force`
