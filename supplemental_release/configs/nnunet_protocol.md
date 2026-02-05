# nnU-Net v2 Protocol (HVSMR-2.0)

## Dataset Setup

- **Source:** HVSMR-2.0 (public; obtain separately)
- **Structures:** 8 foreground classes (LV, RV, LA, RA, Aorta, PA, SVC, IVC)
- **Preprocessing:** 1 mm isotropic, center crop/pad to 192^3

## Splits

- **Train pool:** 40 cases
- **Test set:** 20 cases (fixed, held out)
- **Label budgets:** L5, L10, L20, L40 (nested prefixes of shuffled train pool)
- **Shuffle seed:** 1337

## Training

- nnU-Net v2 default planner and trainer
- 3D full-resolution configuration
- 5-fold cross-validation within each budget (optional; paper reports single-fold or default)

## Evaluation

- **Metrics:** Per-structure Dice, 95th percentile Hausdorff distance (HD95)
- **Macro average:** Mean across 8 structures per case, then mean across cases
- **Paired tests:** t-test and Wilcoxon on per-case macro Dice
