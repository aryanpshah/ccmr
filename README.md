# CCMR - Label-Efficient Whole-Heart CMR Segmentation (Swin-UNETR + LoRA)

This repository implements a full experimental pipeline to study label efficiency for congenital cardiac MRI (CMR) segmentation on HVSMR-2.0 style data. It compares a strong nnU-Net baseline against MONAI Swin-UNETR variants (scratch, full fine-tune, and parameter-efficient LoRA) across multiple label budgets, with standardized preprocessing, splits, training, and evaluation.

No personal or machine-specific information is included here.

## Project overview

### Goal

This project explores how much labeled data is actually required to train accurate whole-heart congenital CMR segmentation models. We systematically evaluate whether lightweight adaptation methods can match the performance of full fine-tuning as the amount of annotated data increases. By quantifying the trade-off between annotation effort and segmentation quality, this work aims to provide practical guidance for building clinically usable models under limited labeling budgets.

### Core hypotheses

- Strong baselines (nnU-Net) set an upper bound on performance under limited labels.
- Parameter-efficient adaptation (LoRA on Swin-UNETR) can approach full fine-tuning performance with far fewer trainable parameters.
- As label budget grows (L=5, 10, 20, 40), the performance gap between lightweight adaptation and full fine-tuning shrinks.

### Key comparisons

- Baseline: nnU-Net v2 (scratch training).
- Swin-UNETR (scratch).
- Swin-UNETR (full fine-tune from BTCV pretrained weights).
- Swin-UNETR + LoRA (adapters on attention Q/V projections, frozen backbone).

Metrics include per-class Dice, mean Dice (all and foreground), and Hausdorff distance (HD95). Additional analysis includes parameter counts, learning curves, and label-budget scaling behavior.

## Hyperparameters

The following hyperparameters are used in the provided scripts (adapted from nnU-Net defaults and MONAI BTCV settings):

| Hyperparameter | Scratch | Full Fine-tune | LoRA |
| --- | --- | --- | --- |
| Optimizer | AdamW | AdamW | AdamW |
| LR | lr_max=2e-4, lr_min=2e-5 | lr_max=6e-5, lr_min=6e-6 | lr_lora=5e-4, lr_backbone=1e-5 |
| Scheduler | cosine/poly to 0 | cosine/poly to 0 | cosine/poly to 0 |
| Weight decay | 1e-2 | 3e-3 | wd_lora=1e-2, wd_backbone=1e-4 |
| Patch size | 128 x 128 x 128 | 128 x 128 x 128 | 128 x 128 x 128 |
| Batch size | 1 | 1 | 1 |
| Max epochs | 300 | 300 | 300 |
| Early stopping | 60 | 60 | 60 |

## Repository layout (detailed file guide)

### Top-level

- `README.md`: this document.
- `requirements.txt`: pinned Python dependencies (MONAI, PyTorch, nnU-Net, etc.).
- `data/`: expected data root. Raw data, processed data, splits, and nnU-Net data live here.
- `pretrained/`: optional pretrained checkpoints (BTCV Swin-UNETR bundles).
- `logs/`, `runs/`, `results/`, `figures/`: output folders used by training and evaluation scripts.

### Models and training utilities

- `models/lora_utils.py`:
  - Implements LoRA adapters for Swin-UNETR attention Q/V projections.
  - Provides parameter counting, LoRA module summaries, and trainable parameter grouping helpers.

### Scripts (by purpose)

#### Data preprocessing and splits

- `scripts/preprocess_hvsmr.py`
  - Core preprocessing script for HVSMR-2.0 style data.
  - Resamples to 1 mm isotropic, center crops/pads to 192^3, saves to `data/processed/images`.
  - Creates stratified train/val/test splits (40/10/10) preserving severity balance.
  - Creates nested label budgets (L=5, 10, 20, 40) using a fixed seed.
  - Can also generate matched labels aligned to processed images.

- `scripts/plot_preprocessed.py`
  - Utility to visualize preprocessed volumes and masks to confirm shape and label integrity.

- `scripts/sanity_check_nnunet_datasets.py`
  - Validates nnU-Net dataset structures and split integrity.

#### Swin-UNETR data loading and model setup

- `scripts/swin_unetr_btcv_setup.py`
  - Shared utilities for Swin-UNETR training and inference.
  - Defines `create_model`, `create_hvsmr_loaders`, and file resolution logic.
  - Implements label-aware random cropping with optional foreground rejection and class-balanced ratios.
  - Contains a CLI entry point for one-off inference with a given checkpoint.

#### Swin-UNETR training

- `scripts/train_swin_unetr_scratch.py`
  - Trains Swin-UNETR from scratch (no pretrained weights).
  - Uses shared loaders and default class weights.

- `scripts/train_swin_unetr_finetune_btcv.py`
  - Full fine-tune of Swin-UNETR from BTCV pretrained weights.
  - Loads `model.pt`, replaces head for 9 classes, and trains all layers.

- `scripts/train_swin_unetr_lora.py`
  - LoRA adapter training on Swin-UNETR attention Q/V projections.
  - By default freezes the backbone and trains LoRA + decoder/head.
  - Supports rare-class sampling and CE reweighting (see below).

- `scripts/train_swin_unetr.py`
  - Legacy training script with a different preprocessing path.
  - Kept for reference only; not the primary training path.

#### Swin-UNETR evaluation and utilities

- `scripts/eval_swin_unetr.py`
  - Evaluates a trained Swin-UNETR on a test split.
  - Computes per-class Dice, mean Dice, and HD95; saves metrics JSON.

- `scripts/training_utils.py`
  - Helper functions for checkpointing and metric aggregation.

#### nnU-Net utilities

- `scripts/setup_nnunet_dataset.py`
  - Converts processed HVSMR data into nnU-Net v2 dataset format.
  - Creates dataset.json and organizes imagesTr/labelsTr.

- `scripts/add_labelsTs.py`
  - Creates labelsTs for existing nnU-Net datasets without altering splits.
  - Reuses the same mask processing pipeline as dataset setup.

- `scripts/build_labelsTs_from_dataset_json.py`
  - Copies labels for test IDs into labelsTs using dataset.json entries.
  - Helpful when test labels need to be packaged for evaluation.

- `scripts/train_nnunet_label_budgets.ps1`
  - PowerShell helper to run nnU-Net training across label budgets.

- `scripts/run_nnunet_inference_and_eval.sh`
  - Runs inference and evaluation for nnU-Net models.

- `scripts/parse_nnunet_log.py`, `scripts/extract_nnunet_metrics.py`
  - Log parsing and metric extraction utilities.

- `scripts/export_all_metrics.sh`
  - Batch-extracts metrics for all label budgets from nnU-Net results folders.

- `scripts/nnunet_env.ps1`
  - Sets nnU-Net v2 environment variables for the local `data/nnunet` layout.

#### Visualization

- `scripts/visualize_predictions.py`
  - Generates qualitative overlays of predictions vs ground truth.

#### Convenience launchers

- `scripts/launch_swin_unetr_scratch.sh`
  - Bash wrapper to launch scratch training with environment variables.

- `scripts/launch_swin_unetr_finetune.sh`
  - Bash wrapper to launch BTCV fine-tuning.

- `scripts/install_btcv_model.sh`
  - Helper to unpack BTCV model bundle zip and locate `model.pt`.

- `scripts/install_ngc.sh`
  - Placeholder for NGC CLI installation steps (currently empty).

## Environment setup

Create a virtual environment and install dependencies from `requirements.txt`.

Linux/macOS (bash):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify MONAI and torch import correctly:

```bash
python - << 'PY'
import torch
import monai
print('torch', torch.__version__)
print('monai', monai.__version__)
print('cuda', torch.cuda.is_available())
PY
```

## Remote GPU setup and MONAI installation (RunPod)

### RunPod workflow (SSH + sync + run)

1. Create a RunPod pod with a CUDA-enabled template and enough disk for data and checkpoints.
2. Open the pod and copy the SSH command from the RunPod UI (use the provided host/port/key).
3. SSH into the pod and set up the repo inside the workspace (most templates mount at `/workspace`):

```bash
cd /workspace
git clone <repo-url> ccmr
cd ccmr
```

4. Create and activate a virtual environment (see Environment setup above), then install dependencies.
5. Copy data to the pod (examples):

```bash
# From your local machine
rsync -av --progress data/ user@runpod:/workspace/ccmr/data/
```

```bash
# Or scp for a single archive
scp /path/to/data.zip user@runpod:/workspace/ccmr/
```

6. Keep training running with a session manager:

```bash
tmux new -s ccmr
```

Detach with `Ctrl+b` then `d`, and reattach with:

```bash
tmux attach -t ccmr
```

7. Monitor GPU usage:

```bash
nvidia-smi
```

### MONAI setup checklist

MONAI is installed via `requirements.txt`, but if you need to reinstall or update:

```bash
pip install --upgrade monai
```

Confirm versions are compatible:

```bash
python - << 'PY'
import monai
import torch
print('monai', monai.__version__)
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
PY
```

If CUDA is not detected, verify:

- GPU drivers are installed and match your CUDA runtime.
- The correct PyTorch build is installed (CUDA-enabled).
- The environment variable `CUDA_VISIBLE_DEVICES` is not hiding GPUs.

## Data preparation

### Raw data layout

The preprocessing script expects HVSMR-2.0 style NIfTI volumes and a clinical CSV:

- Raw NIfTI volumes under `data/raw/HVSMR2/cropped_norm`.
- Clinical metadata CSV at `data/raw/HVSMR2/hvsmr_clinical.csv`.

Files should contain patient IDs in the filename, such as `pat12`.

### Preprocessing and split generation

Run the preprocessing script to resample to 1 mm isotropic spacing, center crop/pad to 192^3, and generate train/val/test splits plus label-budget subsets.

```bash
python -u scripts/preprocess_hvsmr.py
```

Outputs:

- Processed images: `data/processed/images/patX_img_proc.nii.gz`
- Splits:
  - `data/splits/train_ids.txt`
  - `data/splits/val_ids.txt`
  - `data/splits/test_ids.txt`
  - label budgets: `data/splits/train_L5.txt`, `train_L10.txt`, `train_L20.txt`, `train_L40.txt`

The split creation is stratified by clinical severity when available.

### Matched labels for aligned training

If you have labels in a folder with the original affine and you want them aligned to the processed image grid, generate matched labels:

```bash
python -u scripts/preprocess_hvsmr.py \
  --make_matched_labels \
  --raw_label_dir data/processed/hvsmr2/labelsTr \
  --out_label_dir data/processed/hvsmr2/labelsTr_matched
```

Optional:

- `--case_ids pat12,pat34` to restrict to specific cases.
- `--skip_missing_labels` to continue if some labels are missing (writes `missing_labels.txt`).

### What the loaders search for

The shared loader in `scripts/swin_unetr_btcv_setup.py` resolves files flexibly. For each case ID, it tries multiple naming patterns:

Images:

- `{case_id}_image.nii.gz` or `.nii`
- `{case_id}_img_proc.nii.gz` or `.nii`
- `{case_id}.nii.gz` or `.nii`

Labels:

- `{case_id}_label.nii.gz` or `.nii`
- `{case_id}_seg.nii.gz` or `.nii`
- `{case_id}_cropped_seg.nii.gz` or `.nii`

Default search roots:

- `--data_root` (or `--data_root/imagesTr` if present)
- `--label_root` if provided
- Fallback roots: `data/processed/images`, `data/raw/HVSMR2/cropped_norm`

This means you can either:

- Use a standard nnU-Net style layout under `data/processed/hvsmr2/imagesTr` and `data/processed/hvsmr2/labelsTr`, or
- Keep processed images in `data/processed/images` and pass `--label_root` to point to your labels.

## Pretrained BTCV weights (optional)

If you want to fine-tune from the BTCV Swin-UNETR checkpoint, you need the bundle that contains `model.pt`.

Option A - manual download + install helper:

1. Download the BTCV Swin-UNETR bundle zip from the official NGC page.
2. Run the helper to unpack and locate `model.pt`:

```bash
bash scripts/install_btcv_model.sh /path/to/monai_swin_unetr_btcv_segmentation_0.5.6.zip
```

The script prints the path to `model.pt` which you can pass to `--pretrained_ckpt`.

Option B - use the NGC CLI:

- Install and configure NGC CLI with your API key.
- Download the bundle to a local folder.
- Use the extracted `model.pt` as the checkpoint.

Note: Access to NGC models depends on your account permissions. If CLI download fails, use the web UI download instead.

## Training workflows

### Common flags (all Swin-UNETR training scripts)

- `--data_root` path to a dataset root that contains images (and optionally labels).
- `--label_root` optional override for labels.
- `--train_split` and `--val_split` are required and should point to split text files.
- `--roi_size` size of 3D training patches (X Y Z).
- `--batch_size` number of patches per step.
- `--num_workers` DataLoader worker count.
- `--output_dir` where checkpoints and logs are written.

### Train from scratch

```bash
python -u scripts/train_swin_unetr_scratch.py \
  --data_root data/processed/hvsmr2 \
  --train_split data/splits/train_L40.txt \
  --val_split data/splits/val_ids.txt \
  --roi_size 128 128 128 \
  --batch_size 1 \
  --epochs 300 \
  --num_workers 4 \
  --output_dir runs/swin_unetr/L40_scratch
```

### Fine-tune from BTCV weights

```bash
python -u scripts/train_swin_unetr_finetune_btcv.py \
  --data_root data/processed/hvsmr2 \
  --train_split data/splits/train_L40.txt \
  --val_split data/splits/val_ids.txt \
  --roi_size 128 128 128 \
  --batch_size 1 \
  --epochs 300 \
  --num_workers 4 \
  --pretrained_ckpt /path/to/model.pt \
  --output_dir runs/swin_unetr/L40_btcv_finetune
```

### LoRA fine-tune (parameter efficient)

```bash
python -u scripts/train_swin_unetr_lora.py \
  --data_root data/processed/hvsmr2 \
  --label_root data/processed/hvsmr2/labelsTr_matched \
  --train_split data/splits/train_L40.txt \
  --val_split data/splits/val_ids.txt \
  --roi_size 128 128 128 \
  --batch_size 1 \
  --epochs 300 \
  --num_workers 4 \
  --output_dir runs/swin_unetr/L40_lora
```

### Rare class sampling and loss weighting (LoRA)

The LoRA script supports optional flags for rare class exposure and CE weighting. Defaults keep current behavior.

Sampling:

- `--num_samples_per_volume` controls patch count per volume for label-aware crops.
- `--rare_bias_78` biases crop ratios to emphasize classes 7 and 8.

Loss weighting:

- `--ce_bg_w` overrides the background CE weight.
- `--ce_w_78` sets CE weights for classes 7 and 8.

Example:

```bash
python -u scripts/train_swin_unetr_lora.py \
  --data_root data/processed/hvsmr2 \
  --label_root data/processed/hvsmr2/labelsTr_matched \
  --train_split data/splits/train_L40.txt \
  --val_split data/splits/val_ids.txt \
  --roi_size 128 128 128 \
  --batch_size 1 \
  --epochs 300 \
  --num_workers 4 \
  --output_dir runs/swin_unetr/L40_lora_rare \
  --num_samples_per_volume 4 \
  --rare_bias_78 \
  --ce_bg_w 0.02 \
  --ce_w_78 3.0
```

### Overfit/debug mode (LoRA)

`--overfit_debug` forces deterministic behavior, restricts data to a single case, and increases steps per epoch for sanity checking.

## Evaluation

Use `scripts/eval_swin_unetr.py` to compute Dice and HD95 metrics on a test split.

```bash
python -u scripts/eval_swin_unetr.py \
  --data_root data/processed/hvsmr2 \
  --test_split data/splits/test_ids.txt \
  --roi_size 128 128 128 \
  --checkpoint /path/to/best_model.pth \
  --output_json logs/swin_unetr/hvsmr2_eval_metrics.json
```

The script prints per-class metrics and writes a JSON summary.

## Single volume inference

`scripts/swin_unetr_btcv_setup.py` includes a CLI entry point that can run inference on a single volume. It expects a model checkpoint and saves a NIfTI mask to your output directory.

```bash
python -u scripts/swin_unetr_btcv_setup.py \
  --data_dir data/processed/hvsmr2 \
  --output_dir runs/infer \
  --checkpoint_path /path/to/btcv_model.pt \
  --run_inference /path/to/volume.nii.gz
```

## nnU-Net utilities (optional)

You can generate an nnU-Net v2 dataset layout from processed HVSMR volumes:

```bash
python -u scripts/setup_nnunet_dataset.py --label-budget 40
```

This writes to `data/nnunet/nnUNet_raw/Dataset940_HVSMR_L40` by default. Additional utilities exist in `scripts/` for nnU-Net training and evaluation.

## Visualization

To visualize nnU-Net predictions against ground truth:

```bash
python -u scripts/visualize_predictions.py \
  --dataset-id 940 \
  --tag L40 \
  --cases pat6 pat31 pat56 \
  --outdir figures/qualitative
```

## Outputs and logging

Typical outputs by script:

- `train_swin_unetr_scratch.py` and `train_swin_unetr_finetune_btcv.py` write `best_model.pt` and `last_model.pt` in `--output_dir`.
- `train_swin_unetr_lora.py` writes `best_model.pth` and `last_model.pth` plus `run_config.json` and `run_config.txt`.
- Evaluation outputs JSON metrics when `--output_json` is provided.

## Troubleshooting

- Import errors (for example, `ModuleNotFoundError: monai`): ensure your virtual environment is activated.
- CUDA out of memory: reduce `--roi_size`, `--batch_size`, or increase `--grad_accum`.
- Zero foreground Dice: verify labels are aligned and non-empty. Use rare-class sampling and CE weights in the LoRA script if rare classes are underrepresented.
- Missing labels or images: verify split files and ensure case IDs match filenames.

