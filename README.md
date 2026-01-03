# CCMR - HVSMR2 Segmentation Toolkit (Swin-UNETR + LoRA)

This repository provides a complete workflow for cardiac MRI segmentation on HVSMR-2.0 style data using MONAI Swin-UNETR. It includes preprocessing, split generation, training (from scratch, BTCV fine-tune, and LoRA adapters), evaluation, and optional nnU-Net utilities.

Everything below is written to be environment-agnostic and does not include any personal or machine-specific information.

## Table of contents

- Overview
- Repository layout
- Requirements
- Environment setup
- Data preparation
  - Raw data layout
  - Preprocessing and split generation
  - Matched labels for aligned training
  - What the loaders search for
- Pretrained BTCV weights (optional)
- Training workflows
  - Common flags
  - Train from scratch
  - Fine-tune from BTCV weights
  - LoRA fine-tune (parameter efficient)
  - Rare class sampling and loss weighting
  - Overfit/debug mode
- Evaluation
- Single volume inference
- nnU-Net utilities (optional)
- Visualization
- Outputs and logging
- Troubleshooting

## Overview

- The core Swin-UNETR training and data loading utilities live in `scripts/swin_unetr_btcv_setup.py`.
- The training scripts are:
  - `scripts/train_swin_unetr_scratch.py` for scratch training.
  - `scripts/train_swin_unetr_finetune_btcv.py` for full fine-tuning from BTCV weights.
  - `scripts/train_swin_unetr_lora.py` for LoRA adapter training on attention Q/V projections.
- The evaluation script is `scripts/eval_swin_unetr.py`.
- The repo also contains an older baseline `scripts/train_swin_unetr.py` that uses a separate preprocessing path; it is kept for reference and is not the recommended path for new runs.

The Swin-UNETR scripts are configured for 9 classes by default (label 0 is background, labels 1-8 are structures). If your dataset uses a different label set, update `NUM_CLASSES` and related settings in `scripts/swin_unetr_btcv_setup.py` and ensure your labels match.

## Repository layout

Key paths and what they are used for:

- `scripts/` - all training, preprocessing, evaluation, and utility scripts.
- `models/` - model definitions and LoRA utilities.
- `data/` - expected data root (raw and processed data live here).
- `pretrained/` - optional pretrained checkpoints (for BTCV fine-tuning).
- `logs/` or `runs/` - default output folders for training artifacts (config files, checkpoints).

Important scripts:

- `scripts/preprocess_hvsmr.py` - preprocess raw HVSMR-2.0 data, resample, crop/pad, and generate split files.
- `scripts/swin_unetr_btcv_setup.py` - loader/model setup shared by training scripts.
- `scripts/train_swin_unetr_scratch.py` - Swin-UNETR training from scratch.
- `scripts/train_swin_unetr_finetune_btcv.py` - Swin-UNETR full fine-tuning from BTCV weights.
- `scripts/train_swin_unetr_lora.py` - LoRA adapter training on Swin-UNETR attention Q/V.
- `scripts/eval_swin_unetr.py` - evaluation on a test split.
- `scripts/install_btcv_model.sh` - helper to unpack a BTCV Swin-UNETR bundle zip and locate `model.pt`.
- `scripts/setup_nnunet_dataset.py` - create nnU-Net v2 dataset from processed volumes.
- `scripts/visualize_predictions.py` - simple qualitative overlays for nnU-Net predictions.

## Requirements

- Python 3.10 or 3.11 recommended.
- A CUDA-capable GPU is strongly recommended for training. CPU-only training will be extremely slow.
- Dependencies are pinned in `requirements.txt`.

Key dependencies include MONAI, PyTorch, nibabel, scikit-image, and nnU-Net v2.

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

## Remote GPU setup and MONAI installation

This section is tailored to a RunPod workflow and a MONAI-focused setup checklist.

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

For reproducibility on remote systems, consider fixing seeds and enabling deterministic behavior (already done in training scripts).

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

Use `scripts/train_swin_unetr_scratch.py` to train from random initialization.

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

Use `scripts/train_swin_unetr_finetune_btcv.py` to load a BTCV checkpoint and fine-tune all parameters.

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

Use `scripts/train_swin_unetr_lora.py` to train LoRA adapters on Swin-UNETR attention Q/V projections. By default the backbone is frozen and only LoRA and decoder/head parameters train.

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

Key LoRA flags:

- `--lora_rank` rank of the adapters (0 disables LoRA).
- `--lora_alpha` scaling factor.
- `--freeze_backbone` / `--no_freeze_backbone` to control backbone training.
- `--grad_accum` for gradient accumulation.

### Rare class sampling and loss weighting

The LoRA script exposes optional flags for rare class exposure and CE weighting. Defaults keep the current behavior.

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

### Overfit/debug mode

`--overfit_debug` in the LoRA script forces deterministic behavior, restricts data to a single case, and increases steps per epoch for sanity checking. This is useful when verifying preprocessing and label integrity.

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

`scripts/swin_unetr_btcv_setup.py` includes a CLI entry point that can run inference on a single volume. It expects a model checkpoint and saves a NIfTI mask next to your output directory.

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
- Zero foreground Dice: verify labels are aligned and non-empty. Use `--rare_bias_78` and CE weights in the LoRA script if rare classes are underrepresented.
- Missing labels or images: verify your split files and ensure case IDs match filenames.

## Notes on reproducibility

Most training scripts set a fixed random seed (default 42). The LoRA script also logs detailed run configuration and optimizer settings so runs are repeatable.
