#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/processed/hvsmr2}"
TRAIN_SPLIT="${TRAIN_SPLIT:-data/splits/train_L5.txt}"
VAL_SPLIT="${VAL_SPLIT:-data/splits/val_ids.txt}"
ROI_SIZE="${ROI_SIZE:-96 96 96}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PRETRAINED="${PRETRAINED:-pretrained/model.pt}"
OUT_DIR="${OUT_DIR:-logs/swin_unetr/L5_finetune_run}"

echo "Launching Swin-UNETR finetune..."
python -u scripts/train_swin_unetr_finetune_btcv.py \
  --data_root "${DATA_ROOT}" \
  --train_split "${TRAIN_SPLIT}" \
  --val_split "${VAL_SPLIT}" \
  --roi_size ${ROI_SIZE} \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --pretrained_ckpt "${PRETRAINED}" \
  --output_dir "${OUT_DIR}"
