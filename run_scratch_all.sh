#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

# Run from the repository root.
cd "$(dirname "$0")"
source .venv/bin/activate

COMMON_ARGS=(
  --data_root data/processed/hvsmr2
  --val_split data/splits/val_ids.txt
  --epochs 500
  --batch_size 1
  --lr 1e-4
  --roi_size 128 128 128
  --warmup_epochs 10
  --patience 40
)

for L in 5 10 20 40; do
  TRAIN_SPLIT="data/splits/train_L${L}.txt"
  RUN_DIR="runs/scratch_L${L}_FIXED"
  LOG="$RUN_DIR/train.log"

  if [[ ! -f "$TRAIN_SPLIT" ]]; then
    echo "ERROR: missing split file: $TRAIN_SPLIT" >&2
    exit 1
  fi

  mkdir -p "$RUN_DIR"

  {
    echo "============================================"
    echo "START  L=$L   $(date)"
    echo "TRAIN_SPLIT: $TRAIN_SPLIT"
    echo "RUN_DIR:     $RUN_DIR"
    echo "CMD: python -u scripts/train_swin_unetr_scratch.py --train_split $TRAIN_SPLIT --output_dir $RUN_DIR ${COMMON_ARGS[*]}"
    echo "============================================"
  } | tee "$LOG"

  python -u scripts/train_swin_unetr_scratch.py \
    --train_split "$TRAIN_SPLIT" \
    --output_dir "$RUN_DIR" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "$LOG"

  echo "DONE   L=$L   $(date)" | tee -a "$LOG"
done
