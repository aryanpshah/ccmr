#!/usr/bin/env bash
set -euo pipefail

# Activate venv (assumes you are in repo root)
cd "$(dirname "$0")/.."
source .venv/bin/activate

# nnU-Net envs
export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw
export nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed
export nnUNet_results=$PWD/data/nnunet/nnUNet_results

mkdir -p logs/nnunet data/nnunet/predictions

# (dataset_id, label_tag) pairs
declare -a DATASETS=(
  "905 L5"
  "910 L10"
  "920 L20"
  "940 L40"
)

for pair in "${DATASETS[@]}"; do
  set -- $pair
  ds_id="$1"
  tag="$2"

  echo ""
  echo "==============================="
  echo "Processing dataset ${ds_id} (label budget ${tag})"
  echo "==============================="

  RAW_DS="${nnUNet_raw}/Dataset${ds_id}_HVSMR_${tag}"
  RES_DS="${nnUNet_results}/Dataset${ds_id}_HVSMR_${tag}/nnUNetTrainer__nnUNetPlans__3d_fullres"

  IMAGES_TS="${RAW_DS}/imagesTs"
  LABELS_TS="${RAW_DS}/labelsTs"
  PRED_DIR="data/nnunet/predictions/${tag}"

  mkdir -p "${PRED_DIR}"

  echo "[${tag}] Inference with nnUNetv2_predict..."
  nnUNetv2_predict \
    -chk checkpoint_best.pth \
    -i "${IMAGES_TS}" \
    -o "${PRED_DIR}" \
    -d "${ds_id}" \
    -c 3d_fullres \
    -chk checkpoint_best.pth \
    -f 0

  echo "[${tag}] Evaluation with nnUNetv2_evaluate_folder..."
  # This will compute metrics (Dice etc.) between LABELS_TS and predictions in PRED_DIR
  # and write a summary (usually summary.json) plus print to stdout.
  nnUNetv2_evaluate_folder \
    "${LABELS_TS}" \
    "${PRED_DIR}" \
    -djfile "${RES_DS}/dataset.json" \
    -pfile "${RES_DS}/plans.json" \
    > "logs/nnunet/eval_${tag}.txt"

  echo "[${tag}] Done. Metrics log: logs/nnunet/eval_${tag}.txt"
done

echo ""
echo "All label budgets processed."
