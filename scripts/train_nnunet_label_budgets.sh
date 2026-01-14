#!/usr/bin/env bash
set -euo pipefail

# Train nnU-Net v2 (3d_fullres, fold 0) for all label budgets.
# Run from repo root: bash scripts/train_nnunet_label_budgets.sh

cd "$(dirname "$0")/.."

# nnU-Net environment (sets nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
# Your repo has nnunet_env.ps1 for Windows; on Linux you should have a .sh equivalent.
# If you already export these elsewhere, you can comment this out.
if [[ -f scripts/nnunet_env.sh ]]; then
  source scripts/nnunet_env.sh
fi

: "${nnUNet_results:?nnUNet_results is not set. Export it or create scripts/nnunet_env.sh to set it.}"

# Constrain CPU threading / dataloader workers
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-1}"
export nnUNet_n_proc_val="${nnUNet_n_proc_val:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_NUM_THREADS="${PYTORCH_NUM_THREADS:-2}"

LOGS_DIR="logs/nnunet"
mkdir -p "$LOGS_DIR"

# Dataset ID -> label tag (matches PS1)
declare -A CONFIGS=(
  [905]="L5"
  [910]="L10"
  [920]="L20"
  [940]="L40"
)

summary=()

# Sort keys numerically
ids_sorted=$(printf "%s\n" "${!CONFIGS[@]}" | sort -n)

for id in $ids_sorted; do
  label="${CONFIGS[$id]}"
  datasetName="Dataset${id}_HVSMR_${label}"
  trainerDir="${nnUNet_results}/${datasetName}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  logPath="${LOGS_DIR}/nnunet_${label}_fold0_${timestamp}.log"

  echo ""
  echo "==== Training nnU-Net for Dataset ${id} (${label}) ===="

  if [[ -d "$trainerDir" ]]; then
    echo "Skipping ${id} (${label}), fold 0 already trained at ${trainerDir}"
    summary+=("Dataset ${id} (${label}): skipped (already exists)")
    continue
  fi

  echo "Starting training -> ${logPath}"

  # Run training. Use CUDA by default on RunPod; change to cpu if you really want.
  set +e
  nnUNetv2_train "$id" 3d_fullres 0 -device cuda 2>&1 | tee "$logPath"
  exitcode=${PIPESTATUS[0]}
  set -e

  if [[ "$exitcode" -ne 0 ]]; then
    echo "Training failed for ${id} (${label}) with exit code ${exitcode}. See ${logPath}"
    summary+=("Dataset ${id} (${label}): FAILED (exit ${exitcode})")
    continue
  fi

  summary+=("Dataset ${id} (${label}): trained this run")
done

echo ""
echo "==== Summary ===="
for s in "${summary[@]}"; do
  echo "$s"
done
