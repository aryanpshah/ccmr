#!/bin/bash
set -euo pipefail

# Run from repo root (adjust if you keep the script in scripts/)
cd "$(dirname "$0")/.."

export nnUNet_raw=$PWD/data/nnunet/nnUNet_raw
export nnUNet_preprocessed=$PWD/data/nnunet/nnUNet_preprocessed
export nnUNet_results=$PWD/data/nnunet/nnUNet_results

for id in 905 910 920 940; do
  case $id in
    905) label=L5 ;;
    910) label=L10 ;;
    920) label=L20 ;;
    940) label=L40 ;;
  esac

  fold_dir="$nnUNet_results/Dataset${id}_HVSMR_${label}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"
  echo "Extracting metrics for $label at $fold_dir"

  if [[ -d "$fold_dir" ]]; then
    python scripts/extract_nnunet_metrics.py "$fold_dir"
  else
    echo "Skip $label: missing $fold_dir" >&2
  fi
done
