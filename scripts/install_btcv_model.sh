#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: bash scripts/install_btcv_model.sh <path_to_zip>"
  exit 1
fi

ZIP_PATH="$1"

if [ ! -f "$ZIP_PATH" ]; then
  echo "Error: File not found at $ZIP_PATH"
  exit 1
fi

echo "Unzipping BTCV Swin-UNETR model..."
mkdir -p pretrained/btcv
unzip "$ZIP_PATH" -d pretrained/btcv

echo "Searching for model.pt inside pretrained/btcv..."
FOUND_MODEL=$(find pretrained/btcv -name "model.pt" | head -n 1)

if [ -z "$FOUND_MODEL" ]; then
  echo "ERROR: model.pt not found in extracted files."
  exit 1
fi

echo "Located checkpoint:"
echo "$FOUND_MODEL"

echo "Model installed successfully."
echo
echo "Use this path for --pretrained_ckpt:"
echo "$FOUND_MODEL"
