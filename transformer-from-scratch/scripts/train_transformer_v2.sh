#!/bin/bash
# Launch only after choosing and authorizing a resource budget.
set -euo pipefail
cd "$(dirname "$0")/.."
R=runs/transformer-v2-preparation
if test "$#" -lt 4; then
  echo "Usage: $0 ROW_COUNT RUN_NAME cpu|mps|cuda EPOCHS [--resume PATH or --dry-run]" >&2
  exit 2
fi
size=$1 name=$2 device=$3 epochs=$4
shift 4
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec .venv/bin/python scripts/train.py --data-config "$R/data.yaml" \
  --model-config configs/transformer_v2_model.yaml --config configs/transformer_v2_training.yaml \
  --tokenizer-dir "$R/tokenizer" --processed-dir "$R/prepared-$size" --run-name "$name" \
  --device "$device" --epochs "$epochs" "$@"
