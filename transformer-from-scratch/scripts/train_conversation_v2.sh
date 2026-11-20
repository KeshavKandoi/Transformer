#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
P="${TRANSFORMER_EXPERIMENT:-experiments/interview-conversation-v2}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
exec .venv/bin/python scripts/train.py --data-config "$P/data.yaml" --processed-dir "$P/processed" \
  --tokenizer-dir "$P/tokenizer" --model-config configs/interview_model.yaml --config configs/interview_training.yaml \
  --checkpoint-dir "$P/checkpoints" --runs-dir "$P/training" --run-name scratch "$@"
