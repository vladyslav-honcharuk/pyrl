#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_SCRIPT="$ROOT_DIR/scripts/training/train.py"
PLOT_SCRIPT="$ROOT_DIR/scripts/plotting/gambling.py"

MODEL_FILE="${1:-tasks/gambling.py}"
TRIALS_PER_CONDITION="${2:-2}"
SUFFIX="_level2_dist_context_q"

BASE_CMD=(python3 "$TRAIN_SCRIPT" "$MODEL_FILE")
if [[ -n "$SUFFIX" ]]; then
  BASE_CMD+=(--suffix "$SUFFIX")
fi

echo "Training model for $MODEL_FILE"
# "${BASE_CMD[@]}" train --seed 52

echo "Running trials-a for $MODEL_FILE"
"${BASE_CMD[@]}" run "$PLOT_SCRIPT" trials-a "$TRIALS_PER_CONDITION" 

echo "Running trials-b for $MODEL_FILE"
"${BASE_CMD[@]}" run "$PLOT_SCRIPT" trials-b "$TRIALS_PER_CONDITION"

echo "Running behavior plot for $MODEL_FILE"
"${BASE_CMD[@]}" run "$PLOT_SCRIPT" behavior
