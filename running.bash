#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
TASK="tasks/gambling.py"
TRAIN="scripts/training/train.py"
PLOT="scripts/plotting/gambling.py"
DATA_ROOT="data_progress2"

run_trials() {
  local group="$1"
  local suffix="$2"
  local trials_per_condition="${3:-1}"

  "$PYTHON" "$TRAIN" "$TASK" \
    --data-root "$DATA_ROOT/$group" \
    --suffix "$suffix" \
    run "$PLOT" trials-a "$trials_per_condition"
}

plot_behavior() {
  local group="$1"
  local suffix="$2"

  "$PYTHON" "$TRAIN" "$TASK" \
    --data-root "$DATA_ROOT/$group" \
    --suffix "$suffix" \
    run "$PLOT" behavior
}

# Basic activity data and behavior plot.
run_trials "basic_default" "_basic_default" 1
plot_behavior "basic_default" "_basic_default"
