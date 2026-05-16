#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_SCRIPT="$ROOT_DIR/scripts/training/train.py"
PLOT_SCRIPT="$ROOT_DIR/scripts/plotting/gambling.py"

MODEL_FILE="${1:-tasks/gambling.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_PROGRESS_ROOT="${DATA_PROGRESS_ROOT:-data_progress}"

# Set RUN_TRAIN=0 to only regenerate plots from existing checkpoints.
RUN_TRAIN="${RUN_TRAIN:-1}"

# Plot actions to run after each trained checkpoint.
# opto-sweep includes behavior, value, activity, D1/D2 pull, logits, and probability-weighting mega plots.
# context-sweep-0p2 uses context values from -1 to +1 in 0.2 steps.
PLOT_ACTIONS_STR="${PLOT_ACTIONS:-opto-sweep context-sweep-0p2}"
read -r -a PLOT_ACTIONS <<< "$PLOT_ACTIONS_STR"

run_cmd() {
  echo
  echo "================================================================================"
  printf 'RUN:'
  printf ' %q' "$@"
  echo
  echo "================================================================================"
  "$@"
}

train_and_plot() {
  local label="$1"
  local suffix="$2"
  shift 2
  local extra_args=("$@")
  local data_root="$DATA_PROGRESS_ROOT/$label"

  echo
  echo "################################################################################"
  echo "# CONDITION: $label"
  echo "# SUFFIX:    $suffix"
  echo "# DATA ROOT: $data_root"
  echo "################################################################################"

  if [[ "$RUN_TRAIN" == "1" ]]; then
    run_cmd "$PYTHON_BIN" "$TRAIN_SCRIPT" "$MODEL_FILE" \
      --data-root "$data_root" \
      --suffix "$suffix" \
      "${extra_args[@]}" \
      train
  fi

  for action in "${PLOT_ACTIONS[@]}"; do
    run_cmd "$PYTHON_BIN" "$TRAIN_SCRIPT" "$MODEL_FILE" \
      --data-root "$data_root" \
      --suffix "$suffix" \
      "${extra_args[@]}" \
      run "$PLOT_SCRIPT" "$action"
  done
}

model_base_name() {
  basename "$MODEL_FILE" .py
}

kappa_tag() {
  printf '%+.1f' "$1" | sed 's/+//; s/-/neg/; s/\./p/'
}

model_savefile() {
  local label="$1"
  local suffix="$2"
  local name
  name="$(model_base_name)$suffix"
  printf '%s/%s/weights/%s/%s.pkl' "$ROOT_DIR" "$DATA_PROGRESS_ROOT/$label" "$name" "$name"
}

finetune_and_plot() {
  local label="$1"
  local suffix="$2"
  local pretrained="$3"
  shift 3
  local extra_args=("$@")
  local data_root="$DATA_PROGRESS_ROOT/$label"

  echo
  echo "################################################################################"
  echo "# CONDITION: $label"
  echo "# SUFFIX:    $suffix"
  echo "# PRETRAIN:  $pretrained"
  echo "# DATA ROOT: $data_root"
  echo "################################################################################"

  if [[ "$RUN_TRAIN" == "1" ]]; then
    run_cmd "$PYTHON_BIN" "$TRAIN_SCRIPT" "$MODEL_FILE" \
      --data-root "$data_root" \
      --suffix "$suffix" \
      --pretrained "$pretrained" \
      "${extra_args[@]}" \
      finetune
  fi

  for action in "${PLOT_ACTIONS[@]}"; do
    run_cmd "$PYTHON_BIN" "$TRAIN_SCRIPT" "$MODEL_FILE" \
      --data-root "$data_root" \
      --suffix "$suffix" \
      "${extra_args[@]}" \
      run "$PLOT_SCRIPT" "$action"
  done
}

# 1. Default/control model: no dopamine/RPE feedback, no kappa, no tonic VTA context.
# train_and_plot \
#   "default_no_dopamine_feedback" \
#   "_suite_default" \
#   --no-rpe-modulation

# 2. Hard-wired kappa models: fixed positive/negative value-learning asymmetry.
# Override with: KAPPAS="-1.0 -0.5 0.0 0.5 1.0" ./scripts/training/run_gambling_condition_suite.sh
KAPPAS_STR="${KAPPAS:--1.0 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0}"
read -r -a KAPPAS <<< "$KAPPAS_STR"
for kappa in "${KAPPAS[@]}"; do
  kappa_tag="$(kappa_tag "$kappa")"
  train_and_plot \
    "hard_wired_kappa_${kappa}" \
    "_suite_kappa_${kappa_tag}" \
    --no-rpe-modulation \
    --kappa "$kappa"
done

# 3. Finetuned hard-wired kappa ladder: train κ=0 once, then finetune
# 0 → +0.2 → +0.4 ... and 0 → -0.2 → -0.4 ...
KAPPA_FINETUNE_POSITIVE_STR="${KAPPA_FINETUNE_POSITIVE:-0.2 0.4 0.6 0.8 1.0}"
KAPPA_FINETUNE_NEGATIVE_STR="${KAPPA_FINETUNE_NEGATIVE:--0.2 -0.4 -0.6 -0.8 -1.0}"
read -r -a KAPPA_FINETUNE_POSITIVE <<< "$KAPPA_FINETUNE_POSITIVE_STR"
read -r -a KAPPA_FINETUNE_NEGATIVE <<< "$KAPPA_FINETUNE_NEGATIVE_STR"

kappa0_label="finetuned_kappa_0.0"
kappa0_suffix="_suite_ft_kappa_0p0"
train_and_plot \
  "$kappa0_label" \
  "$kappa0_suffix" \
  --no-rpe-modulation \
  --kappa 0.0

kappa0_savefile="$(model_savefile "$kappa0_label" "$kappa0_suffix")"
previous_savefile="$kappa0_savefile"
for kappa in "${KAPPA_FINETUNE_POSITIVE[@]}"; do
  tag="$(kappa_tag "$kappa")"
  label="finetuned_kappa_${kappa}"
  suffix="_suite_ft_kappa_${tag}"
  finetune_and_plot \
    "$label" \
    "$suffix" \
    "$previous_savefile" \
    --no-rpe-modulation \
    --kappa "$kappa"
  previous_savefile="$(model_savefile "$label" "$suffix")"
done

previous_savefile="$kappa0_savefile"
for kappa in "${KAPPA_FINETUNE_NEGATIVE[@]}"; do
  tag="$(kappa_tag "$kappa")"
  label="finetuned_kappa_${kappa}"
  suffix="_suite_ft_kappa_${tag}"
  finetune_and_plot \
    "$label" \
    "$suffix" \
    "$previous_savefile" \
    --no-rpe-modulation \
    --kappa "$kappa"
  previous_savefile="$(model_savefile "$label" "$suffix")"
done

# 4. Tonic/context dopamine: trial-constant VTA context is sampled during training,
# but natural RPE gain is zero so this isolates tonic dopamine context.
train_and_plot \
  "tonic_context_dopamine" \
  "_suite_tonic_vta_ctx" \
  --rpe-modulation \
  --rpe-modulation-gain 0.0 \
  --rpe-modulation-clamp 0.9 \
  --vta-training-context \
  --vta-context-distribution uniform \
  --vta-context-low -0.9 \
  --vta-context-high 0.9 \
  --dopamine-modulation-mode linear \
  --dopamine-sensitivity-learned \
  --dopamine-bias \
  --dopamine-bias-max-abs 0.7

# 5. Base model with natural RPE feedback: no extra tonic context, only natural RPE-derived dopamine.
train_and_plot \
  "natural_rpe_feedback" \
  "_suite_rpe_feedback" \
  --rpe-modulation \
  --rpe-modulation-gain 3.0 \
  --rpe-modulation-clamp 0.9 \
  --dopamine-modulation-mode linear \
  --dopamine-sensitivity-learned \
  --dopamine-bias \
  --dopamine-bias-max-abs 0.7

# 6. RPE model trained with phasic dopamine range: natural RPE plus sampled tonic VTA context,
# with linear activity and linear learning asymmetry.
train_and_plot \
  "rpe_plus_phasic_dopamine_range" \
  "_suite_rpe_phasic_range" \
  --rpe-modulation \
  --rpe-modulation-gain 3.0 \
  --rpe-modulation-clamp 0.9 \
  --vta-training-context \
  --vta-context-distribution gaussian \
  --vta-context-mean 0.3 \
  --vta-context-std 0.5 \
  --dopamine-modulation-mode linear \
  --dopamine-learning-modulation-mode linear \
  --dopamine-sensitivity-learned \
  --dopamine-bias \
  --dopamine-bias-max-abs 0.7

echo
echo "All requested gambling condition runs finished."
