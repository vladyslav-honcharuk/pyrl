#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
TASK="tasks/gambling.py"
TRAIN="scripts/training/train.py"
DATA_ROOT="data_progress2"

kappa_tag() {
  printf '%+.1f' "$1" | sed 's/+//; s/-/neg/; s/\./p/'
}

model_file() {
  local group="$1"
  local suffix="$2"
  local name="gambling${suffix}"
  printf '%s/%s/weights/%s/%s.pkl' "$DATA_ROOT" "$group" "$name" "$name"
}

train_model() {
  local group="$1"
  local suffix="$2"
  shift 2

  "$PYTHON" "$TRAIN" "$TASK" \
    --data-root "$DATA_ROOT/$group" \
    --suffix "$suffix" \
    "$@" \
    train
}

finetune_model() {
  local group="$1"
  local suffix="$2"
  local pretrained="$3"
  shift 3

  "$PYTHON" "$TRAIN" "$TASK" \
    --data-root "$DATA_ROOT/$group" \
    --suffix "$suffix" \
    --pretrained "$pretrained" \
    "$@" \
    finetune
}

train_basic() {
  train_model "basic_default" "_basic_default"
}

train_regular_context() {
  train_model "regular_context" "_regular_context" \
    --training-context-input \
    --context-distribution uniform \
    --context-uniform-low -1.0 \
    --context-uniform-high 1.0
}

train_context_d1d2() {
  train_model "context_d1d2" "_context_d1d2" \
    --training-context-input \
    --context-distribution uniform \
    --context-uniform-low -1.0 \
    --context-uniform-high 1.0 \
    --opponent-modulation \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_d1d2_only() {
  train_model "d1d2_only" "_d1d2_only" \
    --opponent-modulation \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_hardwired_kappa() {
  local kappa="$1"
  local tag
  tag="$(kappa_tag "$kappa")"

  train_model "hardwired_kappa_${kappa}" "_hardwired_kappa_${tag}" \
    --kappa "$kappa"
}

train_rpe_feedback() {
  train_model "phasic_rpe_d1d2" "_phasic_rpe_d1d2" \
    --rpe-modulation \
    --rpe-modulation-gain 3.0 \
    --rpe-modulation-clamp 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_rpe_feedback_legacy_name() {
  train_model "natural_rpe_feedback" "_rpe_feedback" \
    --rpe-modulation \
    --rpe-modulation-gain 3.0 \
    --rpe-modulation-clamp 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_phasic_rpe_d1d2_context() {
  train_model "phasic_rpe_d1d2_context" "_phasic_rpe_d1d2_context" \
    --training-context-input \
    --context-distribution uniform \
    --context-uniform-low -1.0 \
    --context-uniform-high 1.0 \
    --rpe-modulation \
    --rpe-modulation-gain 3.0 \
    --rpe-modulation-clamp 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_tonic_vta_context() {
  train_model "tonic_vta_d1d2" "_tonic_vta_d1d2" \
    --rpe-modulation \
    --rpe-modulation-gain 0.0 \
    --rpe-modulation-clamp 0.9 \
    --vta-training-context \
    --vta-context-distribution uniform \
    --vta-context-low -0.9 \
    --vta-context-high 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_tonic_vta_context_legacy_name() {
  train_model "tonic_vta_context" "_tonic_vta_context" \
    --rpe-modulation \
    --rpe-modulation-gain 0.0 \
    --rpe-modulation-clamp 0.9 \
    --vta-training-context \
    --vta-context-distribution uniform \
    --vta-context-low -0.9 \
    --vta-context-high 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_tonic_vta_d1d2_context() {
  train_model "tonic_vta_d1d2_context" "_tonic_vta_d1d2_context" \
    --training-context-input \
    --context-distribution uniform \
    --context-uniform-low -1.0 \
    --context-uniform-high 1.0 \
    --rpe-modulation \
    --rpe-modulation-gain 0.0 \
    --rpe-modulation-clamp 0.9 \
    --vta-training-context \
    --vta-context-distribution uniform \
    --vta-context-low -0.9 \
    --vta-context-high 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_rpe_plus_vta_context() {
  train_model "phasic_rpe_vta_d1d2" "_phasic_rpe_vta_d1d2" \
    --rpe-modulation \
    --rpe-modulation-gain 3.0 \
    --rpe-modulation-clamp 0.9 \
    --vta-training-context \
    --vta-context-distribution uniform \
    --vta-context-low -0.9 \
    --vta-context-high 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_rpe_plus_vta_context_legacy_name() {
  train_model "rpe_plus_vta_context" "_rpe_plus_vta_context" \
    --rpe-modulation \
    --rpe-modulation-gain 3.0 \
    --rpe-modulation-clamp 0.9 \
    --vta-training-context \
    --vta-context-distribution uniform \
    --vta-context-low -0.9 \
    --vta-context-high 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_phasic_rpe_vta_d1d2_context() {
  train_model "phasic_rpe_vta_d1d2_context" "_phasic_rpe_vta_d1d2_context" \
    --training-context-input \
    --context-distribution uniform \
    --context-uniform-low -1.0 \
    --context-uniform-high 1.0 \
    --rpe-modulation \
    --rpe-modulation-gain 3.0 \
    --rpe-modulation-clamp 0.9 \
    --vta-training-context \
    --vta-context-distribution uniform \
    --vta-context-low -0.9 \
    --vta-context-high 0.9 \
    --dopamine-sensitivity-learned \
    --dopamine-bias
}

train_finetuned_kappa_chain() {
  train_model "finetuned_kappa_0.0" "_ft_kappa_0p0" --kappa 0.0

  local base
  base="$(model_file "finetuned_kappa_0.0" "_ft_kappa_0p0")"

  train_finetuned_kappa_positive_chain "$base" &
  local pos_pid=$!

  train_finetuned_kappa_negative_chain "$base" &
  local neg_pid=$!

  wait "$pos_pid"
  wait "$neg_pid"
}

train_finetuned_kappa_positive_chain() {
  local base="$1"
  local previous="$base"
  for kappa in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    local tag
    tag="$(kappa_tag "$kappa")"
    finetune_model "finetuned_kappa_${kappa}" "_ft_kappa_${tag}" "$previous" \
      --kappa "$kappa" \
      --finetune-iter 500 \
      --finetune-policy-lr 0.0003 \
      --finetune-baseline-lr 0.00003
    previous="$(model_file "finetuned_kappa_${kappa}" "_ft_kappa_${tag}")"
  done
}

train_finetuned_kappa_negative_chain() {
  local base="$1"
  local previous="$base"
  for kappa in -0.1 -0.2 -0.3 -0.4 -0.5 -0.6 -0.7 -0.8 -0.9; do
    local tag
    tag="$(kappa_tag "$kappa")"
    finetune_model "finetuned_kappa_${kappa}" "_ft_kappa_${tag}" "$previous" \
      --kappa "$kappa" \
      --finetune-iter 500 \
      --finetune-policy-lr 0.0003 \
      --finetune-baseline-lr 0.00003
    previous="$(model_file "finetuned_kappa_${kappa}" "_ft_kappa_${tag}")"
  done
}

# Pick the steps you want to run by uncommenting them.
# train_basic
# train_regular_context
# train_context_d1d2
train_d1d2_only

# for kappa in -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#   train_hardwired_kappa "$kappa"
# done

# train_finetuned_kappa_chain
train_rpe_feedback
train_phasic_rpe_d1d2_context
train_tonic_vta_context
train_tonic_vta_d1d2_context
train_rpe_plus_vta_context
train_phasic_rpe_vta_d1d2_context
