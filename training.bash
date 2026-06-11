#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
TASK="tasks/gambling.py"
TRAIN="scripts/training/train.py"
DATA_ROOT="data_progression3"
CONDITION_DEFS="scripts/gambling_condition_defs.bash"

# shellcheck source=/dev/null
source "$CONDITION_DEFS"

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
    --device "$TRAIN_DEVICE" \
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
    --device "$TRAIN_DEVICE" \
    "$@" \
    finetune
}

# Naming guide:
# - tonic: direct task context input through the normal sensory/input channel.
# - hidden_tonic: sampled dopamine context during training without a sensory CONTEXT input.
# - vta_phasic_natural: natural critic-derived RPE drives D1/D2 dopamine modulation.
# - vta_offset: sampled tonic VTA dopamine offset; RPE gain is set to 0.
# - d1d2: opponent policy readout, D1 contribution minus D2 contribution.

train_condition() {
  local target="$1"

  if ! condition_exists "$target"; then
    echo "Unknown training condition: $target" >&2
    return 2
  fi

  condition_group_suffix "$target"
  if [[ "${CONDITION_TRAINABLE:-1}" != "1" ]]; then
    echo "Condition is plot-only and has no training definition: $target" >&2
    return 2
  fi
  condition_train_args "$target"
  train_model "$CONDITION_GROUP" "$CONDITION_SUFFIX" "${CONDITION_ARGS[@]}"
}

train_hardwired_kappa() {
  local kappa="$1"
  local tag
  tag="$(kappa_tag "$kappa")"

  train_model "hardwired_kappa_${kappa}" "_hardwired_kappa_${tag}" \
    --kappa "$kappa"
}

train_d1d2_v() {
  train_model "d1d2_v" "_d1d2_v" \
    --opponent-modulation \
    --positive-policy-readout \
    --pathway-specific-plasticity \
    --opal-d1-negative-scale 0.4 \
    --opal-d2-positive-scale 0.4 \
    --policy-value-feedback
}

train_d1d2_vpop() {
  train_model "d1d2_vpop" "_d1d2_vpop" \
    --opponent-modulation \
    --positive-policy-readout \
    --pathway-specific-plasticity \
    --opal-d1-negative-scale 0.4 \
    --opal-d2-positive-scale 0.4 \
    --policy-value-population-feedback
}

train_d1d2_vmod() {
  train_model "d1d2_vmod" "_d1d2_vmod" \
    --opponent-modulation \
    --positive-policy-readout \
    --pathway-specific-plasticity \
    --opal-d1-negative-scale 0.4 \
    --opal-d2-positive-scale 0.4 \
    --use-value-modulation
}

train_d1d2_vmod_shared() {
  train_model "d1d2_vmod_shared" "_d1d2_vmod_shared" \
    --opponent-modulation \
    --positive-policy-readout \
    --pathway-specific-plasticity \
    --opal-d1-negative-scale 0.4 \
    --opal-d2-positive-scale 0.4 \
    --use-value-modulation \
    --use-value-modulation-shared-gain
}

train_d1d2_recent_rpe() {
  train_model "d1d2_recent_rpe_g01" "_d1d2_recent_rpe_g01" \
    --opponent-modulation \
    --positive-policy-readout \
    --pathway-specific-plasticity \
    --opal-d1-negative-scale 0.4 \
    --opal-d2-positive-scale 0.4 \
    --use-recent-rpe-modulation \
    --recent-rpe-decay 0.7 \
    --recent-rpe-gain 0.1 \
    --recent-rpe-phase decision
}

train_d1d2_recent_rpe_cuedec() {
  train_model "d1d2_recent_rpe_g01_cuedec" "_d1d2_recent_rpe_g01_cuedec" \
    --opponent-modulation \
    --positive-policy-readout \
    --pathway-specific-plasticity \
    --opal-d1-negative-scale 0.4 \
    --opal-d2-positive-scale 0.4 \
    --use-recent-rpe-modulation \
    --recent-rpe-decay 0.7 \
    --recent-rpe-gain 0.1 \
    --recent-rpe-phase cue_decision
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
# train_condition basic
# train_condition d1d2
# train_condition d1d2_plasticity
# train_condition d1d2_plasticity_opal04
# train_d1d2_v
# train_d1d2_vmod
# train_d1d2_vmod_shared
# train_d1d2_vpop
# train_d1d2_recent_rpe
train_d1d2_recent_rpe_cuedec
# train_condition d1d2_plasticity_opal04_rpe_natural
# train_condition tonic
# train_condition tonic_d1d2
# train_condition tonic_d1d2_plasticity
# train_condition hidden_tonic_d1d2_plasticity
# train_condition vta_phasic_natural_d1d2

# for kappa in -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#   train_hardwired_kappa "$kappa"
# done

# train_finetuned_kappa_chain

# train_condition tonic_vta_phasic_natural_d1d2
# train_condition vta_offset_d1d2
# train_condition tonic_vta_offset_d1d2
# train_condition vta_phasic_natural_vta_offset_d1d2
# train_condition tonic_vta_phasic_natural_vta_offset_d1d2
