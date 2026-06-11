#!/usr/bin/env bash

# Shared condition definitions for gambling training and plotting.

canonical_condition() {
  case "$1" in
    basic|tonic|tonic_d1d2|tonic_d1d2_plasticity|hidden_tonic_d1d2_plasticity|d1d2|d1d2_plasticity|d1d2_v|d1d2_recent_rpe|d1d2_recent_rpe_cuedec|d1d2_plasticity_opal01|d1d2_plasticity_opal04|d1d2_plasticity_opal04_fake_vta|d1d2_plasticity_opal04_rpe_natural|d1d2_plasticity_symmetric_ctx|d1d2_plasticity_d2_only_stim|d1d2_plasticity_d1_only_stim|d1d2_plasticity_d2_only_suppress|d1d2_plasticity_d1_only_suppress|vta_phasic_natural_d1d2|tonic_vta_phasic_natural_d1d2|vta_offset_d1d2|tonic_vta_offset_d1d2|vta_phasic_natural_vta_offset_d1d2|tonic_vta_phasic_natural_vta_offset_d1d2)
      printf '%s\n' "$1"
      ;;
    regular_context)
      printf 'tonic\n'
      ;;
    context_d1d2)
      printf 'tonic_d1d2\n'
      ;;
    hidden_tonic_d1d2)
      printf 'hidden_tonic_d1d2_plasticity\n'
      ;;
    d1d2_only)
      printf 'd1d2\n'
      ;;
    d1d2_plasticity_soft_opal|d1d2_plasticity_0p1)
      printf 'd1d2_plasticity_opal01\n'
      ;;
    d1d2_plasticity_0p4)
      printf 'd1d2_plasticity_opal04\n'
      ;;
    d1d2_plasticity_0p4_fake_vta|d1d2_plasticity_opal04_vta_fake)
      printf 'd1d2_plasticity_opal04_fake_vta\n'
      ;;
    d1d2_plasticity_0p4_rpe|d1d2_plasticity_opal04_natural_rpe)
      printf 'd1d2_plasticity_opal04_rpe_natural\n'
      ;;
    rpe_feedback|rpe_feedback_legacy_name)
      printf 'vta_phasic_natural_d1d2\n'
      ;;
    phasic_rpe_d1d2_context)
      printf 'tonic_vta_phasic_natural_d1d2\n'
      ;;
    tonic_vta_context|tonic_vta_context_legacy_name)
      printf 'vta_offset_d1d2\n'
      ;;
    tonic_vta_d1d2_context)
      printf 'tonic_vta_offset_d1d2\n'
      ;;
    rpe_plus_vta_context|rpe_plus_vta_context_legacy_name)
      printf 'vta_phasic_natural_vta_offset_d1d2\n'
      ;;
    phasic_rpe_vta_d1d2_context|full_model|main_model)
      printf 'tonic_vta_phasic_natural_vta_offset_d1d2\n'
      ;;
    *)
      return 1
      ;;
  esac
}

condition_exists() {
  local canonical
  canonical="$(canonical_condition "$1")" || return 1
  case "$canonical" in
    basic|tonic|tonic_d1d2|tonic_d1d2_plasticity|hidden_tonic_d1d2_plasticity|d1d2|d1d2_plasticity|d1d2_v|d1d2_recent_rpe|d1d2_recent_rpe_cuedec|d1d2_plasticity_opal01|d1d2_plasticity_opal04|d1d2_plasticity_opal04_fake_vta|d1d2_plasticity_opal04_rpe_natural|d1d2_plasticity_symmetric_ctx|d1d2_plasticity_d2_only_stim|d1d2_plasticity_d1_only_stim|d1d2_plasticity_d2_only_suppress|d1d2_plasticity_d1_only_suppress|vta_phasic_natural_d1d2|tonic_vta_phasic_natural_d1d2|vta_offset_d1d2|tonic_vta_offset_d1d2|vta_phasic_natural_vta_offset_d1d2|tonic_vta_phasic_natural_vta_offset_d1d2)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

condition_group_suffix() {
  local canonical
  canonical="$(canonical_condition "$1")" || return 1

  CONDITION_GROUP="$canonical"
  CONDITION_SUFFIX="_${canonical}"
  CONDITION_PLOT_MODE="single"
  CONDITION_TRAINABLE=1
  CONDITION_LOAD_GROUP="$CONDITION_GROUP"
  CONDITION_LOAD_SUFFIX="$CONDITION_SUFFIX"
  CONDITION_SWEEP_MODE="symmetric"

  case "$canonical" in
    basic)
      CONDITION_GROUP="basic"
      CONDITION_SUFFIX="_basic"
      ;;
    d1d2)
      CONDITION_GROUP="d1d2"
      CONDITION_SUFFIX="_d1d2"
      ;;
    d1d2_plasticity)
      CONDITION_GROUP="d1d2_plasticity"
      CONDITION_SUFFIX="_d1d2_plasticity_pos_reg"
      CONDITION_PLOT_MODE="context"
      ;;
    d1d2_v)
      CONDITION_GROUP="d1d2_v"
      CONDITION_SUFFIX="_d1d2_v"
      CONDITION_PLOT_MODE="context"
      ;;
    d1d2_recent_rpe)
      CONDITION_GROUP="d1d2_recent_rpe_g01"
      CONDITION_SUFFIX="_d1d2_recent_rpe_g01"
      ;;
    d1d2_recent_rpe_cuedec)
      CONDITION_GROUP="d1d2_recent_rpe_g01_cuedec"
      CONDITION_SUFFIX="_d1d2_recent_rpe_g01_cuedec"
      ;;
    d1d2_plasticity_opal01)
      CONDITION_GROUP="d1d2_plasticity_opal01"
      CONDITION_SUFFIX="_d1d2_plasticity_pos_reg_opal01"
      CONDITION_PLOT_MODE="context"
      ;;
    d1d2_plasticity_opal04)
      CONDITION_GROUP="d1d2_plasticity_opal04"
      CONDITION_SUFFIX="_d1d2_plasticity_pos_reg_opal04"
      CONDITION_PLOT_MODE="context"
      ;;
    d1d2_plasticity_opal04_fake_vta)
      CONDITION_GROUP="d1d2_plasticity_opal04_fake_vta"
      CONDITION_SUFFIX="_d1d2_plasticity_pos_reg_opal04_fake_vta"
      CONDITION_PLOT_MODE="opto_zero_rpe"
      CONDITION_TRAINABLE=0
      CONDITION_LOAD_GROUP="d1d2_plasticity_opal04"
      CONDITION_LOAD_SUFFIX="_d1d2_plasticity_pos_reg_opal04"
      ;;
    d1d2_plasticity_opal04_rpe_natural)
      CONDITION_GROUP="d1d2_plasticity_opal04_rpe_natural"
      CONDITION_SUFFIX="_d1d2_plasticity_pos_reg_opal04_rpe_natural"
      CONDITION_PLOT_MODE="opto"
      ;;
    d1d2_plasticity_symmetric_ctx)
      CONDITION_GROUP="d1d2_plasticity_symmetric_ctx"
      CONDITION_SUFFIX="_d1d2_plasticity_symmetric_ctx"
      CONDITION_PLOT_MODE="pathway_gain"
      CONDITION_TRAINABLE=0
      CONDITION_LOAD_GROUP="d1d2_plasticity_opal04"
      CONDITION_LOAD_SUFFIX="_d1d2_plasticity_pos_reg_opal04"
      CONDITION_SWEEP_MODE="symmetric"
      ;;
    d1d2_plasticity_d2_only_stim)
      CONDITION_GROUP="d1d2_plasticity_d2_only_stim"
      CONDITION_SUFFIX="_d1d2_plasticity_d2_only_stim"
      CONDITION_PLOT_MODE="pathway_gain"
      CONDITION_TRAINABLE=0
      CONDITION_LOAD_GROUP="d1d2_plasticity_opal04"
      CONDITION_LOAD_SUFFIX="_d1d2_plasticity_pos_reg_opal04"
      CONDITION_SWEEP_MODE="d2_only_stim"
      ;;
    d1d2_plasticity_d1_only_stim)
      CONDITION_GROUP="d1d2_plasticity_d1_only_stim"
      CONDITION_SUFFIX="_d1d2_plasticity_d1_only_stim"
      CONDITION_PLOT_MODE="pathway_gain"
      CONDITION_TRAINABLE=0
      CONDITION_LOAD_GROUP="d1d2_plasticity_opal04"
      CONDITION_LOAD_SUFFIX="_d1d2_plasticity_pos_reg_opal04"
      CONDITION_SWEEP_MODE="d1_only_stim"
      ;;
    d1d2_plasticity_d2_only_suppress)
      CONDITION_GROUP="d1d2_plasticity_d2_only_suppress"
      CONDITION_SUFFIX="_d1d2_plasticity_d2_only_suppress"
      CONDITION_PLOT_MODE="pathway_gain"
      CONDITION_TRAINABLE=0
      CONDITION_LOAD_GROUP="d1d2_plasticity_opal04"
      CONDITION_LOAD_SUFFIX="_d1d2_plasticity_pos_reg_opal04"
      CONDITION_SWEEP_MODE="d2_only_suppress"
      ;;
    d1d2_plasticity_d1_only_suppress)
      CONDITION_GROUP="d1d2_plasticity_d1_only_suppress"
      CONDITION_SUFFIX="_d1d2_plasticity_d1_only_suppress"
      CONDITION_PLOT_MODE="pathway_gain"
      CONDITION_TRAINABLE=0
      CONDITION_LOAD_GROUP="d1d2_plasticity_opal04"
      CONDITION_LOAD_SUFFIX="_d1d2_plasticity_pos_reg_opal04"
      CONDITION_SWEEP_MODE="d1_only_suppress"
      ;;
    tonic)
      CONDITION_GROUP="tonic"
      CONDITION_SUFFIX="_tonic"
      CONDITION_PLOT_MODE="context"
      ;;
    tonic_d1d2)
      CONDITION_GROUP="tonic_d1d2"
      CONDITION_SUFFIX="_tonic_d1d2"
      CONDITION_PLOT_MODE="context"
      ;;
    tonic_d1d2_plasticity)
      CONDITION_GROUP="tonic_d1d2_plasticity"
      CONDITION_SUFFIX="_tonic_d1d2_plasticity"
      CONDITION_PLOT_MODE="context"
      ;;
    hidden_tonic_d1d2_plasticity)
      CONDITION_GROUP="hidden_tonic_d1d2_plasticity"
      CONDITION_SUFFIX="_hidden_tonic_d1d2_plasticity"
      CONDITION_PLOT_MODE="context"
      ;;
    vta_phasic_natural_d1d2)
      CONDITION_GROUP="vta_phasic_natural_d1d2"
      CONDITION_SUFFIX="_vta_phasic_natural_d1d2"
      ;;
    tonic_vta_phasic_natural_d1d2)
      CONDITION_GROUP="tonic_vta_phasic_natural_d1d2"
      CONDITION_SUFFIX="_tonic_vta_phasic_natural_d1d2"
      CONDITION_PLOT_MODE="context"
      ;;
    vta_offset_d1d2)
      CONDITION_GROUP="vta_offset_d1d2"
      CONDITION_SUFFIX="_vta_offset_d1d2"
      ;;
    tonic_vta_offset_d1d2)
      CONDITION_GROUP="tonic_vta_offset_d1d2"
      CONDITION_SUFFIX="_tonic_vta_offset_d1d2"
      CONDITION_PLOT_MODE="context"
      ;;
    vta_phasic_natural_vta_offset_d1d2)
      CONDITION_GROUP="vta_phasic_natural_vta_offset_d1d2"
      CONDITION_SUFFIX="_vta_phasic_natural_vta_offset_d1d2"
      ;;
    tonic_vta_phasic_natural_vta_offset_d1d2)
      CONDITION_GROUP="tonic_vta_phasic_natural_vta_offset_d1d2"
      CONDITION_SUFFIX="_tonic_vta_phasic_natural_vta_offset_d1d2"
      CONDITION_PLOT_MODE="context"
      ;;
  esac
}

condition_train_args() {
  local canonical
  canonical="$(canonical_condition "$1")" || return 1

  CONDITION_ARGS=()

  local context_args=(
    --training-context-input
    --context-distribution uniform
    --context-uniform-low -1.0
    --context-uniform-high 1.0
  )
  local d1d2_args=(
    --opponent-modulation
    --dopamine-sensitivity-learned
    --dopamine-bias
  )
  local plasticity_args=(
    --positive-policy-readout
    --pathway-specific-plasticity
    --actor-weight-learning-modulation
    --positive-readout-weight-l2 1e-3
    --opponent-pull-l2 1e-4
    --dopamine-sensitivity-learned
    --dopamine-bias
  )
  local rpe_natural_args=(
    --rpe-modulation
    --rpe-modulation-gain 3.0
    --rpe-modulation-clamp 0.9
  )
  local rpe_zero_args=(
    --rpe-modulation
    --rpe-modulation-gain 0.0
    --rpe-modulation-clamp 0.9
  )
  local vta_context_args=(
    --vta-training-context
    --vta-context-distribution uniform
    --vta-context-low -0.9
    --vta-context-high 0.9
  )

  case "$canonical" in
    basic)
      ;;
    tonic)
      CONDITION_ARGS+=("${context_args[@]}")
      ;;
    tonic_d1d2)
      CONDITION_ARGS+=("${context_args[@]}" "${d1d2_args[@]}")
      ;;
    tonic_d1d2_plasticity)
      CONDITION_ARGS+=("${context_args[@]}")
      CONDITION_ARGS+=(
        --opponent-modulation
        --positive-policy-readout
        --pathway-specific-plasticity
        --actor-weight-learning-modulation
        --positive-readout-weight-l2 1e-3
        --opponent-pull-l2 1e-4
        --dopamine-sensitivity-learned
        --dopamine-bias
      )
      ;;
    hidden_tonic_d1d2_plasticity)
      CONDITION_ARGS+=("${rpe_zero_args[@]}" "${vta_context_args[@]}")
      CONDITION_ARGS+=(
        --opponent-modulation
        --positive-policy-readout
        --exclude-control-action-from-dopamine-modulation
        --pathway-specific-plasticity
        --opal-d1-negative-scale 0.1
        --opal-d2-positive-scale 0.1
        --actor-weight-learning-modulation
        --positive-readout-weight-l2 1e-3
        --opponent-pull-l2 1e-4
      )
      ;;
    d1d2)
      CONDITION_ARGS+=("${d1d2_args[@]}")
      ;;
    d1d2_plasticity)
      CONDITION_ARGS+=(
        --opponent-modulation
        --positive-policy-readout
        --pathway-specific-plasticity
        --actor-weight-learning-modulation
        --positive-readout-weight-l2 1e-3
        --opponent-pull-l2 1e-4
        --dopamine-sensitivity-learned
        --dopamine-bias
      )
      ;;
    d1d2_v)
      CONDITION_ARGS+=(
        --opponent-modulation
        --positive-policy-readout
        --pathway-specific-plasticity
        --opal-d1-negative-scale 0.4
        --opal-d2-positive-scale 0.4
        --policy-value-feedback
      )
      ;;
    d1d2_plasticity_opal01)
      CONDITION_ARGS+=(
        --opponent-modulation
        --positive-policy-readout
        --pathway-specific-plasticity
        --opal-d1-negative-scale 0.1
        --opal-d2-positive-scale 0.1
        --actor-weight-learning-modulation
        --positive-readout-weight-l2 1e-3
        --opponent-pull-l2 1e-4
        --dopamine-sensitivity-learned
        --dopamine-bias
      )
      ;;
    d1d2_plasticity_opal04)
      CONDITION_ARGS+=(
        --opponent-modulation
        --positive-policy-readout
        --pathway-specific-plasticity
        --opal-d1-negative-scale 0.4
        --opal-d2-positive-scale 0.4
      )
      ;;
    d1d2_plasticity_opal04_rpe_natural)
      CONDITION_ARGS+=("${rpe_natural_args[@]}")
      CONDITION_ARGS+=(
        --opponent-modulation
        --positive-policy-readout
        --pathway-specific-plasticity
        --opal-d1-negative-scale 0.4
        --opal-d2-positive-scale 0.4
      )
      ;;
    vta_phasic_natural_d1d2)
      CONDITION_ARGS+=("${rpe_natural_args[@]}" "${d1d2_args[@]}")
      ;;
    tonic_vta_phasic_natural_d1d2)
      CONDITION_ARGS+=("${context_args[@]}" "${rpe_natural_args[@]}" "${d1d2_args[@]}")
      ;;
    vta_offset_d1d2)
      CONDITION_ARGS+=("${rpe_zero_args[@]}" "${vta_context_args[@]}" "${d1d2_args[@]}")
      ;;
    tonic_vta_offset_d1d2)
      CONDITION_ARGS+=("${context_args[@]}" "${rpe_zero_args[@]}" "${vta_context_args[@]}" "${d1d2_args[@]}")
      ;;
    vta_phasic_natural_vta_offset_d1d2)
      CONDITION_ARGS+=("${rpe_natural_args[@]}" "${vta_context_args[@]}" "${d1d2_args[@]}")
      ;;
    tonic_vta_phasic_natural_vta_offset_d1d2)
      CONDITION_ARGS+=("${context_args[@]}" "${rpe_natural_args[@]}" "${vta_context_args[@]}" "${d1d2_args[@]}")
      ;;
    *)
      return 1
      ;;
  esac
}
