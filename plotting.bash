#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
TASK="tasks/gambling.py"
TRAIN="scripts/training/train.py"
PLOT="scripts/plotting/gambling.py"
DATA_ROOT="data_progression3"
CONDITION_DEFS="scripts/gambling_condition_defs.bash"
RERUN_TRIALS=0
SKIP_TRIALS=0
PLOT_BEHAVIOR=0
TARGETS=()

# shellcheck source=/dev/null
source "$CONDITION_DEFS"

for arg in "$@"; do
  case "$arg" in
    --rerun-trials)
      RERUN_TRIALS=1
      ;;
    --skip-trials)
      SKIP_TRIALS=1
      ;;
    --behavior)
      PLOT_BEHAVIOR=1
      ;;
    *)
      TARGETS+=("$arg")
      ;;
  esac
done

kappa_tag() {
  printf '%+.1f' "$1" | sed 's/+//; s/-/neg/; s/\./p/'
}

trial_activity_file() {
  local group="$1"
  local suffix="$2"
  local name="gambling${suffix}"
  printf '%s/%s/trials/%s/trials_activity.pkl' "$DATA_ROOT" "$group" "$name"
}

model_file() {
  local group="$1"
  local suffix="$2"
  local name="gambling${suffix}"
  printf '%s/%s/weights/%s/%s.pkl' "$DATA_ROOT" "$group" "$name" "$name"
}

plot_model() {
  local group="$1"
  local suffix="$2"
  local action="$3"
  shift 3

  "$PYTHON" "$TRAIN" "$TASK" \
    --data-root "$DATA_ROOT/$group" \
    --suffix "$suffix" \
    run "$PLOT" "$action" "$@"
}

plot_model_from_source() {
  local group="$1"
  local suffix="$2"
  local load_group="$3"
  local load_suffix="$4"
  local action="$5"
  shift 5

  "$PYTHON" "$TRAIN" "$TASK" \
    --data-root "$DATA_ROOT/$group" \
    --suffix "$suffix" \
    --load-savefile "$(model_file "$load_group" "$load_suffix")" \
    run "$PLOT" "$action" "$@"
}

run_trials() {
  local group="$1"
  local suffix="$2"
  local trials_per_condition="${3:-1}"

  plot_model "$group" "$suffix" "trials-a" "$trials_per_condition"
}

ensure_trials() {
  local group="$1"
  local suffix="$2"
  local trials_per_condition="${3:-1}"
  local trialsfile
  trialsfile="$(trial_activity_file "$group" "$suffix")"

  if [[ "$SKIP_TRIALS" == "1" ]]; then
    echo "Skipping trials generation (--skip-trials flag set)"
  elif [[ "$RERUN_TRIALS" == "1" || ! -f "$trialsfile" ]]; then
    run_trials "$group" "$suffix" "$trials_per_condition"
  else
    echo "Using existing trials: $trialsfile"
  fi
}

clean_context_trials() {
  local group="$1"
  local suffix="$2"
  local name="gambling${suffix}"
  local trials_dir="$DATA_ROOT/$group/trials/$name"

  if [[ "$RERUN_TRIALS" == "1" && -d "$trials_dir" ]]; then
    echo "Cleaning existing sweep trial files due to --rerun-trials flag..."
    rm -f "$trials_dir"/trials_activity_ctx*.pkl
    rm -f "$trials_dir"/trials_activity_opto*.pkl
    rm -f "$trials_dir"/trials_activity_vfb*.pkl
    rm -f "$trials_dir"/trials_activity_vpopstim_*.pkl
    rm -f "$trials_dir"/trials_activity_rrpe*.pkl
  fi
}

plot_behavior() {
  local group="$1"
  local suffix="$2"

  if [[ "$PLOT_BEHAVIOR" == "1" ]]; then
    plot_model "$group" "$suffix" "behavior"
  fi
}

plot_proportion_chosen() {
  local group="$1"
  local suffix="$2"

  plot_model "$group" "$suffix" "proportion-chosen"
}

plot_choice_probability_curves() {
  local group="$1"
  local suffix="$2"

  plot_model "$group" "$suffix" "choice-probability-curves"
}

plot_pathway_ev_asymmetry() {
  local group="$1"
  local suffix="$2"

  ensure_trials "$group" "$suffix" 1
  plot_model "$group" "$suffix" "pathway-ev-asymmetry"
}

# Naming guide mirrors training.bash:
# - tonic: direct task context input through the normal sensory/input channel.
# - hidden_tonic: sampled dopamine context during training without a sensory CONTEXT input.
# - vta_phasic_natural: natural critic-derived RPE drives D1/D2 dopamine modulation.
# - vta_offset: sampled tonic VTA dopamine offset; RPE gain is set to 0.
# - vta_phasic/opto: artificial VTA output stimulation during plotting/inference.

plot_single_model() {
  local group="$1"
  local suffix="$2"

  ensure_trials "$group" "$suffix" 1
  plot_behavior "$group" "$suffix"
  plot_proportion_chosen "$group" "$suffix"
  plot_choice_probability_curves "$group" "$suffix"
  plot_model "$group" "$suffix" "kappa-single" 0.0 1
}

plot_context_only_model() {
  local group="$1"
  local suffix="$2"

  clean_context_trials "$group" "$suffix"
  ensure_trials "$group" "$suffix" 1
  plot_behavior "$group" "$suffix"
  plot_proportion_chosen "$group" "$suffix"
  plot_choice_probability_curves "$group" "$suffix"
  plot_model "$group" "$suffix" "context-sweep-0p1"
}

plot_context_only_model_from_source() {
  local group="$1"
  local suffix="$2"
  local load_group="$3"
  local load_suffix="$4"
  local action="${5:-context-sweep-0p1}"

  clean_context_trials "$group" "$suffix"
  plot_model_from_source "$group" "$suffix" "$load_group" "$load_suffix" "$action"
}

plot_opto_model() {
  local group="$1"
  local suffix="$2"

  clean_context_trials "$group" "$suffix"
  plot_model "$group" "$suffix" "opto-sweep-0p1"
}

plot_opto_zero_rpe_model_from_source() {
  local group="$1"
  local suffix="$2"
  local load_group="$3"
  local load_suffix="$4"

  clean_context_trials "$group" "$suffix"
  plot_model_from_source "$group" "$suffix" "$load_group" "$load_suffix" "opto-sweep-0p1-zero-rpe"
}

plot_pathway_gain_model() {
  local group="$1"
  local suffix="$2"
  local load_group="$3"
  local load_suffix="$4"
  local sweep_mode="$5"

  clean_context_trials "$group" "$suffix"
  plot_model_from_source "$group" "$suffix" "$load_group" "$load_suffix" "pathway-gain-sweep" "$sweep_mode"
}

plot_condition() {
  local target="$1"
  local canonical

  if ! condition_exists "$target"; then
    echo "Unknown plotting condition: $target" >&2
    return 2
  fi

  canonical="$(canonical_condition "$target")"
  if [[ "$canonical" == "d1d2_v" ]]; then
    plot_d1d2_v
    return
  fi

  condition_group_suffix "$target"
  case "$CONDITION_PLOT_MODE" in
    context)
      plot_context_only_model "$CONDITION_GROUP" "$CONDITION_SUFFIX"
      ;;
    opto)
      plot_opto_model "$CONDITION_GROUP" "$CONDITION_SUFFIX"
      ;;
    opto_zero_rpe)
      plot_opto_zero_rpe_model_from_source \
        "$CONDITION_GROUP" \
        "$CONDITION_SUFFIX" \
        "$CONDITION_LOAD_GROUP" \
        "$CONDITION_LOAD_SUFFIX"
      ;;
    pathway_gain)
      plot_pathway_gain_model \
        "$CONDITION_GROUP" \
        "$CONDITION_SUFFIX" \
        "$CONDITION_LOAD_GROUP" \
        "$CONDITION_LOAD_SUFFIX" \
        "$CONDITION_SWEEP_MODE"
      ;;
    *)
      plot_single_model "$CONDITION_GROUP" "$CONDITION_SUFFIX"
      ;;
  esac
}

plot_condition_pathway_ev_asymmetry() {
  local target="$1"

  if ! condition_exists "$target"; then
    echo "Unknown pathway-EV condition: $target" >&2
    return 2
  fi

  condition_group_suffix "$target"
  plot_pathway_ev_asymmetry "$CONDITION_GROUP" "$CONDITION_SUFFIX"
}

plot_d1d2_v() {
  plot_context_only_model "d1d2_v" "_d1d2_v"
  plot_context_only_model_from_source \
    "d1d2_v_no_v" \
    "_d1d2_v_no_v" \
    "d1d2_v" \
    "_d1d2_v" \
    "context-sweep-0p1-no-v"
}

plot_d1d2_v_no_v() {
  plot_context_only_model_from_source \
    "d1d2_v_no_v" \
    "_d1d2_v_no_v" \
    "d1d2_v" \
    "_d1d2_v" \
    "context-sweep-0p1-no-v"
}

plot_d1d2_v_feedback() {
  clean_context_trials "d1d2_v_feedback" "_d1d2_v_feedback"
  plot_model_from_source \
    "d1d2_v_feedback" \
    "_d1d2_v_feedback" \
    "d1d2_v" \
    "_d1d2_v" \
    "value-feedback-sweep"
}

plot_d1d2_recent_rpe() {
  plot_single_model "d1d2_recent_rpe_g01" "_d1d2_recent_rpe_g01"
  plot_model "d1d2_recent_rpe_g01" "_d1d2_recent_rpe_g01" "recent-rpe-sequential"
}

plot_d1d2_recent_rpe_stim() {
  clean_context_trials "d1d2_recent_rpe_g01_stim" "_d1d2_recent_rpe_g01_stim"
  plot_model_from_source \
    "d1d2_recent_rpe_g01_stim" \
    "_d1d2_recent_rpe_g01_stim" \
    "d1d2_recent_rpe_g01" \
    "_d1d2_recent_rpe_g01" \
    "recent-rpe-sweep"
}

plot_d1d2_recent_rpe_cuedec() {
  plot_single_model "d1d2_recent_rpe_g01_cuedec" "_d1d2_recent_rpe_g01_cuedec"
  plot_model "d1d2_recent_rpe_g01_cuedec" "_d1d2_recent_rpe_g01_cuedec" "recent-rpe-sequential"
}

plot_d1d2_recent_rpe_cuedec_stim() {
  clean_context_trials "d1d2_recent_rpe_g01_cuedec_stim" "_d1d2_recent_rpe_g01_cuedec_stim"
  plot_model_from_source \
    "d1d2_recent_rpe_g01_cuedec_stim" \
    "_d1d2_recent_rpe_g01_cuedec_stim" \
    "d1d2_recent_rpe_g01_cuedec" \
    "_d1d2_recent_rpe_g01_cuedec" \
    "recent-rpe-sweep"
}

plot_d1d2_vpop_stim() {
  clean_context_trials "d1d2_vpop_stim" "_d1d2_vpop_stim"
  plot_model_from_source \
    "d1d2_vpop_stim" \
    "_d1d2_vpop_stim" \
    "d1d2_vpop" \
    "_d1d2_vpop" \
    "value-population-stim-sweep"
}

plot_hardwired_kappa() {
  local kappa="$1"
  local tag
  tag="$(kappa_tag "$kappa")"
  local group="hardwired_kappa_${kappa}"
  local suffix="_hardwired_kappa_${tag}"

  ensure_trials "$group" "$suffix" 2
  plot_behavior "$group" "$suffix"
  plot_proportion_chosen "$group" "$suffix"
  plot_choice_probability_curves "$group" "$suffix"
  plot_model "$group" "$suffix" "kappa-single" "$kappa" 2
}

plot_finetuned_kappa() {
  local kappa="$1"
  local tag
  tag="$(kappa_tag "$kappa")"
  local group="finetuned_kappa_${kappa}"
  local suffix="_ft_kappa_${tag}"

  ensure_trials "$group" "$suffix" 2
  plot_behavior "$group" "$suffix"
  plot_proportion_chosen "$group" "$suffix"
  plot_choice_probability_curves "$group" "$suffix"
}

plot_finetuned_kappa_mega() {
  plot_model "finetuned_kappa_0.0" "_ft_kappa_0p0" "finetuned-kappa-mega"
}

plot_hardwired_kappa_mega() {
  plot_model "hardwired_kappa_0.0" "_hardwired_kappa_0p0" "hardwired-kappa-mega"
}

plot_all_model_variants() {
  plot_condition basic
  plot_condition tonic
  plot_condition tonic_d1d2
  plot_condition d1d2
  plot_condition vta_phasic_natural_d1d2
  plot_condition tonic_vta_phasic_natural_d1d2
  plot_condition vta_offset_d1d2
  plot_condition tonic_vta_offset_d1d2
  plot_condition vta_phasic_natural_vta_offset_d1d2
  plot_condition tonic_vta_phasic_natural_vta_offset_d1d2
}

if (( ${#TARGETS[@]} > 0 )); then
  for target in "${TARGETS[@]}"; do
    if condition_exists "$target"; then
      plot_condition "$target"
      continue
    fi
    if [[ "$target" == *_pathway_ev_asymmetry ]]; then
      base_target="${target%_pathway_ev_asymmetry}"
      if condition_exists "$base_target"; then
        plot_condition_pathway_ev_asymmetry "$base_target"
        continue
      fi
    fi
    fn="plot_${target}"
    if ! declare -F "$fn" >/dev/null; then
      echo "Unknown plot target: $target" >&2
      echo "Available examples:" >&2
      echo "  basic tonic tonic_d1d2 d1d2 d1d2_plasticity d1d2_v d1d2_v_no_v d1d2_v_feedback d1d2_vpop_stim d1d2_recent_rpe d1d2_recent_rpe_stim d1d2_recent_rpe_cuedec d1d2_recent_rpe_cuedec_stim" >&2
      echo "  d1d2_plasticity_symmetric_ctx d1d2_plasticity_d2_only_stim d1d2_plasticity_d1_only_stim" >&2
      echo "  d1d2_plasticity_d2_only_suppress d1d2_plasticity_d1_only_suppress" >&2
      echo "  hidden_tonic_d1d2_plasticity hidden_tonic_d1d2_plasticity_pathway_ev_asymmetry" >&2
      echo "  vta_phasic_natural_d1d2 tonic_vta_phasic_natural_d1d2" >&2
      echo "  vta_offset_d1d2 tonic_vta_offset_d1d2" >&2
      echo "  vta_phasic_natural_vta_offset_d1d2 tonic_vta_phasic_natural_vta_offset_d1d2" >&2
      echo "  all_model_variants hardwired_kappa_mega finetuned_kappa_mega" >&2
      exit 2
    fi
    "$fn"
  done
  exit 0
fi

# Pick the plots you want to run by uncommenting them.
# plot_condition basic

# plot_condition d1d2
# plot_condition d1d2_plasticity
# plot_d1d2_v
# plot_condition tonic

# plot_condition vta_phasic_natural_d1d2
# plot_condition tonic_d1d2
# plot_condition tonic_d1d2_plasticity

# for kappa in -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#   plot_finetuned_kappa "$kappa"
# done
# plot_finetuned_kappa_mega

# for kappa in -0.9 -0.5 0.0 0.5 0.9; do
#   plot_hardwired_kappa "$kappa"
# done
# plot_hardwired_kappa_mega


# plot_condition tonic_vta_phasic_natural_d1d2
# plot_condition vta_offset_d1d2
# plot_condition tonic_vta_offset_d1d2
# plot_condition vta_phasic_natural_vta_offset_d1d2
# plot_condition tonic_vta_phasic_natural_vta_offset_d1d2
# plot_all_model_variants
