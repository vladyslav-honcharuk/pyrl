#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
TASK="tasks/gambling.py"
TRAIN="scripts/training/train.py"
PLOT="scripts/plotting/gambling.py"
DATA_ROOT="data_progress2"
RERUN_TRIALS=0
SKIP_TRIALS=0
PLOT_BEHAVIOR=0
TARGETS=()

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
    echo "Cleaning existing context/opto trial files due to --rerun-trials flag..."
    rm -f "$trials_dir"/trials_activity_ctx*.pkl
    rm -f "$trials_dir"/trials_activity_opto*.pkl
  fi
}

plot_behavior() {
  local group="$1"
  local suffix="$2"

  if [[ "$PLOT_BEHAVIOR" == "1" ]]; then
    plot_model "$group" "$suffix" "behavior"
  fi
}

plot_basic() {
  ensure_trials "basic_default" "_basic_default" 1
  plot_behavior "basic_default" "_basic_default"
  plot_model "basic_default" "_basic_default" "kappa-single" 0.0 1
}

plot_single_model() {
  local group="$1"
  local suffix="$2"

  ensure_trials "$group" "$suffix" 1
  plot_behavior "$group" "$suffix"
  plot_model "$group" "$suffix" "kappa-single" 0.0 1
}

plot_context_model() {
  local group="$1"
  local suffix="$2"

  clean_context_trials "$group" "$suffix"
  ensure_trials "$group" "$suffix" 1
  plot_behavior "$group" "$suffix"
  plot_model "$group" "$suffix" "context-sweep-0p2"
}

plot_runtime_dopamine_model() {
  local group="$1"
  local suffix="$2"

  clean_context_trials "$group" "$suffix"
  ensure_trials "$group" "$suffix" 1
  plot_behavior "$group" "$suffix"
  plot_model "$group" "$suffix" "opto-sweep"
}

plot_context_runtime_dopamine_model() {
  local group="$1"
  local suffix="$2"

  clean_context_trials "$group" "$suffix"
  ensure_trials "$group" "$suffix" 1
  plot_behavior "$group" "$suffix"
  plot_model "$group" "$suffix" "context-sweep-0p2"
  plot_model "$group" "$suffix" "opto-sweep"
}

plot_regular_context() {
  plot_context_model "regular_context" "_regular_context"
}

plot_context_d1d2() {
  plot_context_runtime_dopamine_model "context_d1d2" "_context_d1d2"
}

plot_d1d2_only() {
  plot_runtime_dopamine_model "d1d2_only" "_d1d2_only"
}

plot_hardwired_kappa() {
  local kappa="$1"
  local tag
  tag="$(kappa_tag "$kappa")"
  local group="hardwired_kappa_${kappa}"
  local suffix="_hardwired_kappa_${tag}"

  ensure_trials "$group" "$suffix" 2
  plot_behavior "$group" "$suffix"
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
}

plot_finetuned_kappa_mega() {
  plot_model "finetuned_kappa_0.0" "_ft_kappa_0p0" "finetuned-kappa-mega"
}

plot_hardwired_kappa_mega() {
  plot_model "hardwired_kappa_0.0" "_hardwired_kappa_0p0" "hardwired-kappa-mega"
}

plot_rpe_feedback() {
  plot_runtime_dopamine_model "phasic_rpe_d1d2" "_phasic_rpe_d1d2"
}

plot_rpe_feedback_legacy_name() {
  plot_runtime_dopamine_model "natural_rpe_feedback" "_rpe_feedback"
}

plot_phasic_rpe_d1d2_context() {
  plot_context_runtime_dopamine_model "phasic_rpe_d1d2_context" "_phasic_rpe_d1d2_context"
}

plot_tonic_vta_context() {
  plot_runtime_dopamine_model "tonic_vta_d1d2" "_tonic_vta_d1d2"
}

plot_tonic_vta_context_legacy_name() {
  plot_runtime_dopamine_model "tonic_vta_context" "_tonic_vta_context"
}

plot_tonic_vta_d1d2_context() {
  plot_context_runtime_dopamine_model "tonic_vta_d1d2_context" "_tonic_vta_d1d2_context"
}

plot_rpe_plus_vta_context() {
  plot_runtime_dopamine_model "phasic_rpe_vta_d1d2" "_phasic_rpe_vta_d1d2"
}

plot_rpe_plus_vta_context_legacy_name() {
  plot_runtime_dopamine_model "rpe_plus_vta_context" "_rpe_plus_vta_context"
}

plot_phasic_rpe_vta_d1d2_context() {
  plot_context_runtime_dopamine_model "phasic_rpe_vta_d1d2_context" "_phasic_rpe_vta_d1d2_context"
}

plot_all_model_variants() {
  plot_basic
  plot_regular_context
  plot_context_d1d2
  plot_d1d2_only
  plot_rpe_feedback
  plot_phasic_rpe_d1d2_context
  plot_tonic_vta_context
  plot_tonic_vta_d1d2_context
  plot_rpe_plus_vta_context
  plot_phasic_rpe_vta_d1d2_context
}

if (( ${#TARGETS[@]} > 0 )); then
  for target in "${TARGETS[@]}"; do
    fn="plot_${target}"
    if ! declare -F "$fn" >/dev/null; then
      echo "Unknown plot target: $target" >&2
      echo "Available examples:" >&2
      echo "  basic regular_context context_d1d2 d1d2_only" >&2
      echo "  rpe_feedback phasic_rpe_d1d2_context" >&2
      echo "  tonic_vta_context tonic_vta_d1d2_context" >&2
      echo "  rpe_plus_vta_context phasic_rpe_vta_d1d2_context" >&2
      echo "  all_model_variants hardwired_kappa_mega finetuned_kappa_mega" >&2
      exit 2
    fi
    "$fn"
  done
  exit 0
fi

# Pick the plots you want to run by uncommenting them.
# plot_basic
# plot_regular_context
plot_context_d1d2
# plot_d1d2_only

# for kappa in -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#   plot_finetuned_kappa "$kappa"
# done
# plot_finetuned_kappa_mega

# for kappa in -0.9 -0.5 0.0 0.5 0.9; do
#   plot_hardwired_kappa "$kappa"
# done
# plot_hardwired_kappa_mega

# plot_rpe_feedback
# plot_phasic_rpe_d1d2_context
# plot_tonic_vta_context
# plot_tonic_vta_d1d2_context
# plot_rpe_plus_vta_context
# plot_phasic_rpe_vta_d1d2_context
# plot_all_model_variants
