"""Mapping from the three headline models + dopamine baseline to real pyrl config.

The simulation exposes three models that mirror the project's main experimental
conditions, plus a single "baseline dopamine" knob in [-1, 1]:

  * basic            - plain actor-critic, no D1/D2 opponent machinery.
  * d1d2_04          - OpAL-style D1/D2 opponent plasticity with alpha_d1=alpha_d2=0.4.
  * d1d2_04_phasic   - the above, plus between-trial phasic dopamine
                       (leaky previous-trial RPE biasing the next trial).

Baseline dopamine maps to the documented risk knob:
  * Lower (negative) tonic DA  -> risk-averse  (kappa < 0)
  * Higher (positive) tonic DA -> risk-seeking (kappa > 0)

``kappa`` is baked in at training time. For trained models we additionally
expose ``opto_stim_offset`` so the user can nudge dopamine at inference time
and watch risk preference shift without retraining.
"""

PRESETS = {
    'basic': {
        'label': 'Basic actor-critic',
        'train_flags': [],
        'description': 'Plain policy-gradient RNN. Dopamine baseline sets the '
                       'learning-asymmetry (kappa) that makes the agent risk-averse or risk-seeking.',
    },
    'd1d2_04': {
        'label': 'D1/D2 (alpha 0.4)',
        'train_flags': [
            '--opponent-modulation',
            '--positive-policy-readout',
            '--pathway-specific-plasticity',
            '--opal-alpha-d1', '0.4',
            '--opal-alpha-d2', '0.4',
        ],
        'description': 'OpAL-style opponent Go/NoGo (D1/D2) pathways with asymmetric '
                       'plasticity (alpha=0.4). Tonic dopamine tips the D1/D2 balance.',
    },
    'd1d2_04_phasic': {
        'label': 'D1/D2 (0.4) + phasic DA',
        'train_flags': [
            '--opponent-modulation',
            '--positive-policy-readout',
            '--pathway-specific-plasticity',
            '--opal-alpha-d1', '0.4',
            '--opal-alpha-d2', '0.4',
            '--use-recent-rpe-modulation',
            '--recent-rpe-gain', '0.5',
            '--recent-rpe-decay', '0.6',
        ],
        'description': 'D1/D2 opponent model plus between-trial phasic dopamine: the '
                       'previous trial\'s reward-prediction error leaks forward and biases '
                       'the next choice.',
    },
}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def kappa_for_dopamine(dopamine):
    """Map baseline dopamine in [-1, 1] to kappa in [-0.9, 0.9]."""
    return round(clamp(float(dopamine) * 0.9, -0.9, 0.9), 3)


def train_command(python, train_script, task_spec, preset, dopamine, data_root,
                  suffix, seed, device, max_iter=None):
    """Build the argv list for a live training run."""
    if preset not in PRESETS:
        raise ValueError(f'Unknown preset: {preset}')

    # argparse fills the positional `action` from the first non-option token after
    # the model file, so `train` must come before the optional flags.
    argv = [
        python, train_script, task_spec, 'train',
        '--data-root', data_root,
        '--suffix', suffix,
        '--seed', str(seed),
        '--device', device,
        '--kappa', str(kappa_for_dopamine(dopamine)),
    ]
    argv += PRESETS[preset]['train_flags']
    return argv


def inference_overrides(preset, inference_dopamine):
    """Config overrides applied to a trained model at inference time.

    ``inference_dopamine`` in [-1, 1] is converted to an optogenetic-style
    constant offset on the dopamine/RPE signal, letting the user shift risk
    preference live. Positive = more dopamine.
    """
    offset = round(clamp(float(inference_dopamine), -1.0, 1.0) * 0.9, 3)
    overrides = {
        'opto_stim_offset': offset,
        'opto_stim_phase': 'all',
    }
    if preset in ('d1d2_04', 'd1d2_04_phasic'):
        # These models route dopamine through D1/D2 gain; the opto offset feeds
        # the same pathway, so the live knob is meaningful.
        overrides['recent_rpe_stim_offset'] = offset
    return overrides
