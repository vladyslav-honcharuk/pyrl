"""
Default configuration parameters for recurrent actor-critic training.
"""
import numpy as np

required = ['inputs', 'actions', 'tmax', 'n_gradient', 'n_validation']

default = {
    'Performance':           None,
    'N':                     100,
    'p0':                    1,
    'baseline_N':            100,
    'baseline_p0':           1,
    'lr':                    0.001,
    'baseline_lr':           0.005,
    'lr_decay':              0.0005,    # Learning rate decay factor (0 = no decay, 0.001 = slow decay)
    'baseline_lr_decay':     0.0005,  # Baseline LR decay (mimics synaptic consolidation)
    'max_iter':              4500,
    'fix':                   [],
    'baseline_fix':          [],
    'target_reward':         np.inf,
    'mode':                  'episodic',
    'advantage_mode':        'mc',   # 'mc' (Monte Carlo returns) or 'td' (TD(0) bootstrapping)
    'network_type':          'gru',
    'baseline_network_type': 'gru',
    'R_ABORTED':             -1,
    'R_TERMINAL':            None,
    'abort_on_last_t':       True,
    'checkfreq':             50,
    'dt':                    10,
    'tau':                   100,
    'tau_reward':            np.inf,
    'var_rec':               0.01,
    'baseline_var_rec':      0.01,
    'L2_r':                  0,
    'baseline_L2_r':         0,
    'activity_balance':      0,
    'baseline_activity_balance': 0.3,
    'Win':                   1,
    'baseline_Win':          1,
    'bout':                  0,
    'baseline_bout':         0,
    'Win_mask':              None,
    'baseline_Win_mask':     None,
    'rho':                   2,
    'kappa':                 0,  # Learning-asymmetry signal, clipped to [-0.9, 0.9]
    'baseline_rho':          2,
    'L1_Wrec':               0,
    'L2_Wrec':               1e-5,
    'L2_Wout':               1e-4,     # L2 regularization for MLP output layers (Critic) - OFF
    'baseline_grad_clip':    None, # Gradient clipping for baseline network - OFF
    'policy_dropout':        0.2,
    'policy_seed':           1,
    'baseline_seed':         2,

    'context_decision_only':          False,  # Apply context input only during decision period

    # Baseline input composition
    'baseline_include_state':         True,  # If True, include task inputs U in baseline network inputs

    # Training-time context sampling for the external context signal c
    'context_distribution':           'gaussian',  # 'uniform' or 'gaussian'
    'context_uniform_low':            -1.0,
    'context_uniform_high':           1.0,
    'context_gaussian_mean':          0.0,
    'context_gaussian_std':           0.8,
    'training_context_input':         False,

    # ========== RPE-Based D1/D2 Modulation ==========
    # Use continuous RPE signal from value network to modulate D1/D2 receptor occupancy
    'use_rpe_modulation':             False,  # Enable RPE-based D1/D2 modulation during cue phase
    'rpe_modulation_gain':            3.0,    # Gain factor for RPE → dopamine mapping (default: 1.0)
    'rpe_modulation_clamp':           0.9,    # Clamp RPE signal to [-clamp, +clamp] before modulation

    # Trial-level VTA dopamine context sampled during training.
    # This is added to the natural RPE-derived dopamine signal and applied through
    # D1/D2 gain modulation, not through the sensory context input channel.
    'vta_training_context':           False,
    'vta_context_distribution':       'uniform',
    'vta_context_low':                -0.9,
    'vta_context_high':               0.9,
    'vta_context_mean':               0.0,
    'vta_context_std':                0.3,
    'vta_context_weight':             1.5,

    # Per-neuron dopamine receptor sensitivity. When enabled, dopamine remains
    # push-pull but each MSN has a different gain magnitude.
    'dopamine_heterogeneous_sensitivity': False,
    'dopamine_sensitivity_min':       0.3,
    'dopamine_sensitivity_max':       1.0,
    'dopamine_sensitivity_learned':   False,
    'dopamine_bias_enabled':          False,
    'dopamine_bias_learned':          True,
    'dopamine_bias_init':             0.0,
    'dopamine_bias_max_abs':          0.7,
    'dopamine_modulation_mode':       'linear',  # 'linear' or 'hill'
    'dopamine_hill_base_da':          1.0,
    'dopamine_hill_da_range':         1.0,
    'dopamine_hill_ec50_d1':          1.0,
    'dopamine_hill_ec50_d2':          0.07,
    'dopamine_hill_coefficient':      1.0,
    'dopamine_hill_gain_scale':       2.0,
    'dopamine_learning_modulation_mode': 'linear',  # 'linear' or 'hill'
    'dopamine_learning_eta_min':      0.1,
    'dopamine_learning_eta_max':      1.9,

    # Optional three-factor/Hebbian-style actor update. When enabled, the
    # policy output-weight gradient is multiplied by current weight strength,
    # so learning depends on pre/activity, dopamine/error, and actor weight.
    'actor_weight_learning_modulation': False,
    'actor_weight_learning_floor':    0.05,
    'actor_weight_learning_max':      2.0,
    'actor_weight_learning_normalize': True,
    'positive_policy_readout':        False,

    # ========== Optogenetic VTA Stimulation (Inference Only) ==========
    # Simulate optogenetic manipulation of dopamine neurons during inference
    'opto_stim_offset':               0.0,    # Constant offset added to RPE (+ve = more DA, -ve = less DA)
    'opto_stim_gain':                 1.0,    # Multiplicative gain on RPE (>1 = amplify, <1 = attenuate)
    'opto_stim_phase':                'all',  # When to apply: 'all', 'cue', 'decision', 'fixation'
}
