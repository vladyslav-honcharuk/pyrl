"""
Default configuration parameters for policy gradient training.
"""
import numpy as np

required = ['inputs', 'actions', 'tmax', 'n_gradient', 'n_validation']

default = {
    'Performance':           None,
    'N':                     100,
    'p0':                    1,
    'baseline_N':            100,
    'baseline_p0':           1,
    'lr':                    0.0005,
    'baseline_lr':           0.0005,
    'max_iter':              1000,
    'fix':                   [],
    'baseline_fix':          [],
    'target_reward':         np.inf,
    'mode':                  'episodic',
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
    'Win':                   1,
    'baseline_Win':          1,
    'bout':                  0,
    'baseline_bout':         0,
    'Win_mask':              None,
    'baseline_Win_mask':     None,
    'rho':                   2,
    'kappa':                 0,  # Risk-sensitivity parameter: -1 (risk-averse) to +1 (risk-seeking)
    'baseline_rho':          2,
    'L1_Wrec':               0,
    'L2_Wrec':               1e-5,
    'policy_seed':           1,
    'baseline_seed':         2,

    # ========== Distributional RL Settings ==========
    # Enable these flags to use distributional critic with quantile regression
    # All default to False for backward compatibility with existing models

    'use_distributional_critic':      False,  # Enable 5-quantile distributional critic (instead of single V(s))
    'n_quantiles':                    5,      # Number of quantiles for distributional critic
    'quantile_huber_kappa':           1.0,    # Huber loss threshold for quantile regression

    # Context-based modulation (requires distributional critic for quantile selection)
    'use_context_quantile_selection': False,  # Let context signal select which quantile to use for advantage
    'use_context_temperature':        False,  # Let context signal modulate softmax temperature
    'temperature_base':               1.0,    # Base temperature for softmax (1.0 = standard softmax)
    'temperature_context_scale':      0.5,    # How much context affects temperature (0 = no effect)

    # Context input to baseline network (for future extensions)
    'context_to_baseline':            False,  # Add context as explicit input to baseline network
}
