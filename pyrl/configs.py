"""
Default configuration parameters for policy gradient training.
"""
import numpy as np

required = ['inputs', 'actions', 'tmax', 'n_gradient', 'n_validation']

default = {
    'Performance':           None,
    'N':                     100,
    'p0':                    0.1,
    'baseline_N':            100,
    'baseline_p0':           1,
    'lr':                    0.002,
    'baseline_lr':           0.02,
    'max_iter':              3000,
    'fix':                   [],
    'baseline_fix':          [],
    'target_reward':         np.inf,
    'mode':                  'episodic',
    'network_type':          'gru',
    'baseline_network_type': 'gru',
    'R_ABORTED':             -0.5,
    'R_TERMINAL':            -0.5,
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
    'baseline_bout':         0.7,
    'Win_mask':              None,
    'baseline_Win_mask':     None,
    'rho':                   2,
    'kappa':                 0,  # Risk-sensitivity parameter: -1 (risk-averse) to +1 (risk-seeking)
    'baseline_rho':          2,
    'L1_Wrec':               0,
    'L2_Wrec':               0,
    'policy_seed':           1,
    'baseline_seed':         2,
    'grad_clip':             5,  # Gradient clipping threshold (None = no clipping)
    'baseline_grad_clip':    5,  # Baseline gradient clipping threshold (None = no clipping)
    'entropy_cost':          0,  # Entropy regularization coefficient (higher = more exploration)
    'advantage_clip':        None  # Clip advantages to [-clip, +clip] (None = no clipping)
}
