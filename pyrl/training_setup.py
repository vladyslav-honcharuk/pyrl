"""Construction and configuration helpers for recurrent actor-critic trainers."""
import numpy as np
import torch
import torch.nn as nn

from . import utils
from .networks import Networks


def resolve_device(device=None):
    """Resolve the requested torch device using the legacy default behavior."""
    if device is None:
        return torch.device('mps' if torch.mps.is_available() else 'cpu')
    return torch.device(device)


class SetupMixin:
    def _load_from_file(self, savefile, dt, load):
        """Load model from saved file."""
        save = utils.load(savefile)
        self.save = save
        self.config = save['config']

        # Time step
        self.dt = dt if dt is not None else self.config['dt']

        # Leak
        alpha = self.dt / self.config['tau']

        # Which parameters to load?
        params_p = save['best_policy_params'] if load == 'best' else save['current_policy_params']
        params_b = save['best_baseline_params'] if load == 'best' else save['current_baseline_params']

        # Masks
        masks_p = save.get('policy_masks', {})
        masks_b = save.get('baseline_masks', {})

        # Policy network
        self.policy_config = save['policy_config']
        self.policy_config['alpha'] = alpha
        for key in (
            'dopamine_modulation_mode',
            'dopamine_hill_base_da',
            'dopamine_hill_da_range',
            'dopamine_hill_ec50_d1',
            'dopamine_hill_ec50_d2',
            'dopamine_hill_coefficient',
            'dopamine_hill_gain_scale',
            'dopamine_bias_max_abs',
        ):
            if key in self.config:
                self.policy_config[key] = self.config[key]

        Network = Networks[self.config['network_type']]
        self.policy_net = Network(self.policy_config, params=params_p,
                                  masks=masks_p, name='policy')

        # Baseline network
        self.baseline_config = save['baseline_config']
        self.baseline_config['alpha'] = alpha

        Network = Networks[self.config.get('baseline_network_type',
                                          self.config['network_type'])]
        self.baseline_net = Network(self.baseline_config, params=params_b,
                                    masks=masks_b, name='baseline')

        # Store loaded kappa configuration (will be used in _setup_kappa)
        self.loaded_kappa_mode = save.get('kappa_mode', 'single')
        self.loaded_kappa_dist = save.get('kappa_dist', None)
        self.loaded_kappa_dist_params = save.get('kappa_dist_params', None)
        self.loaded_kappa_neurons = save.get('kappa_neurons', None)
        self.loaded_kappa = save.get('kappa', self.config.get('kappa', 0.0))  # Fallback to config

        baseline_nout = self.baseline_config.get('Nout', 1)
        if baseline_nout > 1:
            raise ValueError(
                "Distributional critic checkpoints are no longer supported. "
                f"Loaded baseline has Nout={baseline_nout}; expected Nout=1."
            )

    def _create_new_model(self, config, dt, seed):
        """Create new model from config."""
        self.config = config

        # Time step
        self.dt = dt if dt is not None else config['dt']

        # Leak
        alpha = self.dt / config['tau']

        # Policy network configuration
        K = config['p0'] * config['N']
        self.policy_config = {
            'Nin': config['Nin'],
            'N': config['N'],
            'Nout': config['Nout'],
            'p0': config['p0'],
            'rho': config['rho'],
            'f_out': 'softmax',
            'Win': config['Win'] * np.sqrt(K) / config['Nin'],
            'Win_mask': config['Win_mask'],
            'bout': config['bout'],
            'fix': config['fix'],
            'L2_r': config['L2_r'],
            'activity_balance': config.get('activity_balance', 0),
            'L1_Wrec': config['L1_Wrec'],
            'L2_Wrec': config['L2_Wrec'],
            'alpha': alpha,
            'dopamine_heterogeneous_sensitivity': config.get('dopamine_heterogeneous_sensitivity', False),
            'dopamine_sensitivity_min': config.get('dopamine_sensitivity_min', 0.3),
            'dopamine_sensitivity_max': config.get('dopamine_sensitivity_max', 1.0),
            'dopamine_sensitivity_learned': config.get('dopamine_sensitivity_learned', False),
            'dopamine_sensitivity_seed': config.get('policy_seed', seed) + 1000,
            'dopamine_bias_enabled': config.get('dopamine_bias_enabled', False),
            'dopamine_bias_learned': config.get('dopamine_bias_learned', True),
            'dopamine_bias_init': config.get('dopamine_bias_init', 0.0),
            'dopamine_bias_max_abs': config.get('dopamine_bias_max_abs', 0.3),
            'dopamine_modulation_mode': config.get('dopamine_modulation_mode', 'linear'),
            'dopamine_hill_base_da': config.get('dopamine_hill_base_da', 1.0),
            'dopamine_hill_da_range': config.get('dopamine_hill_da_range', 1.0),
            'dopamine_hill_ec50_d1': config.get('dopamine_hill_ec50_d1', 1.0),
            'dopamine_hill_ec50_d2': config.get('dopamine_hill_ec50_d2', 0.07),
            'dopamine_hill_coefficient': config.get('dopamine_hill_coefficient', 1.0),
            'dopamine_hill_gain_scale': config.get('dopamine_hill_gain_scale', 2.0),
        }

        Network = Networks[config['network_type']]
        self.policy_net = Network(self.policy_config, seed=config['policy_seed'], name='policy')

        # Baseline network configuration
        K = config['baseline_p0'] * config['N']
        baseline_Nin = self.policy_net.N + len(config['actions'])
        if config.get('baseline_include_state', False):
            baseline_Nin += config['Nin']

        self.baseline_config = {
            'Nin': baseline_Nin,
            'N': config['baseline_N'],
            'Nout': 1,
            'p0': config['baseline_p0'],
            'rho': config['baseline_rho'],
            'f_out': 'linear',
            'Win': config['baseline_Win'] * np.sqrt(K) / baseline_Nin,
            'Win_mask': config['baseline_Win_mask'],
            'bout': config['baseline_bout'] if config['baseline_bout'] is not None else 0.0,
            'x0': config.get('baseline_x0', 0),  # Higher initial state to prevent dead neurons
            'fix': config['baseline_fix'],
            'L2_r': config['baseline_L2_r'],
            'activity_balance': config.get('baseline_activity_balance', 0),
            'L1_Wrec': config['L1_Wrec'],
            'L2_Wrec': config['L2_Wrec'],
            'alpha': alpha
        }

        Network = Networks[config.get('baseline_network_type', config['network_type'])]
        self.baseline_net = Network(self.baseline_config, seed=config['baseline_seed'], name='baseline')

    def _setup_training(self):
        """Setup training parameters and RNG."""
        # Network structure
        self.Nin = self.config['Nin']
        self.N = self.config['N']
        self.Nout = self.config['Nout']
        self.n_actions = len(self.config['actions'])

        # Recurrent noise scaling
        self.scaled_var_rec = (2 * self.config['tau'] / self.dt) * self.config['var_rec']
        self.scaled_baseline_var_rec = ((2 * self.config['tau'] / self.dt) *
                                        self.config['baseline_var_rec'])

        # Run mode
        self.mode = self.config['mode']

        # Maximum trial length
        self.Tmax = int(self.config['tmax'] / self.dt) + 1

        # Reward discounting
        if np.isfinite(self.config['tau_reward']):
            self.alpha_reward = self.dt / self.config['tau_reward']
            self.discount_factor = lambda t: np.exp(-t * self.alpha_reward)

            # Calculate gamma now so we can use it for inference/testing
            self.gamma = np.exp(-self.dt / self.config['tau_reward'])
            self.gamma = min(self.gamma, 0.9999)
        else:
            self.discount_factor = lambda t: 1
            self.gamma = 1.0

        # Terminal/aborted rewards
        self.abort_on_last_t = self.config.get('abort_on_last_t', True)
        self.R_TERMINAL = self.config.get('R_TERMINAL', self.config['R_ABORTED'])
        if self.R_TERMINAL is None:
            self.R_TERMINAL = self.config['R_ABORTED']
        self.R_ABORTED = self.config['R_ABORTED']

        # Random number generator
        self.rng = np.random.RandomState(1)

        # Performance tracker
        self.Performance = self.config['Performance']

        # Policy firing-rate dropout (train-time only)
        dropout_p = self.config.get('policy_dropout', 0.0)
        self.policy_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else None

        # RPE-based D1/D2 modulation
        self.use_rpe_modulation = self.config.get('use_rpe_modulation', False)
        if self.use_rpe_modulation:
            self.rpe_modulation_gain = self.config.get('rpe_modulation_gain', 1.0)
            self.rpe_modulation_clamp = self.config.get('rpe_modulation_clamp', 1.0)

    def _setup_kappa(self, kappa, kappa_dist, kappa_dist_params, seed):
        """
        Setup per-neuron or single kappa values.

        Parameters
        ----------
        kappa : float
            Single kappa value (used if kappa_dist is None).
        kappa_dist : str or None
            Distribution type ('gaussian', 'uniform', or None).
        kappa_dist_params : dict or None
            Distribution parameters.
        seed : int
            Random seed for reproducibility.
        """
        # Check if we're loading from a saved file with kappa configuration
        if hasattr(self, 'loaded_kappa_mode'):
            # Restore from saved file
            self.kappa_mode = self.loaded_kappa_mode
            self.kappa_dist = self.loaded_kappa_dist
            self.kappa_dist_params = self.loaded_kappa_dist_params

            values = self.loaded_kappa_neurons
            if values is None:
                values = np.full(self.baseline_net.N, self.loaded_kappa)
            self._set_kappa_values(values)

            # Clean up temporary attributes
            del self.loaded_kappa_mode
            del self.loaded_kappa_dist
            del self.loaded_kappa_dist_params
            del self.loaded_kappa_neurons
            del self.loaded_kappa
            return

        # New model initialization
        self.kappa_mode = 'single' if kappa_dist is None else 'per_neuron'

        if kappa_dist is None:
            self._set_kappa_values(np.full(self.baseline_net.N, kappa))
        else:
            rng = np.random.RandomState(seed + 1000)
            params = kappa_dist_params or {}

            if kappa_dist == 'gaussian':
                mean = params.get('mean', 0.0)
                std = params.get('std', 0.1)
                kappa_values = rng.normal(mean, std, size=self.baseline_net.N)
                kappa_values = np.clip(kappa_values, -0.9, 0.9)

            elif kappa_dist == 'uniform':
                low = params.get('low', -0.5)
                high = params.get('high', 0.5)
                kappa_values = rng.uniform(low, high, size=self.baseline_net.N)
                kappa_values = np.clip(kappa_values, -0.9, 0.9)

            else:
                raise ValueError(f"Unknown kappa distribution: {kappa_dist}")

            self._set_kappa_values(kappa_values)

    def _set_kappa_values(self, values):
        values = np.clip(np.asarray(values, dtype=np.float32), -0.9, 0.9)
        self.kappa_neurons = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.eta_plus_neurons = 1 + self.kappa_neurons
        self.eta_minus_neurons = 1 - self.kappa_neurons
        self.eta_plus_neurons_expanded = self.eta_plus_neurons.view(1, 1, -1)
        self.eta_minus_neurons_expanded = self.eta_minus_neurons.view(1, 1, -1)
        self.kappa = float(self.kappa_neurons.mean().item())
        self.eta_plus = 1 + self.kappa
        self.eta_minus = 1 - self.kappa

    def make_noise(self, size, var=0):
        """Generate Gaussian noise."""
        if var > 0:
            return torch.randn(*size, device=self.device) * np.sqrt(var)
        return torch.zeros(*size, device=self.device)

    def sample_training_contexts(self, n_trials):
        """Sample training-time external context values c."""
        context_distribution = self.config.get('context_distribution', 'uniform')

        if context_distribution == 'uniform':
            low = self.config.get('context_uniform_low', -1.0)
            high = self.config.get('context_uniform_high', 1.0)
            if high < low:
                raise ValueError("context_uniform_high must be greater than or equal to context_uniform_low")
            return torch.empty(n_trials, device=self.device).uniform_(low, high)

        if context_distribution == 'gaussian':
            mean = self.config.get('context_gaussian_mean', 0.0)
            std = self.config.get('context_gaussian_std', 0.5)
            if std < 0:
                raise ValueError("context_gaussian_std must be non-negative")
            low = self.config.get('context_uniform_low', -1.0)
            high = self.config.get('context_uniform_high', 1.0)
            if high < low:
                raise ValueError("context_uniform_high must be greater than or equal to context_uniform_low")
            return torch.empty(n_trials, device=self.device).normal_(mean, std).clamp_(low, high)

        raise ValueError(
            "context_distribution must be 'uniform' or 'gaussian', "
            f"got {context_distribution!r}"
        )

    def sample_vta_contexts(self, n_trials):
        """Sample trial-constant VTA dopamine contexts in dopamine units."""
        dist = self.config.get('vta_context_distribution', 'uniform')
        weight = self.config.get('vta_context_weight', 1.0)

        if dist == 'uniform':
            low = self.config.get('vta_context_low', -0.9)
            high = self.config.get('vta_context_high', 0.9)
            if high < low:
                raise ValueError("vta_context_high must be greater than or equal to vta_context_low")
            samples = torch.empty(n_trials, device=self.device).uniform_(low, high)
        elif dist == 'gaussian':
            mean = self.config.get('vta_context_mean', 0.0)
            std = self.config.get('vta_context_std', 0.3)
            low = self.config.get('vta_context_low', -0.9)
            high = self.config.get('vta_context_high', 0.9)
            if std < 0:
                raise ValueError("vta_context_std must be non-negative")
            if high < low:
                raise ValueError("vta_context_high must be greater than or equal to vta_context_low")
            samples = torch.empty(n_trials, device=self.device).normal_(mean, std).clamp_(low, high)
        else:
            raise ValueError(
                "vta_context_distribution must be 'uniform' or 'gaussian', "
                f"got {dist!r}"
            )

        return samples * weight
