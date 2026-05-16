"""
Policy Gradient implementation in PyTorch for training RNNs on cognitive tasks.
"""
from collections import OrderedDict
import datetime
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from . import utils
from . import nptools
from .networks import Networks
from .distributional_utils import (
    quantile_huber_loss,
    interpolate_quantiles,
    context_to_quantile_idx,
    get_default_quantiles,
    compute_expected_value_from_quantiles
)


class PolicyGradient:
    """Policy gradient algorithm for training recurrent neural networks."""

    def __init__(self, Task, config_or_savefile, seed, dt=None, load='best', device=None, kappa=0.0,
                 kappa_dist=None, kappa_dist_params=None):
        """
        Initialize PolicyGradient.

        Parameters
        ----------
        kappa : float
            Single kappa value for all neurons (default 0.0).
        kappa_dist : str, optional
            Distribution for per-neuron kappa values. Options:
            - 'gaussian': Normal distribution
            - 'uniform': Uniform distribution
            - None: Use single kappa value
        kappa_dist_params : dict, optional
            Parameters for the kappa distribution:
            - For 'gaussian': {'mean': float, 'std': float}
            - For 'uniform': {'low': float, 'high': float}
        """
        self.task = Task()

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load or create model
        if isinstance(config_or_savefile, str):
            self._load_from_file(config_or_savefile, dt, load)
        else:
            self._create_new_model(config_or_savefile, dt, seed)

        # Move networks to device
        self.policy_net.to(self.device)
        self.baseline_net.to(self.device)

        # Setup
        self._setup_training()

        # Setup kappa (per-neuron or single value)
        self.kappa_dist = kappa_dist
        self.kappa_dist_params = kappa_dist_params
        self._setup_kappa(kappa, kappa_dist, kappa_dist_params, seed)

    def _load_from_file(self, savefile, dt, load):
        """Load model from saved file."""
        save = utils.load(savefile)
        self.save = save
        self.config = save['config']

        print("[ PolicyGradient ]")
        print(f"  Loading {savefile}")
        print(f"  Last saved after {save['iter']} updates.")

        # Performance
        items = OrderedDict()
        items['Best reward'] = f"{save['best_reward']} (after {save['best_iter']} updates)"
        if save['best_perf'] is not None:
            items.update(save['best_perf'].display(output=False))
        utils.print_dict(items)

        # Time step
        self.dt = dt if dt is not None else self.config['dt']
        print(f"Using dt = {self.dt}")

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

        # Detect if loaded model is distributional (based on baseline output size)
        baseline_nout = self.baseline_config.get('Nout', 1)
        if baseline_nout > 1:
            # This is a distributional model
            if not self.config.get('use_distributional_critic', False):
                print(f"\n[ PolicyGradient ] Loaded model has {baseline_nout} quantile outputs.")
                print("  Automatically enabling distributional critic mode.")
                self.config['use_distributional_critic'] = True
                self.config['n_quantiles'] = baseline_nout

    def _create_new_model(self, config, dt, seed):
        """Create new model from config."""
        self.config = config

        # Time step
        self.dt = dt if dt is not None else config['dt']
        print(f"Using dt = {self.dt}")

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
            'L1_Wrec': config['L1_Wrec'],
            'L2_Wrec': config['L2_Wrec'],
            'alpha': alpha
        }

        Network = Networks[config['network_type']]
        self.policy_net = Network(self.policy_config, seed=config['policy_seed'], name='policy')

        # Baseline network configuration
        K = config['baseline_p0'] * config['N']
        baseline_Nin = self.policy_net.N + len(config['actions'])
        if config.get('baseline_include_state', False):
            baseline_Nin += config['Nin']

        # Dynamic output size: 1 for single V(s), n_quantiles for distributional
        baseline_Nout = config.get('n_quantiles', 5) if config.get('use_distributional_critic', False) else 1

        self.baseline_config = {
            'Nin': baseline_Nin,
            'N': config['baseline_N'],
            'Nout': baseline_Nout,
            'p0': config['baseline_p0'],
            'rho': config['baseline_rho'],
            'f_out': 'linear',
            'Win': config['baseline_Win'] * np.sqrt(K) / baseline_Nin,
            'Win_mask': config['baseline_Win_mask'],
            'bout': config['baseline_bout'] if config['baseline_bout'] is not None else 0.0,
            'x0': config.get('baseline_x0', 0),  # Higher initial state to prevent dead neurons
            'fix': config['baseline_fix'],
            'L2_r': config['baseline_L2_r'],
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
        self.rng = nptools.get_rng(seed=1, loc=__name__)

        # Performance tracker
        self.Performance = self.config['Performance']

        # Policy firing-rate dropout (train-time only)
        dropout_p = self.config.get('policy_dropout', 0.0)
        self.policy_dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else None

        # ========== Distributional RL Setup ==========
        # Detect if we're using distributional critic
        self.use_distributional = self.config.get('use_distributional_critic', False)

        if self.use_distributional:
            # Number of quantiles
            self.n_quantiles = self.config.get('n_quantiles', 5)

            # Quantile fractions (tau values)
            self.tau_values = get_default_quantiles(self.n_quantiles).to(self.device)

            # Huber loss threshold for quantile regression
            self.quantile_huber_kappa = self.config.get('quantile_huber_kappa', 1.0)

            print(f"\n[ PolicyGradient ] Distributional critic enabled:")
            print(f"  n_quantiles: {self.n_quantiles}")
            print(f"  tau_values: {self.tau_values.cpu().numpy()}")
            print(f"  Huber kappa: {self.quantile_huber_kappa}")

        # Context-based quantile selection
        self.use_context_quantile = self.config.get('use_context_quantile_selection', False)
        if self.use_context_quantile and not self.use_distributional:
            print("\n[ PolicyGradient ] Warning: use_context_quantile_selection requires "
                  "use_distributional_critic=True. Context quantile selection disabled.")
            self.use_context_quantile = False

        if self.use_context_quantile:
            print(f"\n[ PolicyGradient ] Context-based quantile selection enabled")

        # Expected value computation method
        self.use_quantile_mean_for_ev = self.config.get('use_quantile_mean_for_ev', True)
        if self.use_distributional:
            ev_method = "mean of quantiles" if self.use_quantile_mean_for_ev else "median quantile"
            print(f"\n[ PolicyGradient ] Expected value computed as: {ev_method}")
            if not self.use_quantile_mean_for_ev:
                print("  WARNING: Using median quantile (Q_0.50) as EV is biased for skewed distributions!")

        # Context-based temperature modulation
        self.use_context_temperature = self.config.get('use_context_temperature', False)
        self.temperature_base = self.config.get('temperature_base', 1.0)
        self.temperature_context_scale = self.config.get('temperature_context_scale', 0.5)

        if self.use_context_temperature:
            print(f"\n[ PolicyGradient ] Context-based temperature modulation enabled:")
            print(f"  base_temperature: {self.temperature_base}")
            print(f"  context_scale: {self.temperature_context_scale}")

        # Learned context projection (optional)
        self.use_context_projection_learned = self.config.get('context_projection_learned', False)
        if self.use_context_projection_learned:
            # Create learnable linear projection: baseline_states (N_baseline) → context (1)
            # This can help if simple sum has low variance
            self.context_projection = nn.Linear(self.config['baseline_N'], 1)
            self.context_projection.to(self.device)
            self.context_projection_lr = self.config.get('context_projection_lr', 0.001)

            print(f"\n[ PolicyGradient ] Learned context projection enabled:")
            print(f"  Input dim: {self.config['baseline_N']} (baseline hidden units)")
            print(f"  Output dim: 1 (scalar context)")
            print(f"  Learning rate: {self.context_projection_lr}")

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

            if self.loaded_kappa_neurons is not None:
                # Restore per-neuron kappa values
                kappa_values = self.loaded_kappa_neurons
                self.kappa_neurons = torch.FloatTensor(kappa_values).to(self.device)
                self.eta_plus_neurons = 1 - self.kappa_neurons
                self.eta_minus_neurons = 1 + self.kappa_neurons

                # Cache views for efficient computation
                self.eta_plus_neurons_expanded = self.eta_plus_neurons.view(1, 1, -1)
                self.eta_minus_neurons_expanded = self.eta_minus_neurons.view(1, 1, -1)

                # Store summary statistics
                self.kappa = float(np.mean(kappa_values))
                self.eta_plus = 1 - self.kappa
                self.eta_minus = 1 + self.kappa

                print(f"\n[ PolicyGradient ] Restored per-neuron kappa from saved file:")
                print(f"  Distribution: {self.kappa_dist}")
                print(f"  Parameters: {self.kappa_dist_params}")
                print(f"  Kappa statistics:")
                print(f"    Mean: {self.kappa:.4f}")
                print(f"    Std:  {np.std(kappa_values):.4f}")
                print(f"    Min:  {np.min(kappa_values):.4f}")
                print(f"    Max:  {np.max(kappa_values):.4f}")
                print(f"  Number of neurons: {self.baseline_net.N}")
            else:
                # Single kappa mode from loaded file
                self.kappa = self.loaded_kappa  # Use saved value
                self.eta_plus = 1 - self.kappa
                self.eta_minus = 1 + self.kappa
                self.kappa_neurons = torch.full((self.baseline_net.N,), self.kappa, device=self.device)
                self.eta_plus_neurons = torch.full((self.baseline_net.N,), 1 - self.kappa, device=self.device)
                self.eta_minus_neurons = torch.full((self.baseline_net.N,), 1 + self.kappa, device=self.device)

                # Cache views for efficient computation
                self.eta_plus_neurons_expanded = self.eta_plus_neurons.view(1, 1, -1)
                self.eta_minus_neurons_expanded = self.eta_minus_neurons.view(1, 1, -1)

                print(f"\n[ PolicyGradient ] Restored single kappa = {self.kappa:.4f} from saved file")

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
            # Single kappa value for all neurons
            self.kappa = kappa
            self.eta_plus = 1 - kappa
            self.eta_minus = 1 + kappa
            # Create per-neuron arrays for uniform interface
            self.kappa_neurons = torch.full((self.baseline_net.N,), kappa, device=self.device)
            self.eta_plus_neurons = torch.full((self.baseline_net.N,), 1 - kappa, device=self.device)
            self.eta_minus_neurons = torch.full((self.baseline_net.N,), 1 + kappa, device=self.device)

            # Cache views for efficient computation
            self.eta_plus_neurons_expanded = self.eta_plus_neurons.view(1, 1, -1)
            self.eta_minus_neurons_expanded = self.eta_minus_neurons.view(1, 1, -1)
        else:
            # Per-neuron kappa values from distribution
            rng = nptools.get_rng(seed=seed + 1000, loc=__name__ + '_kappa')

            if kappa_dist == 'gaussian':
                mean = kappa_dist_params.get('mean', 0.0)
                std = kappa_dist_params.get('std', 0.1)
                kappa_values = rng.normal(mean, std, size=self.baseline_net.N)
                # Clip to valid range [-1, 1]
                kappa_values = np.clip(kappa_values, -1.0, 1.0)

            elif kappa_dist == 'uniform':
                low = kappa_dist_params.get('low', -0.5)
                high = kappa_dist_params.get('high', 0.5)
                kappa_values = rng.uniform(low, high, size=self.baseline_net.N)

            else:
                raise ValueError(f"Unknown kappa distribution: {kappa_dist}")

            # Convert to torch tensors
            self.kappa_neurons = torch.FloatTensor(kappa_values).to(self.device)
            self.eta_plus_neurons = 1 - self.kappa_neurons
            self.eta_minus_neurons = 1 + self.kappa_neurons

            # Cache views for efficient computation
            self.eta_plus_neurons_expanded = self.eta_plus_neurons.view(1, 1, -1)
            self.eta_minus_neurons_expanded = self.eta_minus_neurons.view(1, 1, -1)

            # Store summary statistics
            self.kappa = float(np.mean(kappa_values))  # Mean for reporting
            self.eta_plus = 1 - self.kappa
            self.eta_minus = 1 + self.kappa

            print(f"\n[ PolicyGradient ] Per-neuron kappa distribution:")
            print(f"  Distribution: {kappa_dist}")
            print(f"  Parameters: {kappa_dist_params}")
            print(f"  Kappa statistics:")
            print(f"    Mean: {self.kappa:.4f}")
            print(f"    Std:  {np.std(kappa_values):.4f}")
            print(f"    Min:  {np.min(kappa_values):.4f}")
            print(f"    Max:  {np.max(kappa_values):.4f}")
            print(f"  Number of neurons: {self.baseline_net.N}")

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

    def run_trials(self, trials, init=None, init_b=None, return_states=False,
                   perf=None, progress_bar=False, context_input=None, training=False,
                   context_sampling=None):
        """
        Run trials through the network.

        Parameters
        ----------
        context_input : float, array-like, or dict, optional
            Context values for trials. Can be:
            - float: Fixed context for all trials
            - array: Per-trial context values
            - dict: Sampling specification with keys:
                - 'distribution': 'gaussian' or 'uniform'
                - 'mean': Mean for Gaussian (default: 0.0)
                - 'std': Std for Gaussian (default: 0.6)
                - 'low': Lower bound for Uniform (default: -1.0)
                - 'high': Upper bound for Uniform (default: 1.0)
        context_sampling : dict, optional
            DEPRECATED: Use context_input dict instead.
            For backward compatibility only.
        """
        if isinstance(trials, list):
            n_trials = len(trials)
        else:
            n_trials = trials
            trials = []

        # --- 1. SETUP EXTERNAL CONTEXT SIGNAL ---
        # Handle deprecated context_sampling parameter
        if context_sampling is not None and context_input is None:
            context_input = context_sampling

        if context_input is not None:
            # Check if context_input is a sampling specification (dict)
            if isinstance(context_input, dict):
                dist_type = context_input.get('distribution', 'gaussian')
                if dist_type == 'gaussian':
                    mean = context_input.get('mean', 0.0)
                    std = context_input.get('std', 0.6)
                    low = context_input.get('low', -1.0)
                    high = context_input.get('high', 1.0)
                    contexts = torch.empty(n_trials, device=self.device).normal_(mean, std).clamp_(low, high)
                elif dist_type == 'uniform':
                    low = context_input.get('low', -1.0)
                    high = context_input.get('high', 1.0)
                    contexts = torch.empty(n_trials, device=self.device).uniform_(low, high)
                else:
                    raise ValueError(f"Unknown distribution type: {dist_type}. Use 'gaussian' or 'uniform'.")
            # Inference mode: user provided a specific context value(s)
            elif isinstance(context_input, (int, float)):
                contexts = torch.full((n_trials,), float(context_input), device=self.device)
            else:
                contexts = torch.tensor(context_input, device=self.device)
        elif training:
            # Training mode: sample c from the configured distribution.
            contexts = self.sample_training_contexts(n_trials)
        else:
            # Default fallback
            contexts = torch.zeros(n_trials, device=self.device)

        # Storage
        U = torch.zeros(self.Tmax, n_trials, self.Nin, device=self.device)
        Z = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)
        A = torch.zeros(self.Tmax, n_trials, self.n_actions, device=self.device)
        R = torch.zeros(self.Tmax, n_trials, device=self.device)
        M = torch.zeros(self.Tmax, n_trials, device=self.device)

        # --- NEW: Storage for the Policy's Subjective Values ---
        Policy_Values = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)
        Policy_D1_Pull = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)
        Policy_D2_Pull = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)

        # Baseline storage: shape depends on distributional mode
        if self.use_distributional:
            Z_b = torch.zeros(self.Tmax, n_trials, self.n_quantiles, device=self.device)
        else:
            Z_b = torch.zeros(self.Tmax, n_trials, device=self.device)

        # Storage for trial-level information
        prob_l = torch.zeros(n_trials, device=self.device)
        prob_r = torch.zeros(n_trials, device=self.device)
        size_l = torch.zeros(n_trials, device=self.device)
        size_r = torch.zeros(n_trials, device=self.device)

        # Noise
        Q = self.make_noise((self.Tmax, n_trials, self.policy_net.N), self.scaled_var_rec)
        Q_b = self.make_noise((self.Tmax, n_trials, self.baseline_net.N), self.scaled_baseline_var_rec)

        if return_states:
            r_policy = torch.zeros(self.Tmax, n_trials, self.policy_net.N, device=self.device)
            r_value = torch.zeros(self.Tmax, n_trials, self.baseline_net.N, device=self.device)

        if perf is None:
            perf = self.Performance()

        if progress_bar:
            progress_inc = max(int(n_trials / 50), 1)
            progress_half = 25 * progress_inc
            if progress_half > n_trials:
                progress_half = -1
            utils.println("[ PolicyGradient.run_trials ] ")

        with torch.no_grad():
            for n in range(n_trials):
                if progress_bar and n % progress_inc == 0:
                    if n == 0:
                        utils.println("0")
                    elif n == progress_half:
                        utils.println("50")
                    else:
                        utils.println("|")

                if hasattr(self.task, 'start_trial'):
                    self.task.start_trial()

                if n < len(trials):
                    trial = trials[n]
                else:
                    trial = self.task.get_condition(self.rng, self.dt)
                    trials.append(trial)

                if 'prob_l' in trial:
                    prob_l[n] = trial['prob_l']
                    prob_r[n] = trial['prob_r']
                    size_l[n] = trial['size_l']
                    size_r[n] = trial['size_r']

                # t = 0
                t = 0
                if init is None:
                    z_t, x_t = self.policy_net.step_0()
                    z_t_b, x_t_b = self.baseline_net.step_0()
                else:
                    z_t, x_t = init
                    z_t_b, x_t_b = init_b

                Z[t, n] = z_t
                if self.use_distributional:
                    Z_b[t, n] = z_t_b
                else:
                    Z_b[t, n] = z_t_b.squeeze() if z_t_b.dim() > 0 else z_t_b

                if return_states:
                    r_t_policy = self.policy_net.firing_rate(x_t)
                    ctx_val = self._context_for_step(trial, t, contexts[n])
                    r_t_policy_mod = self._apply_opponent_modulation(
                        r_t_policy.unsqueeze(0),
                        torch.as_tensor([ctx_val], device=self.device)
                    ).squeeze(0)
                    r_policy[t, n] = r_t_policy_mod
                    r_value[t, n] = self.baseline_net.firing_rate(x_t_b)

                # --- ACTION SELECTION ---
                r_t_for_action = self.policy_net.firing_rate(x_t.unsqueeze(0))
                ctx_val = self._context_for_step(trial, t, contexts[n])
                r_t_for_action = self._apply_opponent_modulation(
                    r_t_for_action,
                    torch.as_tensor([ctx_val], device=self.device)
                )
                logits_t = self.policy_net.output_layer(r_t_for_action, temperature=1.0, return_logits=True)
                
                # --- NEW: Extract the Policy's Value Computations ---
                # 1. The total subjective value (V) for Fixate, Left, and Right
                Policy_Values[t, n] = logits_t.squeeze(0).detach()
                
                # 2. Look under the hood at how D1 and D2 contribute to that value
                half_N = self.policy_net.N // 2
                Wout = self.policy_net.Wout
                
                # D1 (Upside/Go) contribution
                d1_pull = torch.matmul(r_t_for_action[..., :half_N], Wout[:half_N, :])
                Policy_D1_Pull[t, n] = d1_pull.squeeze(0).detach()
                
                # D2 (Downside/NoGo) contribution
                d2_pull = torch.matmul(r_t_for_action[..., half_N:], Wout[half_N:, :])
                Policy_D2_Pull[t, n] = d2_pull.squeeze(0).detach()
                # -----------------------------------------------------
                
                # Apply temperature modulation based on explicit context
                if self.use_context_temperature:
                    ctx_val = self._context_for_step(trial, t, contexts[n])
                    temp = self._compute_temperature(1, context=ctx_val.unsqueeze(0))
                    z_t_np = torch.softmax(logits_t / temp, dim=-1).squeeze(0).cpu().detach().numpy()
                else:
                    z_t_np = torch.softmax(logits_t, dim=-1).squeeze(0).cpu().detach().numpy()
                
                z_t_np = z_t_np.reshape(self.Nout)
                a_t = self.rng.choice(self.Nout, p=z_t_np)
                A[t, n, a_t] = 1

                # Task step
                u_t_np, r_t, status = self.task.get_step(self.rng, self.dt, trial, t+1, a_t)
                u_t_tensor = torch.FloatTensor(u_t_np).to(self.device)
                
                # --- INJECT CONTEXT ---
                # Since we added 'CONTEXT' to the task inputs, u_t_tensor is already size 8.
                # We overwrite that empty 8th slot (index -1) instead of concatenating a 9th.
                ctx_val = self._context_for_step(trial, t, contexts[n])
                if self.config.get('use_opponent_modulation', False):
                    ctx_val = 0.0

                if u_t_tensor.shape[0] == self.Nin:
                    u_t_tensor[-1] = ctx_val
                    U[t, n] = u_t_tensor
                else:
                    # Fallback for tasks that don't declare context in their inputs map
                    U[t, n] = torch.cat([u_t_tensor, torch.tensor([ctx_val], device=self.device)], dim=0)
                    
                R[t, n] = r_t
                M[t, n] = 1

                # t > 0
                for t in range(1, self.Tmax):
                    if not status['continue']:
                        break

                    u_t = U[t-1, n:n+1]
                    q_t = Q[t, n:n+1]
                    x_t = x_t.unsqueeze(0)
                    z_t, x_t = self.policy_net.step_t(u_t, q_t, x_t)
                    x_t = x_t.squeeze(0)
                    Z[t, n] = z_t

                    r_t_policy = self.policy_net.firing_rate(x_t)
                    ctx_val = self._context_for_step(trial, t, contexts[n])
                    r_t_policy_mod = self._apply_opponent_modulation(
                        r_t_policy.unsqueeze(0),
                        torch.as_tensor([ctx_val], device=self.device)
                    ).squeeze(0)

                    if self.config.get('baseline_include_state', False):
                        u_t_b = torch.cat([U[t-1, n], r_t_policy_mod, A[t-1, n]], dim=-1).unsqueeze(0)
                    else:
                        u_t_b = torch.cat([r_t_policy_mod, A[t-1, n]], dim=-1).unsqueeze(0)
                    q_t_b = Q_b[t, n:n+1]
                    x_t_b = x_t_b.unsqueeze(0)
                    z_t_b, x_t_b = self.baseline_net.step_t(u_t_b, q_t_b, x_t_b)
                    x_t_b = x_t_b.squeeze(0)
                    
                    if self.use_distributional:
                        Z_b[t, n] = z_t_b.squeeze(0)
                    else:
                        Z_b[t, n] = z_t_b.squeeze() if z_t_b.dim() > 0 else z_t_b

                    if return_states:
                        r_policy[t, n] = r_t_policy_mod
                        r_value[t, n] = self.baseline_net.firing_rate(x_t_b)

                    # --- ACTION SELECTION ---
                    r_t_for_action = self.policy_net.firing_rate(x_t.unsqueeze(0))
                    ctx_val = self._context_for_step(trial, t, contexts[n])
                    r_t_for_action = self._apply_opponent_modulation(
                        r_t_for_action,
                        torch.as_tensor([ctx_val], device=self.device)
                    )
                    logits_t = self.policy_net.output_layer(r_t_for_action, temperature=1.0, return_logits=True)
                    
                    # --- NEW: Extract the Policy's Value Computations ---
                    # 1. The total subjective value (V) for Fixate, Left, and Right
                    Policy_Values[t, n] = logits_t.squeeze(0).detach()
                    
                    # 2. Look under the hood at how D1 and D2 contribute to that value
                    half_N = self.policy_net.N // 2
                    Wout = self.policy_net.Wout
                    
                    # D1 (Upside/Go) contribution
                    d1_pull = torch.matmul(r_t_for_action[..., :half_N], Wout[:half_N, :])
                    Policy_D1_Pull[t, n] = d1_pull.squeeze(0).detach()
                    
                    # D2 (Downside/NoGo) contribution
                    d2_pull = torch.matmul(r_t_for_action[..., half_N:], Wout[half_N:, :])
                    Policy_D2_Pull[t, n] = d2_pull.squeeze(0).detach()
                    # -----------------------------------------------------
                    
                    if self.use_context_temperature:
                        ctx_val = self._context_for_step(trial, t, contexts[n])
                        temp = self._compute_temperature(1, context=ctx_val.unsqueeze(0))
                        z_t_np = torch.softmax(logits_t / temp, dim=-1).squeeze(0).cpu().detach().numpy()
                    else:
                        z_t_np = torch.softmax(logits_t, dim=-1).squeeze(0).cpu().detach().numpy()
                    
                    z_t_np = z_t_np.reshape(self.Nout)
                    a_t = self.rng.choice(self.Nout, p=z_t_np)
                    A[t, n, a_t] = 1

                    if self.abort_on_last_t and t == self.Tmax - 1:
                        U[t, n] = 0
                        R[t, n] = self.R_TERMINAL
                        status = {'continue': False, 'reward': R[t, n].item()}
                    else:
                        u_t_np, r_t, status = self.task.get_step(self.rng, self.dt, trial, t+1, a_t)
                        u_t_tensor = torch.FloatTensor(u_t_np).to(self.device)
                        
                        # --- INJECT CONTEXT ---
                        # Since we added 'CONTEXT' to the task inputs, u_t_tensor is already size 8.
                        # We overwrite that empty 8th slot (index -1) instead of concatenating a 9th.
                        ctx_val = self._context_for_step(trial, t, contexts[n])
                        if self.config.get('use_opponent_modulation', False):
                            ctx_val = 0.0

                        if u_t_tensor.shape[0] == self.Nin:
                            u_t_tensor[-1] = ctx_val
                            U[t, n] = u_t_tensor
                        else:
                            # Fallback for tasks that don't declare context in their inputs map
                            U[t, n] = torch.cat([u_t_tensor, torch.tensor([ctx_val], device=self.device)], dim=0)
                            
                        R[t, n] = r_t

                    M[t, n] = 1

                perf.update(trial, status)

        if progress_bar:
            print("100")

        with torch.no_grad():
            RPE_objective = self._compute_online_td_error(R, Z_b, M, self.gamma)

            if self.kappa_mode == 'per_neuron':
                RPE_exp = RPE_objective.unsqueeze(-1)
                RPE_subj_neurons = torch.where(
                    RPE_exp > 0,
                    self.eta_plus_neurons_expanded * RPE_exp,
                    self.eta_minus_neurons_expanded * RPE_exp
                )
                RPE_subjective = RPE_subj_neurons.mean(dim=-1)
            else:
                RPE_subjective = torch.where(
                    RPE_objective > 0,
                    self.eta_plus * RPE_objective,
                    self.eta_minus * RPE_objective
                )

        # --- 3. EXPOSE CONTEXTS IN RESULTS ---
        results = {
            'U': U, 'Q': Q, 'Q_b': Q_b, 'Z': Z, 'Z_b': Z_b,
            'A': A, 'R': R, 'M': M, 'perf': perf,
            'contexts': contexts,
            'prob_l': prob_l, 'prob_r': prob_r,
            'size_l': size_l, 'size_r': size_r,
            'RPE_objective': RPE_objective,
            'RPE_subjective': RPE_subjective,
            'Policy_Values': Policy_Values,     # NEW
            'Policy_D1_Pull': Policy_D1_Pull,   # NEW
            'Policy_D2_Pull': Policy_D2_Pull    # NEW
        }
        if return_states:
            results['r_policy'] = r_policy
            results['r_value'] = r_value

        return results

    def _select_quantile(self, q_values, context=None):
        """
        Select quantile values based on context signal or compute expected value.

        If context-based quantile selection is disabled, this method either:
        1. Computes expected value as mean of all quantiles (use_quantile_mean_for_ev=True)
        2. Returns the median quantile Q_0.50 (use_quantile_mean_for_ev=False)

        IMPORTANT: Median ≠ Expected Value for skewed distributions!
        Option 1 is statistically correct, option 2 is biologically plausible but biased.

        Parameters
        ----------
        q_values : torch.Tensor, shape (T, B, n_quantiles)
            Predicted quantile values from distributional critic.
        context : torch.Tensor, shape (T, B), (B,), or scalar, optional
            Context signal for quantile selection.
            - (T, B): Time-varying context per timestep and trial
            - (B,): Per-trial context (constant across time)
            - scalar: Same context for all trials
            Range: typically [-1, +1], but will be normalized via tanh.

        Returns
        -------
        selected_values : torch.Tensor, shape (T, B)
            Selected quantile values (one per timestep and trial).
        """
        if not self.use_distributional:
            # Not distributional, return as-is (should be shape (T, B))
            return q_values

        if not self.use_context_quantile or context is None:
            # No context modulation: use default baseline value
            if self.use_quantile_mean_for_ev:
                # Correct expected value: mean across all quantiles
                # E[Z] = ∫ Q(τ) dτ ≈ (1/n) Σ Q(τ_i)
                return compute_expected_value_from_quantiles(q_values, self.tau_values, method='mean')
            else:
                # Biologically plausible but biased: use median quantile
                # Note: For skewed distributions, median ≠ mean!
                median_idx = self.n_quantiles // 2
                return q_values[:, :, median_idx]

        # Context-based quantile selection
        T, B = q_values.shape[:2]

        # Ensure context is a tensor
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, device=self.device)

        # Handle time-varying context: (T, B)
        if context.dim() == 2 and context.shape[0] == T:
            # Time-varying context: apply per-timestep quantile selection
            # This is more accurate but slower
            selected_values = []
            for t in range(T):
                # Map context at time t to quantile indices
                quantile_idx_t = context_to_quantile_idx(context[t], self.n_quantiles)
                # Select quantile for all trials at time t
                # q_values[t:t+1] has shape (1, B, n_quantiles)
                selected_t = interpolate_quantiles(q_values[t:t+1], quantile_idx_t)
                selected_values.append(selected_t)
            return torch.cat(selected_values, dim=0)  # Shape: (T, B)

        # Handle per-trial context: (B,) or scalar
        elif context.dim() <= 1:
            # Handle scalar context (broadcast to batch)
            if context.dim() == 0:
                context = context.unsqueeze(0).expand(B)

            # Map context to quantile index
            quantile_idx = context_to_quantile_idx(context, self.n_quantiles)

            # Interpolate between quantiles
            # Same quantile index used for all timesteps
            selected_values = interpolate_quantiles(q_values, quantile_idx)

            return selected_values
        else:
            raise ValueError(f"Unexpected context shape: {context.shape}. "
                           f"Expected (T={T}, B={B}), ({B},), or scalar.")

    def run_context_sweep(self, c_min=-1.0, c_max=1.0, c_step=0.1,
                         context_std=0.1, n_trials_per_context=20,
                         context_bounds=(-1.0, 1.0), **run_trials_kwargs):
        """
        Run trials across a sweep of context means with Gaussian sampling.

        Instead of testing with fixed context values, this samples multiple trials
        from N(c, σ²) for each mean context c. This provides a more robust estimate
        of behavior at each context level and simulates realistic neural variability.

        Parameters
        ----------
        c_min : float, default=-1.0
            Minimum context mean value.
        c_max : float, default=1.0
            Maximum context mean value.
        c_step : float, default=0.1
            Step size for context sweep.
        context_std : float, default=0.1
            Standard deviation of Gaussian sampling around each context mean.
        n_trials_per_context : int, default=20
            Number of trials to sample at each context mean.
        context_bounds : tuple, default=(-1.0, 1.0)
            (low, high) bounds for clamping sampled contexts.
        **run_trials_kwargs
            Additional keyword arguments passed to run_trials().

        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'context_means': Array of context mean values tested
            - 'all_results': List of result dicts for each mean
            - 'all_contexts': List of actual sampled contexts for each mean

        Examples
        --------
        >>> # Load model and run sweep
        >>> pg = PolicyGradient.load('model.pkl')
        >>> sweep = pg.run_context_sweep(c_min=-1.0, c_max=1.0, c_step=0.1,
        ...                              context_std=0.1, n_trials_per_context=20)
        >>>
        >>> # Analyze results
        >>> for i, c_mean in enumerate(sweep['context_means']):
        ...     results = sweep['all_results'][i]
        ...     mean_reward = results['R'].sum(dim=0).mean().item()
        ...     print(f"c={c_mean:.2f}: reward={mean_reward:.2f}")
        """
        import numpy as np

        context_means = np.arange(c_min, c_max + c_step/2, c_step)
        all_results = []
        all_contexts = []

        for c_mean in context_means:
            # Create context specification
            context_spec = {
                'distribution': 'gaussian',
                'mean': float(c_mean),
                'std': context_std,
                'low': context_bounds[0],
                'high': context_bounds[1]
            }

            # Run trials
            results = self.run_trials(n_trials_per_context,
                                     context_input=context_spec,
                                     **run_trials_kwargs)

            # Store results and actual contexts
            all_results.append(results)
            all_contexts.append(results['contexts'].cpu().numpy())

        return {
            'context_means': context_means,
            'all_results': all_results,
            'all_contexts': all_contexts
        }

    def _compute_temperature(self, batch_size, context=None):
        """
        Compute softmax temperature from context signal.

        Parameters
        ----------
        batch_size : int
            Batch size (number of trials).
        context : torch.Tensor, shape (B,) or scalar, optional
            Context signal for temperature modulation.

        Returns
        -------
        temperature : torch.Tensor, shape (B,)
            Temperature values for each trial.
        """
        if not self.use_context_temperature or context is None:
            # Return base temperature for all trials
            return torch.full((batch_size,), self.temperature_base, device=self.device)

        # Ensure context is a tensor
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, device=self.device)

        # Handle scalar context (broadcast to batch)
        if context.dim() == 0:
            context = context.unsqueeze(0).expand(batch_size)

        # Modulate temperature based on context
        # High context → high temperature → more exploration
        # Low context → low temperature → more exploitation
        temperature = self.temperature_base * (1.0 + self.temperature_context_scale * torch.tanh(context))

        return temperature

    def _context_for_step(self, trial, t, context_value):
        """Return the context value for a timestep, optionally limiting it to decision time."""
        if not self.config.get('context_decision_only', False):
            return context_value

        decision_epoch = trial.get('epochs', {}).get('decision') if trial is not None else None
        if decision_epoch is None:
            return context_value

        return context_value if t in decision_epoch else 0.0

    def train(self, savefile, recover=False):
        """Train the policy and baseline networks."""
        # Training parameters
        max_iter = self.config['max_iter']
        lr = self.config['lr']
        baseline_lr = self.config['baseline_lr']
        n_gradient = self.config['n_gradient']
        n_validation = self.config['n_validation']
        checkfreq = self.config['checkfreq']

        use_x0 = (self.mode == 'continuous')

        # Print settings
        items = OrderedDict()
        items['Device'] = str(self.device)
        items['Network type (policy)'] = self.config['network_type']
        items['Network type (baseline)'] = self.config.get('baseline_network_type',
                                                           self.config['network_type'])
        items['N (policy)'] = self.config['N']
        items['N (baseline)'] = self.config['baseline_N']
        items['Conn. prob. (policy)'] = self.config['p0']
        items['Conn. prob. (baseline)'] = self.config['baseline_p0']
        items['dt'] = f"{self.dt} ms"
        items['tau_reward'] = f"{self.config['tau_reward']} ms"
        items['var_rec (policy)'] = self.config['var_rec']
        items['var_rec (baseline)'] = self.config['baseline_var_rec']
        items['Learning rate (policy)'] = lr
        items['Learning rate (baseline)'] = baseline_lr
        items['Max time steps'] = self.Tmax
        items['Num. trials (gradient)'] = n_gradient
        items['Num. trials (validation)'] = n_validation
        utils.print_dict(items)

        # Optimizers
        policy_optimizer = optim.Adam(self.policy_net.get_trainable_params(), lr=lr)
        baseline_optimizer = optim.Adam(self.baseline_net.get_trainable_params(), lr=baseline_lr)

        # Context projection optimizer (if learned projection is enabled)
        if hasattr(self, 'context_projection'):
            context_projection_optimizer = optim.Adam(
                self.context_projection.parameters(),
                lr=self.context_projection_lr
            )
            print(f"[ PolicyGradient ] Context projection optimizer initialized (lr={self.context_projection_lr})")
        else:
            context_projection_optimizer = None

        # Initialize training state
        if recover and hasattr(self, 'save'):
            print("Resume training.")
            iter_start = self.save['iter']
            print(f"Last saved was after {self.save['iter']} updates.")

            self.rng.set_state(self.save['rng_state'])

            best_iter = self.save['best_iter']
            best_reward = self.save['best_reward']
            best_perf = self.save['best_perf']
            best_policy_params = self.save['best_policy_params']
            best_baseline_params = self.save['best_baseline_params']

            training_history = self.save['training_history']
            trials_tot = self.save['trials_tot']

            # Restore optimizer states if available
            if 'policy_optimizer_state' in self.save:
                policy_optimizer.load_state_dict(self.save['policy_optimizer_state'])
            if 'baseline_optimizer_state' in self.save:
                baseline_optimizer.load_state_dict(self.save['baseline_optimizer_state'])
        else:
            iter_start = 0
            best_iter = -1
            best_reward = -np.inf
            best_perf = None
            best_policy_params = self.policy_net.get_state_dict_numpy()
            best_baseline_params = self.baseline_net.get_state_dict_numpy()
            training_history = []
            trials_tot = 0

        # Training loop
        if hasattr(self.task, 'start_session'):
            self.task.start_session(self.rng)

        tstart = datetime.datetime.now()

        try:
            for iter_ in range(iter_start, max_iter + 1):
                # Validation
                if iter_ % checkfreq == 0 or iter_ == max_iter:
                    if n_validation > 0:
                        elapsed = utils.elapsed_time(tstart)
                        print(f"After {iter_} updates ({elapsed})")

                        # Save RNG state
                        rng_state = self.rng.get_state()

                        # Generate validation trials
                        val_trials = [self.task.get_condition(self.rng, self.dt)
                                     for _ in range(n_validation)]

                        # Run validation
                        val_results = self.run_trials(val_trials, progress_bar=True)
                        perf = val_results['perf']

                        if hasattr(self.task, 'update'):
                            self.task.update(perf)

                        # Check termination
                        terminate = False
                        if hasattr(self.task, 'terminate'):
                            if self.task.terminate(perf):
                                terminate = True

                        # Compute mean reward
                        mean_reward = torch.sum(val_results['R'] * val_results['M']).item() / n_validation

                        # Save if best
                        record = {
                            'iter': iter_,
                            'mean_reward': mean_reward,
                            'n_trials': trials_tot,
                            'perf': perf
                        }

                        if mean_reward > best_reward or terminate:
                            best_iter = iter_
                            best_reward = mean_reward
                            best_perf = perf
                            best_policy_params = self.policy_net.get_state_dict_numpy()
                            best_baseline_params = self.baseline_net.get_state_dict_numpy()
                            record['new_best'] = True
                        else:
                            record['new_best'] = False

                        training_history.append(record)

                        # Save checkpoint
                        save = {
                            'iter': iter_,
                            'config': self.config,
                            'policy_config': self.policy_config,
                            'baseline_config': self.baseline_config,
                            'policy_masks': self.policy_net.masks,
                            'baseline_masks': self.baseline_net.masks,
                            'current_policy_params': self.policy_net.get_state_dict_numpy(),
                            'current_baseline_params': self.baseline_net.get_state_dict_numpy(),
                            'best_iter': best_iter,
                            'best_reward': best_reward,
                            'best_perf': best_perf,
                            'best_policy_params': best_policy_params,
                            'best_baseline_params': best_baseline_params,
                            'rng_state': rng_state,
                            'training_history': training_history,
                            'trials_tot': trials_tot,
                            'policy_optimizer_state': policy_optimizer.state_dict(),
                            'baseline_optimizer_state': baseline_optimizer.state_dict(),
                            # Kappa configuration
                            'kappa_mode': self.kappa_mode,
                            'kappa_dist': self.kappa_dist,
                            'kappa_dist_params': self.kappa_dist_params,
                            'kappa_neurons': self.kappa_neurons.cpu().numpy() if self.kappa_mode == 'per_neuron' else None,
                            'kappa': self.kappa  # Save scalar kappa value
                        }
                        utils.save(savefile, save)

                        # Display results
                        items = OrderedDict()
                        items['Best reward'] = f'{best_reward} (iteration {best_iter})'
                        items['Mean reward'] = f'{mean_reward}'

                        if perf is not None:
                            items.update(perf.display(output=False))

                        # Value prediction error
                        V = torch.zeros_like(val_results['R'])
                        for k in range(V.shape[0]):
                            V[k] = torch.sum(val_results['R'][k:] * val_results['M'][k:], dim=0)
                        
                        # Handle distributional values: use median quantile for error
                        Z_b_error = val_results['Z_b']
                        if len(Z_b_error.shape) == 3:
                            # Distributional mode: extract median quantile
                            Z_b_error = Z_b_error[..., Z_b_error.shape[-1] // 2]
                        
                        error = torch.sqrt(torch.sum((Z_b_error - V)**2 * val_results['M']) /
                                          torch.sum(val_results['M'])).item()
                        items['Prediction error'] = f'{error}'

                        # ==========================================
                        # LIVE CONTEXT DIAGNOSTICS
                        # ==========================================
                        try:
                            # 1. Check if the connection is alive (Weight Magnitude)
                            if hasattr(self.policy_net, 'Win'):
                                ctx_weights = self.policy_net.Win[:, -1].detach()
                                items['Ctx Weight Std'] = f"{ctx_weights.std().item():.4f}"
                            
                            # 2. Live Logit Divergence (The absolute proof)
                            # Get correct initial state shape (usually 1D, needs to be 2D for step_t)
                            _, x0 = self.policy_net.step_0() 
                            if x0.dim() == 1:
                                x0 = x0.unsqueeze(0)
                            
                            # Create 2D dummy inputs: (Batch=1, Features=Nin)
                            u_neg = torch.zeros(1, self.Nin, device=self.device)
                            u_pos = torch.zeros(1, self.Nin, device=self.device)
                            
                            # Inject contexts into the 8th dimension (index -1)
                            u_neg[0, -1] = -1.0
                            u_pos[0, -1] = 1.0
                            
                            q_dummy = torch.zeros(1, self.policy_net.N, device=self.device)
                            
                            # Forward pass a single timestep
                            _, x_neg = self.policy_net.step_t(u_neg, q_dummy, x0)
                            _, x_pos = self.policy_net.step_t(u_pos, q_dummy, x0)
                            
                            # Extract raw logits
                            logits_neg = self.policy_net.output_layer(self.policy_net.firing_rate(x_neg), temperature=1.0, return_logits=True)
                            logits_pos = self.policy_net.output_layer(self.policy_net.firing_rate(x_pos), temperature=1.0, return_logits=True)
                            
                            # Calculate the maximum absolute difference between the decisions
                            max_diff = torch.max(torch.abs(logits_pos - logits_neg)).item()
                            items['Ctx Logit Shift'] = f"{max_diff:.4f}"
                            
                        except Exception as e:
                            # Actually print the error so we aren't flying blind!
                            items['Ctx Logit Shift'] = f"Crash: {str(e)}"
                        # ==========================================

                        utils.print_dict(items)

                        # Check termination conditions
                        if best_reward >= self.config['target_reward']:
                            print("Target reward reached.")
                            return

                        if terminate:
                            print("Termination criterion satisfied.")
                            return

                if iter_ == max_iter:
                    print(f"Reached maximum number of iterations ({iter_}).")
                    sys.exit(0)

                # Training step
                # Generate training trials
                train_trials = [self.task.get_condition(self.rng, self.dt)
                               for _ in range(n_gradient)]

                # Run trials
                train_results = self.run_trials(train_trials, return_states=True, training=True)

                trials_tot += n_gradient

                # Update baseline network
                self._update_baseline(train_results, baseline_optimizer)

                # Update policy network (and context projection if enabled)
                self._update_policy(train_results, policy_optimizer, context_projection_optimizer)

        except KeyboardInterrupt:
            print(f"Training interrupted by user during iteration {iter_}.")
            sys.exit(0)

    def _compute_returns(self, rewards, gamma):
        """
        Compute discounted returns (Monte Carlo) for each timestep.
        This works better than TD for sparse reward tasks.

        Return_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^(T-t)*r_T
        """
        T, B = rewards.shape
        returns = torch.zeros_like(rewards)

        # Work backwards from end of episode
        running_return = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns

    def _compute_online_td_error(self, rewards, values, mask, gamma):
        """
        Compute online TD error.

        TD_error(t) = r(t) + γ*V(t+1) - V(t)

        Parameters
        ----------
        rewards : torch.Tensor, shape (T, B)
            Immediate rewards at each timestep
        values : torch.Tensor, shape (T, B) or (T, B, n_quantiles)
            Value predictions at each timestep. If distributional (3D),
            uses expected value (mean of quantiles) or median based on config.
        mask : torch.Tensor, shape (T, B)
            Valid timesteps (1 = valid, 0 = invalid)
        gamma : float
            Discount factor

        Returns
        -------
        td_error : torch.Tensor, shape (T, B)
            Online TD error at each timestep
        """
        # Handle distributional values: use expected value or median
        if len(values.shape) == 3:
            # Distributional mode: shape (T, B, n_quantiles)
            if self.use_quantile_mean_for_ev:
                # Correct expected value: mean of quantiles
                values = compute_expected_value_from_quantiles(values, self.tau_values, method='mean')
            else:
                # Biased but biologically plausible: median quantile
                values = values[..., values.shape[-1] // 2]

        T, B = rewards.shape
        td_error = torch.zeros_like(rewards)

        for t in range(T - 1):
            # TD error = r(t) + γ*V(t+1) - V(t)
            # This ONLY uses information available up to time t+1
            td_error[t] = rewards[t] + gamma * values[t+1] - values[t]

        # Last timestep: no future value
        td_error[T-1] = rewards[T-1] - values[T-1]

        # Apply mask
        td_error = td_error * mask

        return td_error


    def _update_baseline(self, results, optimizer):
            """Update baseline network using standard MSE."""
            # Ensure gamma is set
            if not hasattr(self, 'gamma'):
                if np.isinf(self.config.get('tau_reward', np.inf)):
                    self.gamma = 1
                else:
                    self.gamma = np.exp(-self.dt / self.config['tau_reward'])
                self.gamma = min(self.gamma, 0.9999)

            # Call the new Standard MSE loss
            loss, z_all = self._standard_mse_loss(results)
            
            # Store unbiased Z_b for the policy update
            results["Z_b"] = z_all.detach()

            # Update parameters
            optimizer.zero_grad()
            loss.backward()

            grad_clip = self.config.get('baseline_grad_clip', None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.baseline_net.parameters(), grad_clip)

            optimizer.step()

    def _standard_mse_loss(self, results):
            """Asymmetric MSE loss so the Critic learns context-modulated (subjective) values."""
            R = results['R']
            M = results['M']

            r_policy = results['r_policy']
            A = results['A']
            if self.config.get('baseline_include_state', False):
                U = results['U']
                baseline_inputs = torch.cat([U, r_policy, A], dim=-1)
            else:
                baseline_inputs = torch.cat([r_policy, A], dim=-1)
            baseline_inputs_trimmed = baseline_inputs[:-1]
            B_size = baseline_inputs_trimmed.shape[1]
            x0 = self.baseline_net.x0.unsqueeze(0).expand(B_size, -1)

            z_pred, states_b = self.baseline_net(
                baseline_inputs_trimmed,
                results['Q_b'][:-1],
                x0
            )
            z_0, _ = self.baseline_net.step_0(x0)
            
            if z_0.dim() == 2:
                z_0 = z_0.squeeze(-1)
            if z_pred.dim() == 3:
                z_pred = z_pred.squeeze(-1)

            z_all = torch.cat([z_0.unsqueeze(0), z_pred], dim=0)

            with torch.no_grad():
                returns = self._compute_returns(R, self.gamma)

            # 1. Compute standard delta
            delta = returns - z_all
            
            # 2. Derive Bounded Context Multipliers (Same as Actor)
            context_signal = results.get('contexts', None)
            
            if context_signal is not None:
                # Keep it bounded between [-0.9, 0.9] to prevent zero-gradients
                c_bounded = context_signal * 0.9 
                
                eta_plus = 1.0 + c_bounded  
                eta_minus = 1.0 - c_bounded 
                
                eta_plus = eta_plus.unsqueeze(0).expand_as(delta)
                eta_minus = eta_minus.unsqueeze(0).expand_as(delta)
            else:
                eta_plus = torch.ones_like(delta)
                eta_minus = torch.ones_like(delta)

            # 3. Asymmetric Squared Error
            # Penalize underestimations by eta_plus, overestimations by eta_minus
            delta_squared = torch.where(delta > 0, eta_plus * (delta ** 2), eta_minus * (delta ** 2))
            
            # Apply valid timestep mask
            delta_squared = delta_squared * M
            
            n_valid = M.sum()
            if n_valid > 0:
                loss = torch.sum(delta_squared) / n_valid
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            reg = self.baseline_net.get_regs(x0, states_b, M[:-1])
            loss += reg

            return loss, z_all

    def _expectile_mse_loss(self, results):
        """
        Original expectile MSE loss for single-value baseline.

        Uses kappa-modulated asymmetric loss to implement risk-sensitive value learning.
        """
        # Extract data
        R = results['R']  # Shape (T, B)
        M = results['M']  # Shape (T, B)

        # Get baseline predictions
        r_policy = results['r_policy']
        A = results['A']
        if self.config.get('baseline_include_state', False):
            U = results['U']
            baseline_inputs = torch.cat([U, r_policy, A], dim=-1)
        else:
            baseline_inputs = torch.cat([r_policy, A], dim=-1)
        baseline_inputs_trimmed = baseline_inputs[:-1]
        B_size = baseline_inputs_trimmed.shape[1]
        x0 = self.baseline_net.x0.unsqueeze(0).expand(B_size, -1)

        z_pred, states_b = self.baseline_net(
            baseline_inputs_trimmed,
            results['Q_b'][:-1],
            x0
        )
        z_0, _ = self.baseline_net.step_0(x0)
        if z_0.dim() == 2:
            z_0 = z_0.squeeze(-1)
        if z_pred.dim() == 3:
            z_pred = z_pred.squeeze(-1)

        z_all = torch.cat([z_0.unsqueeze(0), z_pred], dim=0)  # Shape (T, B)

        # Compute Monte Carlo returns from each state
        with torch.no_grad():
            returns = self._compute_returns(R, self.gamma)  # Shape (T, B)

        # Monte Carlo advantage = actual return - predicted value
        delta = returns - z_all

        # Apply mask
        delta = delta * M
        n_valid = M.sum()

        # Risk-sensitive transformation (per-neuron or single value)
        if self.kappa_mode == 'per_neuron':
            # Apply per-neuron eta transformations to the Monte Carlo advantage
            # Expand delta to match neuron dimensions: (T, B) -> (T, B, 1)
            delta_expanded = delta.unsqueeze(-1)

            # Create per-neuron modulated deltas
            # Shape: (T, B, N) where each neuron gets its own eta scaling
            delta_prime_per_neuron = torch.where(
                delta_expanded > 0,
                self.eta_plus_neurons_expanded * delta_expanded,
                self.eta_minus_neurons_expanded * delta_expanded
            )

            # Average across neurons to get the final delta_prime (T, B)
            delta_prime = delta_prime_per_neuron.mean(dim=-1)

        else:
            # Single kappa value (original behavior)
            eta_plus = self.eta_plus
            eta_minus = self.eta_minus

            delta_prime = torch.where(delta > 0,
                                    eta_plus * delta,
                                    eta_minus * delta)

        # Compute loss
        if n_valid > 0:
            loss = torch.sum(delta_prime**2 * M) / n_valid
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        reg = self.baseline_net.get_regs(x0, states_b, M[:-1])
        loss += reg

        return loss, z_all, delta_prime

    def _quantile_huber_loss(self, results):
        """
        Quantile Huber loss for distributional baseline.

        Trains the baseline network to predict multiple quantiles of the return distribution.
        """
        # Extract data
        R = results['R']  # Shape (T, B)
        M = results['M']  # Shape (T, B)

        # Get baseline quantile predictions
        r_policy = results['r_policy']
        A = results['A']
        if self.config.get('baseline_include_state', False):
            U = results['U']
            baseline_inputs = torch.cat([U, r_policy, A], dim=-1)
        else:
            baseline_inputs = torch.cat([r_policy, A], dim=-1)
        baseline_inputs_trimmed = baseline_inputs[:-1]
        B_size = baseline_inputs_trimmed.shape[1]
        x0 = self.baseline_net.x0.unsqueeze(0).expand(B_size, -1)

        z_pred, states_b = self.baseline_net(
            baseline_inputs_trimmed,
            results['Q_b'][:-1],
            x0
        )
        z_0, _ = self.baseline_net.step_0(x0)

        # Ensure correct shape for distributional output
        # z_0 shape: (B, n_quantiles) or (B, 1, n_quantiles)
        # z_pred shape: (T-1, B, n_quantiles)
        if z_0.dim() == 3:
            z_0 = z_0.squeeze(1)  # (B, n_quantiles)

        z_all = torch.cat([z_0.unsqueeze(0), z_pred], dim=0)  # Shape (T, B, n_quantiles)

        # Compute Monte Carlo returns from each state
        with torch.no_grad():
            returns = self._compute_returns(R, self.gamma)  # Shape (T, B)

        # Quantile Huber loss
        loss = quantile_huber_loss(
            pred_quantiles=z_all,
            targets=returns,
            tau_values=self.tau_values,
            kappa=self.quantile_huber_kappa,
            mask=M
        )

        # Add regularization
        reg = self.baseline_net.get_regs(x0, states_b, M[:-1])
        loss += reg

        return loss, z_all

    def _compute_distributional_advantage(self, returns, z_all_quantiles, context=None):
        """
        Compute advantage using context-selected quantile as baseline.

        Parameters
        ----------
        returns : torch.Tensor, shape (T, B)
            Monte Carlo returns.
        z_all_quantiles : torch.Tensor, shape (T, B, n_quantiles)
            Predicted quantile values.
        context : torch.Tensor, shape (B,) or scalar, optional
            Context signal for quantile selection.

        Returns
        -------
        advantage : torch.Tensor, shape (T, B)
            Advantage values for policy gradient.
        """
        # Select appropriate quantile based on context
        selected_baseline = self._select_quantile(z_all_quantiles, context)

        # Advantage = return - selected quantile baseline
        advantage = returns - selected_baseline

        return advantage

    def _apply_opponent_modulation(self, r, context_signal):
        """
        Intercepts network firing rates to apply D1/D2 multiplicative gain.
        Preserves Vanilla RL behavior if use_opponent_modulation is False.
        """
        if not self.config.get('use_opponent_modulation', False) or context_signal is None:
            return r

        if context_signal.dim() == 0:
            context_signal = context_signal.unsqueeze(0)

        # 1. Biological bounded gains (Mechanism 1)
        c_bounded = torch.clamp(context_signal, -1.0, 1.0) * 0.9
        gain_D1 = 1.0 + c_bounded  # Dopamine boosts D1 (Optimism)
        gain_D2 = 1.0 - c_bounded  # Dopamine suppresses D2 (Pessimism)

        # 2. Auto-reshape gains to broadcast over r's dimensions
        if r.dim() == 3:      # (T, B, N) - Sequence
            gain_D1 = gain_D1.view(1, -1, 1)
            gain_D2 = gain_D2.view(1, -1, 1)
        elif r.dim() == 2:    # (B, N) - Batch
            gain_D1 = gain_D1.view(-1, 1)
            gain_D2 = gain_D2.view(-1, 1)
        elif r.dim() == 1:    # (N) - Single state
            gain_D1 = gain_D1.view(1)
            gain_D2 = gain_D2.view(1)

        # 3. Split hidden state into Opponent Pathways
        half_N = r.shape[-1] // 2
        r_D1_mod = r[..., :half_N] * gain_D1
        r_D2_mod = r[..., half_N:] * gain_D2

        # 4. Recombine modulated state
        return torch.cat([r_D1_mod, r_D2_mod], dim=-1)

    def _update_policy(self, results, optimizer, context_projection_optimizer=None):
            """Update policy network using context-modulated advantages (Mechanism 1)."""
            U = results['U']
            A = results['A']
            M = results['M']
            R = results['R']
            Q_trimmed = results['Q'][:-1]
            
            # 1. Get the unbiased baseline (Critic) and Returns
            baseline_value = results["Z_b"]  # Shape: (T, B)
            
            with torch.no_grad():
                returns = self._compute_returns(R, self.gamma)
                
            # 2. Compute standard, unbiased Advantage
            delta = returns - baseline_value

            # 3. Apply Mechanism 1: Context dictates learning rate asymmetry
            context_signal = results.get('contexts', None)
            
            if context_signal is not None:
                # Keep the asymmetry bounded so gradients never collapse to zero.
                c_bounded = torch.clamp(context_signal, -1.0, 1.0) * 0.9

                # High dopamine (positive context) -> eta_plus > eta_minus -> Risk Seeking
                # Low dopamine (negative context) -> eta_plus < eta_minus -> Risk Averse
                eta_plus = 1.0 + c_bounded
                eta_minus = 1.0 - c_bounded
                
                # Expand to match delta shape: (T, B)
                eta_plus = eta_plus.unsqueeze(0).expand_as(delta)
                eta_minus = eta_minus.unsqueeze(0).expand_as(delta)
            else:
                # Fallback to standard RL if no context is provided
                eta_plus = torch.ones_like(delta)
                eta_minus = torch.ones_like(delta)

            # 4. Asymmetrically scale the advantage based on the sign of the RPE
            scaled_advantage = torch.where(delta > 0, eta_plus * delta, eta_minus * delta)

            # 5. Forward pass through policy network
            U_trimmed = U[:-1]
            B_size = U_trimmed.shape[1]
            x0 = self.policy_net.x0.unsqueeze(0).expand(B_size, -1)
            _, states = self.policy_net(U_trimmed, Q_trimmed, x0)

            # Apply Temperature
            if self.use_context_temperature and context_signal is not None:
                temperature = self._compute_temperature(B_size, context=context_signal)
            else:
                temperature = None

            r_0 = self.policy_net.firing_rate(x0)
            r_0 = self._apply_opponent_modulation(r_0, context_signal)
            if self.policy_dropout is not None:
                r_0 = self.policy_dropout(r_0)
            log_z_0 = self.policy_net.log_output(r_0, temperature=temperature)

            r_pred = self.policy_net.firing_rate(states)
            r_pred = self._apply_opponent_modulation(r_pred, context_signal)
            if self.policy_dropout is not None:
                r_pred = self.policy_dropout(r_pred)
            log_z_pred = self.policy_net.log_output(r_pred, temperature=temperature)

            # 6. Compute log probabilities of chosen actions
            logpi_0 = torch.sum(log_z_0 * A[0], dim=-1)
            logpi_t = torch.sum(log_z_pred * A[1:], dim=-1)
            logpi_all = torch.cat([logpi_0.unsqueeze(0), logpi_t], dim=0)

            # 7. REINFORCE objective with SCALED advantage
            weighted_logpi = logpi_all * scaled_advantage * M
            J = torch.sum(weighted_logpi) / B_size

            reg = self.policy_net.get_regs(x0, states, M[:-1])
            loss = -J + reg

            # 8. Gradient update
            optimizer.zero_grad()
            if context_projection_optimizer is not None:
                context_projection_optimizer.zero_grad()

            loss.backward()

            grad_clip = self.config.get('grad_clip', None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), grad_clip)
                if context_projection_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.context_projection.parameters(), grad_clip)

            optimizer.step()
            if context_projection_optimizer is not None:
                context_projection_optimizer.step()
