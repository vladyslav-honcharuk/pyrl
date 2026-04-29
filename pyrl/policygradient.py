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
    get_default_quantiles
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

        # Context-based temperature modulation
        self.use_context_temperature = self.config.get('use_context_temperature', False)
        self.temperature_base = self.config.get('temperature_base', 1.0)
        self.temperature_context_scale = self.config.get('temperature_context_scale', 0.5)

        if self.use_context_temperature:
            print(f"\n[ PolicyGradient ] Context-based temperature modulation enabled:")
            print(f"  base_temperature: {self.temperature_base}")
            print(f"  context_scale: {self.temperature_context_scale}")

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

    def run_trials(self, trials, init=None, init_b=None, return_states=False,
                   perf=None, progress_bar=False):
        """
        Run trials through the network.

        Parameters
        ----------
        trials : int or list
            Number of trials to run, or list of trial conditions.
        init : tuple, optional
            Initial policy network state (z, x).
        init_b : tuple, optional
            Initial baseline network state (z, x).
        return_states : bool
            Whether to return internal states.
        perf : Performance, optional
            Performance tracker.
        progress_bar : bool
            Whether to show progress bar.

        Returns
        -------
        results : dict
            Dictionary containing trial data including:
            - RPE_objective: Online TD error (r(t) + γV(t+1) - V(t))
            - RPE_subjective: Online TD error with kappa filter
        """
        if isinstance(trials, list):
            n_trials = len(trials)
        else:
            n_trials = trials
            trials = []

        # Storage
        U = torch.zeros(self.Tmax, n_trials, self.Nin, device=self.device)
        Z = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)
        A = torch.zeros(self.Tmax, n_trials, self.n_actions, device=self.device)
        R = torch.zeros(self.Tmax, n_trials, device=self.device)
        M = torch.zeros(self.Tmax, n_trials, device=self.device)

        # Baseline storage: shape depends on distributional mode
        if self.use_distributional:
            Z_b = torch.zeros(self.Tmax, n_trials, self.n_quantiles, device=self.device)
        else:
            Z_b = torch.zeros(self.Tmax, n_trials, device=self.device)

        # Storage for trial-level information (for risk-sensitive learning)
        prob_l = torch.zeros(n_trials, device=self.device)
        prob_r = torch.zeros(n_trials, device=self.device)
        size_l = torch.zeros(n_trials, device=self.device)
        size_r = torch.zeros(n_trials, device=self.device)

        # Noise
        Q = self.make_noise((self.Tmax, n_trials, self.policy_net.N), self.scaled_var_rec)
        Q_b = self.make_noise((self.Tmax, n_trials, self.baseline_net.N), self.scaled_baseline_var_rec)

        # Firing rates storage
        if return_states:
            r_policy = torch.zeros(self.Tmax, n_trials, self.policy_net.N, device=self.device)
            r_value = torch.zeros(self.Tmax, n_trials, self.baseline_net.N, device=self.device)

        # Performance tracking
        if perf is None:
            perf = self.Performance()

        # Progress bar
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

                # Initialize trial
                if hasattr(self.task, 'start_trial'):
                    self.task.start_trial()

                # Generate trial condition
                if n < len(trials):
                    trial = trials[n]
                else:
                    trial = self.task.get_condition(self.rng, self.dt)
                    trials.append(trial)

                # Extract trial information if available (for risk-sensitive learning)
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
                # Baseline: store depending on distributional mode
                if self.use_distributional:
                    Z_b[t, n] = z_t_b  # Shape: (n_quantiles,)
                else:
                    # Single value, squeeze to scalar
                    Z_b[t, n] = z_t_b.squeeze() if z_t_b.dim() > 0 else z_t_b

                if return_states:
                    r_policy[t, n] = self.policy_net.firing_rate(x_t)
                    r_value[t, n] = self.baseline_net.firing_rate(x_t_b)

                # Select action
                z_t_np = z_t.cpu().numpy().reshape(self.Nout)
                a_t = self.rng.choice(self.Nout, p=z_t_np)
                A[t, n, a_t] = 1

                # Task step
                u_t_np, r_t, status = self.task.get_step(self.rng, self.dt, trial, t+1, a_t)
                U[t, n] = torch.FloatTensor(u_t_np).to(self.device)
                R[t, n] = r_t
                M[t, n] = 1

                # t > 0
                for t in range(1, self.Tmax):
                    if not status['continue']:
                        break

                    # Policy network step
                    u_t = U[t-1, n:n+1]
                    q_t = Q[t, n:n+1]
                    x_t = x_t.unsqueeze(0)
                    z_t, x_t = self.policy_net.step_t(u_t, q_t, x_t)
                    x_t = x_t.squeeze(0)
                    Z[t, n] = z_t

                    # Baseline network step
                    r_t_policy = self.policy_net.firing_rate(x_t)
                    u_t_b = torch.cat([r_t_policy, A[t-1, n]], dim=-1).unsqueeze(0)
                    q_t_b = Q_b[t, n:n+1]
                    x_t_b = x_t_b.unsqueeze(0)
                    z_t_b, x_t_b = self.baseline_net.step_t(u_t_b, q_t_b, x_t_b)
                    x_t_b = x_t_b.squeeze(0)
                    # Baseline: store depending on distributional mode
                    if self.use_distributional:
                        Z_b[t, n] = z_t_b.squeeze(0)  # Shape: (n_quantiles,)
                    else:
                        # Single value, squeeze to scalar
                        Z_b[t, n] = z_t_b.squeeze() if z_t_b.dim() > 0 else z_t_b

                    if return_states:
                        r_policy[t, n] = r_t_policy
                        r_value[t, n] = self.baseline_net.firing_rate(x_t_b)

                    # Select action
                    z_t_np = z_t.cpu().numpy().reshape(self.Nout)
                    a_t = self.rng.choice(self.Nout, p=z_t_np)
                    A[t, n, a_t] = 1

                    # Task step
                    if self.abort_on_last_t and t == self.Tmax - 1:
                        U[t, n] = 0
                        R[t, n] = self.R_TERMINAL
                        status = {'continue': False, 'reward': R[t, n].item()}
                    else:
                        u_t_np, r_t, status = self.task.get_step(self.rng, self.dt, trial, t+1, a_t)
                        U[t, n] = torch.FloatTensor(u_t_np).to(self.device)
                        R[t, n] = r_t

                    M[t, n] = 1

                # Update performance
                perf.update(trial, status)

        if progress_bar:
            print("100")

        # Calculate online reward prediction error:
        # RPE(t) = r(t) + γ*V(t+1) - V(t)

        with torch.no_grad():
            # Compute online TD error: δ(t) = r(t) + γ*V(t+1) - V(t)
            RPE_objective = self._compute_online_td_error(R, Z_b, M, self.gamma)

            # Apply kappa-based risk sensitivity.

            if self.kappa_mode == 'per_neuron':
                RPE_exp = RPE_objective.unsqueeze(-1)  # reshape

                RPE_subj_neurons = torch.where(
                    RPE_exp > 0,
                    self.eta_plus_neurons_expanded * RPE_exp,   # Dampen gains
                    self.eta_minus_neurons_expanded * RPE_exp   # Amplify losses
                )
                RPE_subjective = RPE_subj_neurons.mean(dim=-1)

            else:
                RPE_subjective = torch.where(
                    RPE_objective > 0,
                    self.eta_plus * RPE_objective,  # Dampen gains
                    self.eta_minus * RPE_objective  # Amplify losses
                )

        # Return results
        results = {
            'U': U, 'Q': Q, 'Q_b': Q_b, 'Z': Z, 'Z_b': Z_b,
            'A': A, 'R': R, 'M': M, 'perf': perf,
            'prob_l': prob_l, 'prob_r': prob_r,
            'size_l': size_l, 'size_r': size_r,
            'RPE_objective': RPE_objective,    # Online TD error: r(t) + γV(t+1) - V(t)
            'RPE_subjective': RPE_subjective   # Online TD error with kappa filter
        }
        if return_states:
            results['r_policy'] = r_policy
            results['r_value'] = r_value

        return results

    def _select_quantile(self, q_values, context=None):
        """
        Select quantile values based on context signal.

        If context-based quantile selection is disabled, returns the median quantile.

        Parameters
        ----------
        q_values : torch.Tensor, shape (T, B, n_quantiles)
            Predicted quantile values from distributional critic.
        context : torch.Tensor, shape (B,) or scalar, optional
            Context signal for quantile selection.
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
            # Use median quantile (index = n_quantiles // 2)
            median_idx = self.n_quantiles // 2
            return q_values[:, :, median_idx]

        # Context-based quantile selection
        B = q_values.shape[1]

        # Ensure context is a tensor
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context, device=self.device)

        # Handle scalar context (broadcast to batch)
        if context.dim() == 0:
            context = context.unsqueeze(0).expand(B)

        # Map context to quantile index
        quantile_idx = context_to_quantile_idx(context, self.n_quantiles)

        # Interpolate between quantiles
        selected_values = interpolate_quantiles(q_values, quantile_idx)

        return selected_values

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
                        error = torch.sqrt(torch.sum((val_results['Z_b'] - V)**2 * val_results['M']) /
                                          torch.sum(val_results['M'])).item()
                        items['Prediction error'] = f'{error}'

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
                train_results = self.run_trials(train_trials, return_states=True)

                trials_tot += n_gradient

                # Update baseline network
                self._update_baseline(train_results, baseline_optimizer)

                # Update policy network
                self._update_policy(train_results, policy_optimizer)

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
            uses median quantile.
        mask : torch.Tensor, shape (T, B)
            Valid timesteps (1 = valid, 0 = invalid)
        gamma : float
            Discount factor

        Returns
        -------
        td_error : torch.Tensor, shape (T, B)
            Online TD error at each timestep
        """
        # Handle distributional values: use median quantile
        if len(values.shape) == 3:
            # Distributional mode: shape (T, B, n_quantiles)
            # Use median quantile (index n_quantiles//2)
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
        """
        Update baseline network.

        Dispatches to either expectile MSE loss (original) or quantile Huber loss
        (distributional) based on configuration.
        """
        # Initialize gamma if needed
        if not hasattr(self, 'gamma'):
            if np.isinf(self.config.get('tau_reward', np.inf)):
                self.gamma = 1
            else:
                self.gamma = np.exp(-self.dt / self.config['tau_reward'])
            self.gamma = min(self.gamma, 0.9999)  # Cap at 0.99

        if self.use_distributional:
            # NEW: Distributional critic with quantile Huber loss
            loss, z_all_quantiles = self._quantile_huber_loss(results)
            # Store quantile predictions for policy update
            results["Z_b_all_quantiles"] = z_all_quantiles.detach()
        else:
            # ORIGINAL: Single-value critic with expectile MSE loss
            loss, z_all, delta_prime = self._expectile_mse_loss(results)
            # Store for policy (use delta_prime as advantage)
            results["delta_prime"] = delta_prime.detach()
            results["Z_b"] = z_all.detach()

        # Update parameters
        optimizer.zero_grad()
        loss.backward()

        grad_clip = self.config.get('baseline_grad_clip', None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.baseline_net.parameters(), grad_clip)

        optimizer.step()

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

    def _update_policy(self, results, optimizer):
        """Update policy network using Monte Carlo returns with risk-sensitive advantage."""

        # --- Extract relevant data ---
        U = results['U']
        A = results['A']
        M = results['M']
        R = results['R']
        Q_trimmed = results['Q'][:-1]

        # --- Compute advantage ---
        if self.use_distributional:
            # NEW: Distributional advantage
            z_all_quantiles = results["Z_b_all_quantiles"]  # Shape: (T, B, n_quantiles)

            # Compute returns
            with torch.no_grad():
                returns = self._compute_returns(R, self.gamma)

            # Context signal (for now, None - will add context input later)
            context = None

            # Compute distributional advantage
            advantage = self._compute_distributional_advantage(returns, z_all_quantiles, context)
        else:
            # ORIGINAL: Risk-sensitive advantage from expectile MSE
            advantage = results["delta_prime"]

        # --- Forward pass through policy network ---
        U_trimmed = U[:-1]
        B_size = U_trimmed.shape[1]
        x0 = self.policy_net.x0.unsqueeze(0).expand(B_size, -1)
        _, states = self.policy_net(U_trimmed, Q_trimmed, x0)

        r_0 = self.policy_net.firing_rate(x0)
        log_z_0 = self.policy_net.log_output(r_0)

        r_pred = self.policy_net.firing_rate(states)
        log_z_pred = self.policy_net.log_output(r_pred)

        # --- Compute log probabilities of chosen actions ---
        logpi_0 = torch.sum(log_z_0 * A[0], dim=-1)
        logpi_t = torch.sum(log_z_pred * A[1:], dim=-1)
        logpi_all = torch.cat([logpi_0.unsqueeze(0), logpi_t], dim=0)

        # --- REINFORCE objective (Monte Carlo Policy Gradient) ---
        # ∇J = E[∇log π(a|s) * A(s,a)]
        # where A(s,a) is the advantage (risk-sensitive or distributional)
        weighted_logpi = logpi_all * advantage * M

        J = torch.sum(weighted_logpi) / B_size

        # Regularization
        reg = self.policy_net.get_regs(x0, states, M[:-1])
        loss = -J + reg

        # --- Gradient update ---
        optimizer.zero_grad()
        loss.backward()

        # --- Gradient clipping ---
        grad_clip = self.config.get('grad_clip', None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), grad_clip)

        optimizer.step()
