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


class PolicyGradient:
    """Policy gradient algorithm for training recurrent neural networks."""

    def __init__(self, Task, config_or_savefile, seed, dt=None, load='best', device=None, kappa=0.0):
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

        self.kappa = kappa
        self.eta_plus = 1 - kappa
        self.eta_minus = 1 + kappa

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
        self.baseline_config = {
            'Nin': baseline_Nin,
            'N': config['baseline_N'],
            'Nout': 1,
            'p0': config['baseline_p0'],
            'rho': config['baseline_rho'],
            'f_out': 'linear',
            'Win': config['baseline_Win'] * np.sqrt(K) / baseline_Nin,
            'Win_mask': config['baseline_Win_mask'],
            'bout': config['baseline_bout'] if config['baseline_bout'] is not None else config['R_ABORTED'],
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
        else:
            self.discount_factor = lambda t: 1

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
            Dictionary containing trial data.
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
                # Baseline has Nout=1, squeeze to scalar
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
                    # Baseline has Nout=1, squeeze to scalar
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
                        R[t, n] = r_t *  1 #self.discount_factor(t)

                    M[t, n] = 1

                # Update performance
                perf.update(trial, status)

        if progress_bar:
            print("100")

        # Return results
        results = {
            'U': U, 'Q': Q, 'Q_b': Q_b, 'Z': Z, 'Z_b': Z_b,
            'A': A, 'R': R, 'M': M, 'perf': perf,
            'prob_l': prob_l, 'prob_r': prob_r,
            'size_l': size_l, 'size_r': size_r
        }
        if return_states:
            results['r_policy'] = r_policy
            results['r_value'] = r_value

        return results

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

        grad_norms_policy = []
        grad_norms_baseline = []

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
                            'baseline_optimizer_state': baseline_optimizer.state_dict()
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

    def _update_baseline(self, results, optimizer):
        """Update baseline (value) network using risk-sensitive MONTE CARLO returns."""

        # --- DEBUGGING: Initialize step counter if not exists ---
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        self._debug_step += 1
        
        # Only print detailed debug every N steps to avoid spam
        debug_freq = 100
        verbose_debug = (self._debug_step % debug_freq == 0)

        # --- Extract relevant data from results ---
        R = results['R']  # Shape (T, B), now contains UNDISCOUNTED r_t
        M = results['M']  # Shape (T, B), mask

        # --- DEBUG: Reward statistics ---
        if verbose_debug:
            print(f"\n{'='*80}")
            print(f"BASELINE UPDATE DEBUG - Step {self._debug_step}, κ={self.kappa:.3f}")
            print(f"{'='*80}")
            print(f"Rewards R:")
            print(f"  Shape: {R.shape}")
            print(f"  Mean: {R.mean().item():.6f}")
            print(f"  Std: {R.std().item():.6f}")
            print(f"  Min: {R.min().item():.6f}")
            print(f"  Max: {R.max().item():.6f}")
            print(f"  Positive: {(R > 0).sum().item()}/{R.numel()} ({100*(R > 0).float().mean().item():.1f}%)")
            print(f"  Negative: {(R < 0).sum().item()}/{R.numel()} ({100*(R < 0).float().mean().item():.1f}%)")
            print(f"  Zero: {(R == 0).sum().item()}/{R.numel()}")

        # --- Get baseline predictions V(s_t) ---
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
        
        # z_all = V(s_t) for t=0..T-1
        z_all = torch.cat([z_0.unsqueeze(0), z_pred], dim=0) # Shape (T, B)

        # --- DEBUG: Value function statistics ---
        if verbose_debug:
            print(f"\nValue Function V(s):")
            print(f"  Shape: {z_all.shape}")
            print(f"  Mean: {z_all.mean().item():.6f}")
            print(f"  Std: {z_all.std().item():.6f}")
            print(f"  Min: {z_all.min().item():.6f}")
            print(f"  Max: {z_all.max().item():.6f}")

        # --- Compute Monte Carlo returns: G_t = sum_{k=t}^{T} γ^{k-t} * R_k ---
        if not hasattr(self, 'gamma'):
            self.gamma = np.exp(-self.dt / self.config['tau_reward'])
            print(f"\n*** Initialized gamma = {self.gamma:.6f} ***")
            self.gamma = 1

        # Compute discounted returns from the end backwards
        T, B = R.shape
        G = torch.zeros_like(R)  # Monte Carlo returns
        G_running = torch.zeros(B, device=self.device)

        # Backward pass to compute returns
        for t in range(T - 1, -1, -1):
            G_running = R[t] + self.gamma * G_running * M[t]
            G[t] = G_running

        if verbose_debug:
            print(f"\nMonte Carlo Returns G:")
            print(f"  Shape: {G.shape}")
            print(f"  Mean: {G.mean().item():.6f}")
            print(f"  Std: {G.std().item():.6f}")
            print(f"  Min: {G.min().item():.6f}")
            print(f"  Max: {G.max().item():.6f}")

        # Compute advantage: δ_t = G_t - V(s_t)
        V_s_t = z_all  # Shape (T, B)
        delta = G - V_s_t

        # --- DEBUG: Advantage statistics (BEFORE risk-sensitivity) ---
        if verbose_debug:
            print(f"\nAdvantage δ = G - V(s) (BEFORE risk-sensitivity):")
            print(f"  Formula: δ = G_t - V(s_t), where G_t = sum_{{k=t}}^T γ^{{k-t}} * R_k")
            print(f"  γ = {self.gamma:.6f}")
            print(f"  Mean: {delta.mean().item():.6f}")
            print(f"  Std: {delta.std().item():.6f}")
            print(f"  Min: {delta.min().item():.6f}")
            print(f"  Max: {delta.max().item():.6f}")
            print(f"  Positive δ: {(delta > 0).sum().item()}/{delta.numel()} ({100*(delta > 0).float().mean().item():.1f}%)")
            print(f"  Negative δ: {(delta < 0).sum().item()}/{delta.numel()} ({100*(delta < 0).float().mean().item():.1f}%)")

            pos_deltas = delta[delta > 0]
            neg_deltas = delta[delta < 0]
            if len(pos_deltas) > 0:
                print(f"  Positive δ - Mean: {pos_deltas.mean().item():.6f}, Max: {pos_deltas.max().item():.6f}")
            if len(neg_deltas) > 0:
                print(f"  Negative δ - Mean: {neg_deltas.mean().item():.6f}, Min: {neg_deltas.min().item():.6f}")

        # --- Risk-Sensitive Transformation ---
        # Use the values set during initialization
        eta_plus = self.eta_plus
        eta_minus = self.eta_minus

        # --- DEBUG: Risk-sensitivity parameters ---
        if verbose_debug:
            print(f"\nRisk-Sensitivity Parameters:")
            print(f"  κ (kappa): {self.kappa:.6f}")
            print(f"  η⁺ (eta_plus): {eta_plus:.6f}  [scales POSITIVE δ]")
            print(f"  η⁻ (eta_minus): {eta_minus:.6f}  [scales NEGATIVE δ]")
            print(f"  Interpretation:")
            if self.kappa > 0:
                print(f"    Risk-AVERSE: Amplify bad news (η⁻={eta_minus:.2f}x), dampen good news (η⁺={eta_plus:.2f}x)")
            elif self.kappa < 0:
                print(f"    Risk-SEEKING: Amplify good news (η⁺={eta_plus:.2f}x), dampen bad news (η⁻={eta_minus:.2f}x)")
            else:
                print(f"    Risk-NEUTRAL: No scaling")
                
            # Warning checks
            if eta_plus < 0:
                print(f"  ⚠️  WARNING: η⁺ < 0! Positive errors will flip sign!")
            if eta_minus < 0:
                print(f"  ⚠️  WARNING: η⁻ < 0! Negative errors will flip sign!")
            if abs(eta_plus) > 2.0 or abs(eta_minus) > 2.0:
                print(f"  ⚠️  WARNING: Extreme scaling factors (>2x) may cause instability!")
        
        delta_prime = torch.where(delta > 0,
                                eta_plus * delta,
                                eta_minus * delta)

        # --- DEBUG: Advantage statistics (AFTER risk-sensitivity) ---
        if verbose_debug:
            print(f"\nAdvantage δ′ (AFTER risk-sensitivity):")
            print(f"  Formula: δ′ = η⁺·δ if δ>0, else η⁻·δ")
            print(f"  Mean: {delta_prime.mean().item():.6f}")
            print(f"  Std: {delta_prime.std().item():.6f}")
            print(f"  Min: {delta_prime.min().item():.6f}")
            print(f"  Max: {delta_prime.max().item():.6f}")
            
            # Compare scaling effects
            pos_mask = delta > 0
            neg_mask = delta < 0
            if pos_mask.any():
                orig_pos = delta[pos_mask].abs().mean()
                scaled_pos = delta_prime[pos_mask].abs().mean()
                print(f"  Positive δ → δ′:")
                print(f"    Original mean |δ|: {orig_pos.item():.6f}")
                print(f"    Scaled mean |δ′|: {scaled_pos.item():.6f}")
                print(f"    Ratio (scaled/orig): {(scaled_pos/orig_pos).item():.6f} (should be ≈{eta_plus:.6f})")
                
            if neg_mask.any():
                orig_neg = delta[neg_mask].abs().mean()
                scaled_neg = delta_prime[neg_mask].abs().mean()
                print(f"  Negative δ → δ′:")
                print(f"    Original mean |δ|: {orig_neg.item():.6f}")
                print(f"    Scaled mean |δ′|: {scaled_neg.item():.6f}")
                print(f"    Ratio (scaled/orig): {(scaled_neg/orig_neg).item():.6f} (should be ≈{eta_minus:.6f})")
            
            # Check for sign flips
            sign_flips = ((delta > 0) & (delta_prime < 0)) | ((delta < 0) & (delta_prime > 0))
            if sign_flips.any():
                print(f"  ⚠️  SIGN FLIPS DETECTED: {sign_flips.sum().item()} errors changed sign!")
            
            # Overall effect
            ratio = delta_prime.abs().mean() / (delta.abs().mean() + 1e-10)
            print(f"  Overall magnitude ratio |δ′|/|δ|: {ratio.item():.6f}")

        # --- Compute baseline loss ---
        loss = torch.sum((delta_prime)**2 * M) / torch.sum(M)

        # --- DEBUG: Loss statistics ---
        if verbose_debug:
            loss_unscaled = torch.sum(delta**2 * M) / torch.sum(M)
            print(f"\nLoss:")
            print(f"  Baseline loss (with risk-sensitivity): {loss.item():.6f}")
            print(f"  Baseline loss (without risk-sensitivity): {loss_unscaled.item():.6f}")
            print(f"  Ratio (risk/neutral): {(loss/loss_unscaled).item():.6f}")

        reg = self.baseline_net.get_regs(x0, states_b, M[:-1])
        if verbose_debug:
            print(f"  Regularization: {reg.item():.6f}")
            print(f"  Total loss: {(loss + reg).item():.6f}")
        
        loss += reg

        # --- Gradient update ---
        optimizer.zero_grad()
        loss.backward()

        # --- DEBUG: Gradient statistics ---
        if verbose_debug:
            total_norm = 0
            for p in self.baseline_net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"  Gradient norm (baseline - before clipping): {total_norm:.6f}")

        # --- Gradient clipping ---
        grad_clip = self.config.get('baseline_grad_clip', None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.baseline_net.parameters(), grad_clip)
            if verbose_debug:
                # Compute gradient norm after clipping
                total_norm_clipped = 0
                for p in self.baseline_net.parameters():
                    if p.grad is not None:
                        total_norm_clipped += p.grad.data.norm(2).item() ** 2
                total_norm_clipped = total_norm_clipped ** 0.5
                print(f"  Gradient norm (baseline - after clipping to {grad_clip}): {total_norm_clipped:.6f}")

        optimizer.step()

        # --- Store shared δ′ for policy update (detached) ---
        results["delta_prime"] = delta_prime.detach()
        results["Z_b"] = z_all.detach()
        
        # --- DEBUG: Summary ---
        if verbose_debug:
            print(f"{'='*80}\n")


    def _update_policy(self, results, optimizer):
        """Update policy network with risk-sensitive advantages (based on Monte Carlo returns)."""

        # Use same debug counter as baseline
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        
        debug_freq = 100
        verbose_debug = (self._debug_step % debug_freq == 0)

        # --- Extract relevant data ---
        U = results['U']
        A = results['A']
        M = results['M']
        Q_trimmed = results['Q'][:-1]
        delta_prime = results["delta_prime"]

        if verbose_debug:
            print(f"\n{'='*80}")
            print(f"POLICY UPDATE DEBUG - Step {self._debug_step}, κ={self.kappa:.3f}")
            print(f"{'='*80}")
            print(f"Advantage (delta_prime) from baseline:")
            print(f"  Shape: {delta_prime.shape}")
            print(f"  Mean: {delta_prime.mean().item():.6f}")
            print(f"  Std: {delta_prime.std().item():.6f}")
            print(f"  Min: {delta_prime.min().item():.6f}")
            print(f"  Max: {delta_prime.max().item():.6f}")
            print(f"  Positive: {(delta_prime > 0).float().mean().item()*100:.1f}%")
            print(f"  Negative: {(delta_prime < 0).float().mean().item()*100:.1f}%")

        # --- Forward pass through policy network ---
        U_trimmed = U[:-1]
        B_size = U_trimmed.shape[1]
        x0 = self.policy_net.x0.unsqueeze(0).expand(B_size, -1)
        z_pred, states = self.policy_net(U_trimmed, Q_trimmed, x0)
        
        r_0 = self.policy_net.firing_rate(x0)
        log_z_0 = self.policy_net.log_output(r_0)
        
        r_pred = self.policy_net.firing_rate(states)
        log_z_pred = torch.stack([
            self.policy_net.log_output(r_pred[t]) for t in range(r_pred.shape[0])
        ])

        # --- Compute log probabilities of chosen actions ---
        logpi_0 = torch.sum(log_z_0 * A[0], dim=-1) * M[0]
        logpi_t = torch.sum(log_z_pred * A[1:], dim=-1) * M[1:]
        logpi_all = torch.cat([logpi_0.unsqueeze(0), logpi_t], dim=0)
        
        if verbose_debug:
            print(f"\nLog-probabilities:")
            print(f"  Shape: {logpi_all.shape}")
            print(f"  Mean: {logpi_all.mean().item():.6f}")
            print(f"  Std: {logpi_all.std().item():.6f}")
            print(f"  Min: {logpi_all.min().item():.6f}")
            print(f"  Max: {logpi_all.max().item():.6f}")

        # --- Policy gradient objective (TD Actor-Critic) ---
        weighted_logpi = logpi_all * delta_prime * M
        
        if verbose_debug:
            print(f"\nWeighted log-probs (log π · δ′):")
            print(f"  Mean: {weighted_logpi.mean().item():.6f}")
            print(f"  Sum: {weighted_logpi.sum().item():.6f}")
            print(f"  This will be divided by batch size {B_size}")
        
        J = torch.sum(weighted_logpi) / B_size

        # Regularization
        reg = self.policy_net.get_regs(x0, states, M[:-1])
        loss = -J + reg

        if verbose_debug:
            print(f"\nPolicy Loss:")
            print(f"  Objective J (before negation): {J.item():.6f}")
            print(f"  Regularization: {reg.item():.6f}")
            print(f"  Total loss (-J + reg): {loss.item():.6f}")

        # --- Gradient update ---
        optimizer.zero_grad()
        loss.backward()

        if verbose_debug or self.kappa != 0:
            # Gradient diagnostics
            total_grad_norm = 0
            max_grad = 0
            min_grad = float('inf')
            nan_params, inf_params = [], []

            for name, param in self.policy_net.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    max_grad = max(max_grad, param.grad.abs().max().item())
                    min_grad = min(min_grad, param.grad.abs().min().item())
                    if torch.isnan(param.grad).any():
                        nan_params.append(name)
                    if torch.isinf(param.grad).any():
                        inf_params.append(name)
            
            if verbose_debug:
                print(f"\nGradient Statistics (policy - before clipping):")
                print(f"  Total norm: {total_grad_norm:.6f}")
                print(f"  Max absolute: {max_grad:.6f}")
                print(f"  Min absolute: {min_grad:.6f}")

                if nan_params:
                    print(f"  ⚠️  NaN gradients in: {nan_params}")
                if inf_params:
                    print(f"  ⚠️  Inf gradients in: {inf_params}")

            # Extra warnings for extreme kappa values
            if abs(self.kappa) > 0.5:
                if nan_params or inf_params:
                    print(f"\n⚠️⚠️⚠️  GRADIENT ISSUES WITH κ={self.kappa:.3f}  ⚠️⚠️⚠️")
                if total_grad_norm > 1000:
                    print(f"\n⚠️  Very large gradient norm ({total_grad_norm:.1f}) with κ={self.kappa:.3f}")

        # --- Gradient clipping ---
        grad_clip = self.config.get('grad_clip', None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), grad_clip)
            if verbose_debug or self.kappa != 0:
                # Compute gradient norm after clipping
                total_grad_norm_clipped = 0
                for param in self.policy_net.parameters():
                    if param.grad is not None:
                        total_grad_norm_clipped += param.grad.norm().item()
                if verbose_debug:
                    print(f"\nGradient Statistics (policy - after clipping to {grad_clip}):")
                    print(f"  Total norm: {total_grad_norm_clipped:.6f}")

        optimizer.step()
        
        if verbose_debug:
            print(f"{'='*80}\n")