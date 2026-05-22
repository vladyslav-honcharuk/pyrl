"""Rollout and inference code for recurrent actor-critic trainers."""

import numpy as np
import torch

from . import utils


class RolloutMixin:
    def _format_task_input(self, u_t_tensor, ctx_val, direct_context_requested):
        """Return a network input vector, placing context only in a real context slot."""
        inputs = self.config.get('inputs', {})
        context_idx = inputs.get('CONTEXT') if isinstance(inputs, dict) else None
        use_direct_context = direct_context_requested or context_idx is not None

        if self.config.get('use_opponent_modulation', False):
            ctx_val = 0.0

        if u_t_tensor.shape[0] == self.Nin:
            if context_idx is not None:
                u_t_tensor[context_idx] = ctx_val
            elif use_direct_context:
                if float(torch.as_tensor(ctx_val).detach().cpu()) != 0.0:
                    raise ValueError(
                        "Direct context input was requested but the model has no CONTEXT input capacity. "
                        "Train with --training-context-input or omit context_input for this model."
                    )
            return u_t_tensor

        if use_direct_context and u_t_tensor.shape[0] + 1 == self.Nin:
            return torch.cat([
                u_t_tensor,
                torch.as_tensor([ctx_val], device=self.device, dtype=u_t_tensor.dtype)
            ], dim=0)

        raise ValueError(
            f"Task input length {u_t_tensor.shape[0]} does not match network Nin={self.Nin}. "
            "If this model needs direct context, build it with a CONTEXT input."
        )

    def _run_trials(self, trials, init=None, init_b=None, return_states=False,
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
        elif training and self.config.get('training_context_input', False):
            # Training mode: sample c from the configured distribution.
            contexts = self.sample_training_contexts(n_trials)
        else:
            # Default fallback
            contexts = torch.zeros(n_trials, device=self.device)

        direct_context_requested = (
            context_input is not None or
            (training and self.config.get('training_context_input', False))
        )

        if training and self.config.get('vta_training_context', False):
            vta_contexts = self.sample_vta_contexts(n_trials)
        else:
            vta_contexts = torch.zeros(n_trials, device=self.device)

        # Storage
        U = torch.zeros(self.Tmax, n_trials, self.Nin, device=self.device)
        Z = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)
        A = torch.zeros(self.Tmax, n_trials, self.n_actions, device=self.device)
        R = torch.zeros(self.Tmax, n_trials, device=self.device)
        M = torch.zeros(self.Tmax, n_trials, device=self.device)

        Policy_Values = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)
        Policy_D1_Pull = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)
        Policy_D2_Pull = torch.zeros(self.Tmax, n_trials, self.Nout, device=self.device)

        # Storage for continuous RPE signals (if RPE modulation is enabled)
        if self.use_rpe_modulation:
            RPE_continuous = torch.zeros(self.Tmax, n_trials, device=self.device)

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
            r_policy_mod = torch.zeros(self.Tmax, n_trials, self.policy_net.N, device=self.device)
            r_value = torch.zeros(self.Tmax, n_trials, self.baseline_net.N, device=self.device)

        if perf is None:
            perf = self.Performance()

        if progress_bar:
            progress_inc = max(int(n_trials / 50), 1)
            progress_half = 25 * progress_inc
            if progress_half > n_trials:
                progress_half = -1
            utils.println("[ ActorCriticTrainer.run_trials ] ")

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
                prev_dopamine_signal = torch.tensor([0.0], device=self.device)

                Z[t, n] = z_t
                Z_b[t, n] = z_t_b.squeeze() if z_t_b.dim() > 0 else z_t_b

                if return_states:
                    r_t_policy = self.policy_net.firing_rate(x_t)
                    r_policy[t, n] = r_t_policy
                    r_value[t, n] = self.baseline_net.firing_rate(x_t_b)

                # --- ACTION SELECTION ---
                r_t_for_action = self.policy_net.firing_rate(x_t.unsqueeze(0))

                # Compute modulation signal (either context or RPE-based)
                if self.use_rpe_modulation:
                    # At t=0, RPE is 0 (no previous value to compare)
                    modulation_signal = torch.tensor([0.0], device=self.device)
                else:
                    ctx_val = self._context_for_step(trial, t, contexts[n])
                    modulation_signal = torch.as_tensor([ctx_val], device=self.device)

                r_t_for_action = self._apply_opponent_modulation(
                    r_t_for_action,
                    modulation_signal
                )
                if return_states:
                    r_policy_mod[t, n] = r_t_for_action.squeeze(0).detach()
                logits_t = self.policy_net.output_layer(r_t_for_action, temperature=1.0, return_logits=True)
                
                Policy_Values[t, n] = logits_t.squeeze(0).detach()
                
                half_N = self.policy_net.N // 2
                Wout = self.policy_net.Wout
                
                d1_pull = torch.matmul(r_t_for_action[..., :half_N], Wout[:half_N, :])
                Policy_D1_Pull[t, n] = d1_pull.squeeze(0).detach()
                
                d2_pull = torch.matmul(r_t_for_action[..., half_N:], Wout[half_N:, :])
                Policy_D2_Pull[t, n] = d2_pull.squeeze(0).detach()
                
                z_t_np = torch.softmax(logits_t, dim=-1).squeeze(0).cpu().detach().numpy()
                
                z_t_np = z_t_np.reshape(self.Nout)
                a_t = self.rng.choice(self.Nout, p=z_t_np)
                A[t, n, a_t] = 1

                # Task step
                u_t_np, r_t, status = self.task.get_step(self.rng, self.dt, trial, t+1, a_t)
                u_t_tensor = torch.FloatTensor(u_t_np).to(self.device)
                
                ctx_val = self._context_for_step(trial, t, contexts[n])
                U[t, n] = self._format_task_input(u_t_tensor, ctx_val, direct_context_requested)
                    
                R[t, n] = r_t
                M[t, n] = 1

                # t > 0
                for t in range(1, self.Tmax):
                    if not status['continue']:
                        break

                    u_t = U[t-1, n:n+1]
                    q_t = Q[t, n:n+1]
                    x_t = x_t.unsqueeze(0)
                    recurrent_dopamine = prev_dopamine_signal if self.use_rpe_modulation else None
                    z_t, x_t = self.policy_net.step_t(
                        u_t, q_t, x_t, dopamine_signal=recurrent_dopamine
                    )
                    x_t = x_t.squeeze(0)
                    Z[t, n] = z_t

                    r_t_policy = self.policy_net.firing_rate(x_t)

                    if self.config.get('baseline_include_state', False):
                        u_t_b = torch.cat([U[t-1, n], r_t_policy, A[t-1, n]], dim=-1).unsqueeze(0)
                    else:
                        u_t_b = torch.cat([r_t_policy, A[t-1, n]], dim=-1).unsqueeze(0)
                    q_t_b = Q_b[t, n:n+1]
                    x_t_b = x_t_b.unsqueeze(0)
                    z_t_b, x_t_b = self.baseline_net.step_t(u_t_b, q_t_b, x_t_b)
                    x_t_b = x_t_b.squeeze(0)
                    
                    Z_b[t, n] = z_t_b.squeeze() if z_t_b.dim() > 0 else z_t_b

                    if return_states:
                        r_policy[t, n] = r_t_policy
                        r_value[t, n] = self.baseline_net.firing_rate(x_t_b)

                    # --- ACTION SELECTION ---
                    r_t_for_action = self.policy_net.firing_rate(x_t.unsqueeze(0))

                    # Compute modulation signal (either RPE-based or context-based)
                    if self.use_rpe_modulation:
                        v_t = Z_b[t, n]
                        v_tm1 = Z_b[t-1, n]

                        # RPE = r(t-1) + γ*V(t) - V(t-1)
                        # Pass timestep for phase-specific optostimulation
                        phase = self._get_task_phase(t)
                        rpe_signal = self._compute_rpe_signal(
                            R[t-1, n],
                            v_t,
                            v_tm1,
                            timestep=t,
                            phase=phase,
                            dopamine_offset=vta_contexts[n]
                        )
                        RPE_continuous[t, n] = rpe_signal
                        modulation_signal = rpe_signal.unsqueeze(0)
                        prev_dopamine_signal = modulation_signal.detach()
                    else:
                        ctx_val = self._context_for_step(trial, t, contexts[n])
                        modulation_signal = torch.as_tensor([ctx_val], device=self.device)

                    r_t_for_action = self._apply_opponent_modulation(
                        r_t_for_action,
                        modulation_signal
                    )
                    if return_states:
                        r_policy_mod[t, n] = r_t_for_action.squeeze(0).detach()
                    logits_t = self.policy_net.output_layer(r_t_for_action, temperature=1.0, return_logits=True)
                    
                    Policy_Values[t, n] = logits_t.squeeze(0).detach()
                    
                    half_N = self.policy_net.N // 2
                    Wout = self.policy_net.Wout
                    
                    d1_pull = torch.matmul(r_t_for_action[..., :half_N], Wout[:half_N, :])
                    Policy_D1_Pull[t, n] = d1_pull.squeeze(0).detach()
                    
                    d2_pull = torch.matmul(r_t_for_action[..., half_N:], Wout[half_N:, :])
                    Policy_D2_Pull[t, n] = d2_pull.squeeze(0).detach()
                    
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
                        
                        ctx_val = self._context_for_step(trial, t, contexts[n])
                        U[t, n] = self._format_task_input(u_t_tensor, ctx_val, direct_context_requested)
                            
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

        results = {
            'U': U, 'Q': Q, 'Q_b': Q_b, 'Z': Z, 'Z_b': Z_b,
            'A': A, 'R': R, 'M': M, 'perf': perf,
            'contexts': contexts,
            'vta_contexts': vta_contexts,
            'prob_l': prob_l, 'prob_r': prob_r,
            'size_l': size_l, 'size_r': size_r,
            'RPE_objective': RPE_objective,
            'RPE_subjective': RPE_subjective,
            'Policy_Values': Policy_Values,
            'Policy_D1_Pull': Policy_D1_Pull,
            'Policy_D2_Pull': Policy_D2_Pull,
        }
        if self.use_rpe_modulation:
            results['RPE_continuous'] = RPE_continuous  # Continuous RPE used for D1/D2 modulation
        if return_states:
            results['r_policy'] = r_policy
            results['r_policy_mod'] = r_policy_mod
            results['r_value'] = r_value

        return results

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
        >>> trainer = ActorCriticTrainer.load('model.pkl')
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

    def _context_for_step(self, trial, t, context_value):
        """Return the context value for a timestep, optionally limiting it to decision time."""
        if not self.config.get('context_decision_only', False):
            return context_value

        decision_epoch = trial.get('epochs', {}).get('decision') if trial is not None else None
        if decision_epoch is None:
            return context_value

        return context_value if t in decision_epoch else 0.0

    def _get_task_phase(self, timestep):
        """
        Determine task phase from timestep.
        Assumes gambling task structure: fixation (0-24), cue (25-49), decision (50+)
        """
        if timestep < 25:
            return 'fixation'
        elif timestep < 50:
            return 'cue'
        else:
            return 'decision'

    def _compute_rpe_signal(self, r_t, v_t, v_tm1, timestep=None, phase=None,
                            dopamine_offset=0.0):
        """
        Compute continuous RPE signal for D1/D2 modulation.

        RPE(t) = r(t) + γ*V(t) - V(t-1)

        This is the TD error that can be used as a tonic dopamine signal.

        Parameters
        ----------
        r_t : float or torch.Tensor
            Reward at time t
        v_t : torch.Tensor
            Value estimate at time t
        v_tm1 : torch.Tensor
            Value estimate at time t-1
        timestep : int, optional
            Current timestep (for phase-specific optostimulation)
        phase : str, optional
            Current task phase ('fixation', 'cue', 'decision')

        Returns
        -------
        rpe : torch.Tensor
            RPE signal, optionally manipulated and clamped for D1/D2 modulation
        """
        # Compute natural TD error.
        rpe = r_t + self.gamma * v_t - v_tm1

        # Convert natural RPE to dopamine units.
        rpe = rpe * self.rpe_modulation_gain

        # Add training-time VTA context in dopamine units.
        if not isinstance(dopamine_offset, torch.Tensor):
            dopamine_offset = torch.tensor(dopamine_offset, device=self.device)
        rpe = rpe + dopamine_offset

        # Apply optogenetic stimulation in dopamine units (inference only).
        if hasattr(self, 'opto_stim_offset') and hasattr(self, 'opto_stim_gain'):
            opto_offset = self.opto_stim_offset
            opto_gain = self.opto_stim_gain
            opto_phase = getattr(self, 'opto_stim_phase', 'all')

            # Check if stimulation should be applied to this phase
            apply_opto = (opto_phase == 'all' or
                         (phase is not None and opto_phase == phase))

            if apply_opto and (opto_offset != 0.0 or opto_gain != 1.0):
                # Apply optostimulation: RPE_opto = gain * RPE_DA + DA_offset
                rpe = opto_gain * rpe + opto_offset

        # Clamp final dopamine signal before D1/D2 gain modulation.
        rpe = torch.clamp(rpe, -self.rpe_modulation_clamp, self.rpe_modulation_clamp)

        return rpe

    def _apply_opponent_modulation(self, r, context_signal):
        """
        Intercepts network firing rates to apply D1/D2 multiplicative gain.
        Preserves Vanilla RL behavior if use_opponent_modulation is False.

        When use_rpe_modulation is True, context_signal contains the continuous RPE.

        BIOLOGICAL MECHANISM:
        - Positive dopamine/RPE: increases D1 excitability and decreases D2 excitability
        - Negative dopamine/RPE: decreases D1 excitability and increases D2 excitability
        """
        # Enable modulation if either opponent_modulation or rpe_modulation is active
        modulation_enabled = (self.config.get('use_opponent_modulation', False) or
                             self.use_rpe_modulation)

        if not modulation_enabled or context_signal is None:
            return r

        if context_signal.dim() == 0:
            context_signal = context_signal.unsqueeze(0)

        return self.policy_net._apply_dopamine_modulation(r, context_signal)
