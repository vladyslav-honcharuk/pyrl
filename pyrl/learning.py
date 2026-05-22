"""Actor, critic, training-loop, and diagnostic helpers."""
from collections import OrderedDict
import datetime
import sys

import numpy as np
import torch
import torch.optim as optim

from . import utils


class LearningMixin:
    def _train(self, savefile, recover=False):
        """Train the policy and baseline networks."""
        # Training parameters
        max_iter = self.config['max_iter']
        lr = self.config['lr']
        baseline_lr = self.config['baseline_lr']
        n_gradient = self.config['n_gradient']
        n_validation = self.config['n_validation']
        checkfreq = self.config['checkfreq']


        # Optimizers
        policy_optimizer = optim.Adam(self.policy_net.get_trainable_params(), lr=lr)
        baseline_optimizer = optim.Adam(self.baseline_net.get_trainable_params(), lr=baseline_lr)

        # Initialize training state
        if recover and hasattr(self, 'save'):
            iter_start = self.save['iter']

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
                        
                        Z_b_error = val_results['Z_b']
                        error = torch.sqrt(torch.sum((Z_b_error - V)**2 * val_results['M']) /
                                          torch.sum(val_results['M'])).item()
                        items['Prediction error'] = f'{error}'

                        # ==========================================
                        # CRITIC CAPACITY DIAGNOSTICS
                        # ==========================================
                        try:
                            crit_diag = self.diagnose_critic(n_trials=200)
                            items['V range'] = f"[{crit_diag['V_min']:.2f}, {crit_diag['V_max']:.2f}]"
                            items['Return range'] = f"[{crit_diag['Return_min']:.2f}, {crit_diag['Return_max']:.2f}]"
                            items['V coverage'] = f"{crit_diag['V_range_coverage']:.2%}"
                            items['V bias'] = f"{crit_diag['V_minus_Return_bias']:+.3f}"
                            items['V RMSE'] = f"{crit_diag['V_RMSE']:.3f}"
                            items['Terminal V vs R'] = (
                                f"{crit_diag['Terminal_V_mean']:+.3f} vs "
                                f"{crit_diag['Terminal_R_mean']:+.3f}"
                            )
                        except Exception as e:
                            items['Critic diag'] = f"Crash: {str(e)}"

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

                # Update policy network
                self._update_policy(train_results, policy_optimizer)

                # Learning rate decay (biological synaptic consolidation)
                # Formula: lr(t) = lr_0 / (1 + decay * t)
                if self.config.get('baseline_lr_decay', 0) > 0:
                    new_baseline_lr = baseline_lr / (1 + self.config['baseline_lr_decay'] * iter_)
                    for param_group in baseline_optimizer.param_groups:
                        param_group['lr'] = new_baseline_lr

                if self.config.get('lr_decay', 0) > 0:
                    new_policy_lr = lr / (1 + self.config['lr_decay'] * iter_)
                    for param_group in policy_optimizer.param_groups:
                        param_group['lr'] = new_policy_lr

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
        values : torch.Tensor, shape (T, B)
            Scalar value predictions at each timestep.
        mask : torch.Tensor, shape (T, B)
            Valid timesteps (1 = valid, 0 = invalid)
        gamma : float
            Discount factor

        Returns
        -------
        td_error : torch.Tensor, shape (T, B)
            Online TD error at each timestep
        """
        values = self._scalar_values(values)

        T, B = rewards.shape
        td_error = torch.zeros_like(rewards)

        for t in range(T - 1):
            # TD error = r(t) + γ*V(t+1)*M(t+1) - V(t)
            # This wipes out the hallucinated V(t+1) if the trial ended at t
            td_error[t] = rewards[t] + gamma * values[t+1] * mask[t+1] - values[t]

        # Last timestep: no future value
        td_error[T-1] = rewards[T-1] - values[T-1]

        # Apply mask
        td_error = td_error * mask

        return td_error

    def _update_baseline(self, results, optimizer):
        """Update the critic and refresh cached values used by the policy step."""
        loss, _ = self._baseline_loss(results)

        optimizer.zero_grad()
        loss.backward()

        grad_clip = self.config.get('baseline_grad_clip', None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.baseline_net.parameters(), grad_clip)

        optimizer.step()

        with torch.no_grad():
            _, z_all_fresh = self._baseline_loss(results)
        results['Z_b'] = z_all_fresh.detach()

    def _baseline_inputs(self, results):
        """Build critic inputs from policy rates, action, and optional task state."""
        inputs = [results['r_policy'], results['A']]
        if self.config.get('baseline_include_state', False):
            inputs.insert(0, results['U'])
        return torch.cat(inputs, dim=-1)

    def _baseline_forward(self, results, squeeze_scalar=True):
        """Run the critic over the stored trial batch and return all timesteps."""
        baseline_inputs = self._baseline_inputs(results)[:-1]
        batch_size = baseline_inputs.shape[1]
        x0 = self.baseline_net.x0.unsqueeze(0).expand(batch_size, -1)

        z_pred, states = self.baseline_net(
            baseline_inputs,
            results['Q_b'][:-1],
            x0
        )
        z0, _ = self.baseline_net.step_0(x0)

        if squeeze_scalar:
            if z0.dim() == 2 and z0.shape[-1] == 1:
                z0 = z0.squeeze(-1)
            if z_pred.dim() == 3 and z_pred.shape[-1] == 1:
                z_pred = z_pred.squeeze(-1)
        elif z0.dim() == 3:
            z0 = z0.squeeze(1)

        return torch.cat([z0.unsqueeze(0), z_pred], dim=0), states, x0

    def _scalar_values(self, values):
        """Return scalar critic values."""
        if values.dim() == 3:
            raise ValueError('Critic values must have shape (T, B).')
        return values

    def _critic_target(self, rewards, values, mask):
        """Return the critic target for the configured advantage mode."""
        mode = self.config.get('advantage_mode', 'mc')
        if mode == 'mc':
            return self._compute_returns(rewards, self.gamma)
        if mode == 'lambda':
            return self._compute_lambda_returns(
                rewards,
                values.detach(),
                mask,
                self.gamma,
                lam=self.config.get('td_lambda', 0.9)
            )

        target = torch.zeros_like(values)
        target[:-1] = rewards[:-1] + self.gamma * values[1:].detach() * mask[1:]
        target[-1] = rewards[-1]
        return target

    def _baseline_loss(self, results):
        """Compute critic loss with kappa-asymmetric residual scaling."""
        return self._standard_mse_loss(results)

    def _standard_mse_loss(self, results):
        """Mean-squared critic loss against kappa-scaled MC, TD(0), or lambda residuals."""
        R = results['R']
        M = results['M']
        z_all, states_b, x0 = self._baseline_forward(results, squeeze_scalar=True)

        with torch.no_grad():
            target = self._critic_target(R, z_all, M)

        delta = (target - z_all) * M
        delta_prime = self._apply_kappa_asymmetry(delta)
        n_valid = M.sum()
        if n_valid > 0:
            loss = torch.sum(delta_prime**2 * M) / n_valid
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = loss + self.baseline_net.get_regs(x0, states_b, M[:-1])
        return loss, z_all

    def _kappa_eta_tensors(self, delta):
        """Return actor/critic kappa etas broadcast to a scalar residual tensor."""
        if self.kappa_mode == 'per_neuron':
            eta_plus = self.eta_plus_neurons.mean()
            eta_minus = self.eta_minus_neurons.mean()
        else:
            eta_plus = torch.as_tensor(self.eta_plus, dtype=delta.dtype, device=delta.device)
            eta_minus = torch.as_tensor(self.eta_minus, dtype=delta.dtype, device=delta.device)

        return torch.ones_like(delta) * eta_plus, torch.ones_like(delta) * eta_minus

    def _apply_kappa_asymmetry(self, delta):
        """Scale positive and negative residuals by the hardwired kappa etas."""
        eta_plus, eta_minus = self._kappa_eta_tensors(delta)
        return torch.where(delta > 0, eta_plus * delta, eta_minus * delta)

    def _hill_occupancy(self, da, ec50, hill_n):
        da = torch.clamp(da, min=1e-6)
        ec50_t = torch.as_tensor(ec50, dtype=da.dtype, device=da.device).clamp_min(1e-6)
        da_n = da ** hill_n
        return da_n / (da_n + ec50_t ** hill_n)

    def _compute_learning_etas(self, context_signal, delta):
        """Compute eta_plus/eta_minus for dopamine-biased policy learning."""
        signal = torch.clamp(context_signal, -1.0, 1.0) * 0.9

        if self.config.get('dopamine_learning_modulation_mode', 'linear') == 'hill':
            base_da = float(self.config.get('dopamine_hill_base_da', 1.0))
            da_range = float(self.config.get('dopamine_hill_da_range', 1.0))
            hill_n = float(self.config.get('dopamine_hill_coefficient', 1.0))
            scale = float(self.config.get('dopamine_hill_gain_scale', 2.0))

            da = torch.clamp(base_da + da_range * signal, min=1e-4)
            da0 = torch.as_tensor(base_da, dtype=signal.dtype, device=signal.device).clamp_min(1e-4)

            occ_D1 = self._hill_occupancy(da, self.config.get('dopamine_hill_ec50_d1', 1.0), hill_n)
            occ_D2 = self._hill_occupancy(da, self.config.get('dopamine_hill_ec50_d2', 0.07), hill_n)
            occ0_D1 = self._hill_occupancy(da0, self.config.get('dopamine_hill_ec50_d1', 1.0), hill_n)
            occ0_D2 = self._hill_occupancy(da0, self.config.get('dopamine_hill_ec50_d2', 0.07), hill_n)

            eta_plus = 1.0 + scale * (occ_D1 - occ0_D1)
            eta_minus = 1.0 - scale * (occ_D2 - occ0_D2)
        else:
            eta_plus = 1.0 + signal
            eta_minus = 1.0 - signal

        eta_min = float(self.config.get('dopamine_learning_eta_min', 0.1))
        eta_max = float(self.config.get('dopamine_learning_eta_max', 1.9))
        eta_plus = torch.clamp(eta_plus, eta_min, eta_max)
        eta_minus = torch.clamp(eta_minus, eta_min, eta_max)

        return eta_plus.unsqueeze(0).expand_as(delta), eta_minus.unsqueeze(0).expand_as(delta)

    def _compute_policy_advantage(self, results):
        """Compute the scalar advantage used by the actor update."""
        R = results['R']
        M = results['M']
        baseline = results['Z_b']
        mode = self.config.get('advantage_mode', 'mc')

        with torch.no_grad():
            if mode == 'mc':
                returns = self._compute_returns(R, self.gamma)
                return (returns - baseline) * M
            if mode == 'gae':
                return self._compute_gae(
                    R,
                    baseline,
                    M,
                    self.gamma,
                    lam=self.config.get('gae_lambda', 0.95)
                )
            return self._compute_online_td_error(R, baseline, M, self.gamma)

    def _policy_learning_context(self, results):
        """Return trial-level and optional timestep-level dopamine/context signals."""
        if self.use_rpe_modulation and 'RPE_continuous' in results:
            rpe = results['RPE_continuous']
            mask = results['M']
            valid_steps = mask.sum(dim=0).clamp(min=1.0)
            context = (rpe * mask).sum(dim=0) / valid_steps
            return context, rpe

        return results.get('contexts'), None

    def _modulate_policy_rates(self, rates, context_signal, context_timeseries=None, start=0):
        """Apply either trial-constant context or timestep-specific RPE modulation."""
        if context_timeseries is None:
            return self._apply_opponent_modulation(rates, context_signal)

        if rates.dim() == 2:
            return self._apply_opponent_modulation(rates, context_timeseries[start])

        modulated = [
            self._apply_opponent_modulation(rates[t], context_timeseries[start + t])
            for t in range(rates.shape[0])
        ]
        return torch.stack(modulated, dim=0)

    def _apply_actor_weight_learning_modulation(self):
            """
            Multiply actor output-weight gradients by current actor weight strength.

            This optional three-factor modifier keeps the existing policy-gradient
            update, then adds a postsynaptic/current-actor-weight factor on Wout.
            """
            if not self.config.get('actor_weight_learning_modulation', False):
                return

            Wout = getattr(self.policy_net, 'Wout', None)
            if Wout is None or Wout.grad is None:
                return

            floor = float(self.config.get('actor_weight_learning_floor', 0.05))
            max_factor = self.config.get('actor_weight_learning_max', 2.0)
            normalize = self.config.get('actor_weight_learning_normalize', True)

            with torch.no_grad():
                strength = torch.abs(Wout.detach()) + floor
                if normalize:
                    strength = strength / strength.mean().clamp_min(1e-8)
                if max_factor is not None:
                    strength = torch.clamp(strength, max=float(max_factor))
                Wout.grad.mul_(strength)

    def _update_policy(self, results, optimizer):
            """Update policy network using context- or RPE-modulated advantages."""
            U = results['U']
            A = results['A']
            M = results['M']
            Q_trimmed = results['Q'][:-1]

            delta = self._compute_policy_advantage(results)
            context_signal, context_timeseries = self._policy_learning_context(results)

            if context_signal is not None:
                eta_plus, eta_minus = self._compute_learning_etas(context_signal, delta)
            else:
                eta_plus = torch.ones_like(delta)
                eta_minus = torch.ones_like(delta)

            kappa_eta_plus, kappa_eta_minus = self._kappa_eta_tensors(delta)
            eta_plus = eta_plus * kappa_eta_plus
            eta_minus = eta_minus * kappa_eta_minus

            # 4. Asymmetrically scale the advantage based on the sign of the RPE
            scaled_advantage = torch.where(delta > 0, eta_plus * delta, eta_minus * delta)

            # 5. Forward pass through policy network
            U_trimmed = U[:-1]
            B_size = U_trimmed.shape[1]
            x0 = self.policy_net.x0.unsqueeze(0).expand(B_size, -1)
            recurrent_dopamine = None
            if context_timeseries is not None:
                recurrent_dopamine = context_timeseries[:-1]

            _, states = self.policy_net(
                U_trimmed,
                Q_trimmed,
                x0,
                dopamine_signal=recurrent_dopamine
            )

            r_0 = self.policy_net.firing_rate(x0)
            r_0 = self._modulate_policy_rates(
                r_0,
                context_signal,
                context_timeseries=context_timeseries,
                start=0
            )
            if self.policy_dropout is not None:
                r_0 = self.policy_dropout(r_0)
            log_z_0 = self.policy_net.log_output(r_0)

            r_pred = self.policy_net.firing_rate(states)
            r_pred = self._modulate_policy_rates(
                r_pred,
                context_signal,
                context_timeseries=context_timeseries,
                start=1
            )
            if self.policy_dropout is not None:
                r_pred = self.policy_dropout(r_pred)
            log_z_pred = self.policy_net.log_output(r_pred)

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

            loss.backward()

            self._apply_actor_weight_learning_modulation()

            grad_clip = self.config.get('grad_clip', None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), grad_clip)

            optimizer.step()

    def _compute_gae(self, rewards, values, mask, gamma, lam=0.95):
        """
        Generalized Advantage Estimation (Schulman et al. 2016).

        GAE_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
        where δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        λ=1.0  recovers Monte Carlo (high variance, no bias)
        λ=0.0  recovers TD(0)        (low variance, high bias)
        λ=0.95 is the PPO/A2C sweet spot for sparse-reward episodic tasks.

        Parameters
        ----------
        rewards : (T, B)
        values  : (T, B)
        mask    : (T, B) — 1 for valid timesteps, 0 after trial ends
        gamma   : float
        lam     : float in [0, 1]

        Returns
        -------
        advantages : (T, B)
        """
        values = self._scalar_values(values)

        T, B = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)

        for t in reversed(range(T)):
            if t == T - 1:
                # No bootstrap past the end of the buffer
                next_value = torch.zeros(B, device=rewards.device)
                next_nonterminal = torch.zeros(B, device=rewards.device)
            else:
                # mask[t+1] = 0 means trial ended, so next_value contributes nothing
                next_value = values[t + 1]
                next_nonterminal = mask[t + 1]

            delta_t = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta_t + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae

        # Mask out padding timesteps
        advantages = advantages * mask
        return advantages

    def _diagnose_critic(self, n_trials=200):
        """
        Quick diagnostic: does the critic's output range cover the return range?
        Print this every checkfreq during training to catch a saturating/collapsing critic.

        Returns a dict you can also log into your `items` display in train().
        """
        with torch.no_grad():
            results = self.run_trials(n_trials, training=False)
            R = results['R']
            M = results['M']
            Z_b = results['Z_b']

            returns = self._compute_returns(R, self.gamma)

            # Only consider valid (non-padded) timesteps
            valid = M.bool()
            ret_valid = returns[valid]
            v_valid = Z_b[valid]

            # Per-trial terminal return (the actual reward each trial earned)
            terminal_R = (R * M).sum(dim=0)  # (B,)

            # Terminal-step V: V at the last valid timestep of each trial
            # Find last valid index per trial
            last_idx = M.sum(dim=0).long() - 1  # (B,)
            last_idx = last_idx.clamp(min=0)
            terminal_V = Z_b[last_idx, torch.arange(M.shape[1])]

            diag = {
                'V_min': v_valid.min().item(),
                'V_max': v_valid.max().item(),
                'V_mean': v_valid.mean().item(),
                'V_std': v_valid.std().item(),
                'Return_min': ret_valid.min().item(),
                'Return_max': ret_valid.max().item(),
                'Return_mean': ret_valid.mean().item(),
                'Return_std': ret_valid.std().item(),
                'Terminal_R_min': terminal_R.min().item(),
                'Terminal_R_max': terminal_R.max().item(),
                'Terminal_R_mean': terminal_R.mean().item(),
                'Terminal_V_mean': terminal_V.mean().item(),
                # Coverage ratio: does V's range cover the return range?
                'V_range_coverage': (v_valid.max() - v_valid.min()).item() /
                                    max((ret_valid.max() - ret_valid.min()).item(), 1e-6),
                # Bias: is V systematically off from returns?
                'V_minus_Return_bias': (v_valid - ret_valid).mean().item(),
                # RMSE
                'V_RMSE': torch.sqrt(((v_valid - ret_valid) ** 2).mean()).item(),
            }
            return diag

    def _compute_lambda_returns(self, rewards, values, mask, gamma, lam=0.9):
        """
        λ-return: the TD(λ) target.
        
        Biologically: this is what you get when each synapse keeps an
        eligibility trace decaying at rate γλ, and the dopamine RPE
        signal updates all eligible synapses. The offline λ-return is
        mathematically equivalent to the forward view of TD(λ).
        
        G^λ_t = r_t + γ[(1-λ)V(s_{t+1}) + λ G^λ_{t+1}]
        
        λ=0.0 → TD(0) (synapses have no memory beyond one step)
        λ=0.9 → eligibility decays over ~10 steps (biologically realistic;
                matches Yagishita et al. 2014 ~2s synaptic tag window with dt=10ms)
        λ=1.0 → Monte Carlo (synapses remember the entire episode)
        """
        T, B = rewards.shape
        returns = torch.zeros_like(rewards)
        last_return = torch.zeros(B, device=rewards.device)
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(B, device=rewards.device)
                next_nonterminal = torch.zeros(B, device=rewards.device)
            else:
                next_value = values[t + 1]
                next_nonterminal = mask[t + 1]
            
            bootstrap = (1 - lam) * next_value + lam * last_return
            last_return = rewards[t] + gamma * next_nonterminal * bootstrap
            returns[t] = last_return
        
        return returns
