"""Actor, critic, training-loop, and diagnostic helpers."""
from collections import OrderedDict
import datetime
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from . import utils


class LearningMixin:
    def _policy_aux_params(self):
        """Return trainable auxiliary policy-feedback parameters, if any."""
        params = []
        for module_name in ('value_pop_proj_d1', 'value_pop_proj_d2'):
            module = getattr(self, module_name, None)
            if module is not None:
                params.extend(list(module.parameters()))
        return params

    def _value_pop_feedback_state(self):
        """Return checkpointable state for critic-population feedback modules."""
        if getattr(self, 'value_pop_proj_d1', None) is None:
            return None
        return {
            'd1': self.value_pop_proj_d1.state_dict(),
            'd2': self.value_pop_proj_d2.state_dict(),
        }

    def _policy_input_win_norm(self, input_index):
        """Return the norm of one policy input row in Win."""
        Win = getattr(self.policy_net, 'Win', None)
        if Win is None:
            return None

        with torch.no_grad():
            return torch.linalg.vector_norm(Win[input_index]).item()

    def _policy_named_input_win_norms(self):
        """Return selected policy-input Win diagnostics keyed by readable labels."""
        inputs = self.config.get('inputs', {})
        if not isinstance(inputs, dict):
            return OrderedDict()

        items = OrderedDict()

        fixation_idx = inputs.get('FIXATION')
        if fixation_idx is not None:
            fixation_norm = self._policy_input_win_norm(fixation_idx)
            if fixation_norm is not None:
                items['Fixation Win norm'] = f'{fixation_norm:.6f}'

        for prefix, label in (
            ('LEFT_', 'Left color mean Win norm'),
            ('RIGHT_', 'Right color mean Win norm'),
        ):
            indices = [
                index for name, index in inputs.items()
                if str(name).startswith(prefix)
            ]
            if indices:
                norms = [self._policy_input_win_norm(index) for index in indices]
                norms = [value for value in norms if value is not None]
                if norms:
                    items[label] = f'{(sum(norms) / len(norms)):.6f}'

        feedback_win_norm = self._policy_value_feedback_win_norm()
        if feedback_win_norm is not None:
            items['Value feedback Win norm'] = f'{feedback_win_norm:.6f}'

        return items

    def _policy_value_feedback_win_norm(self):
        """Return the norm of the policy Win row for the scalar value-feedback input."""
        if not getattr(self, 'policy_value_feedback', False):
            return None

        Win = getattr(self.policy_net, 'Win', None)
        if Win is None:
            return None

        with torch.no_grad():
            return torch.linalg.vector_norm(Win[-1]).item()

    def _recent_rpe_modulation_diagnostics(self, results):
        """Return readable diagnostics for previous-trial-RPE-driven D1/D2 modulation."""
        if not self.config.get('use_recent_rpe_modulation', False):
            return OrderedDict()

        items = OrderedDict()
        items['Recent RPE state'] = f"{float(getattr(self, 'recent_rpe_state', 0.0)):+.6f}"
        items['Recent RPE decay'] = f"{float(getattr(self, 'recent_rpe_decay', 0.0)):.3f}"
        items['Recent RPE gain'] = f"{float(getattr(self, 'recent_rpe_gain', 0.0)):.3f}"
        items['Recent RPE phase'] = str(getattr(self, 'recent_rpe_phase', 'decision'))

        if 'Policy_D1_Pull' in results and 'Policy_D2_Pull' in results:
            mask = results['M']
            d1 = results['Policy_D1_Pull']
            d2 = results['Policy_D2_Pull']
            expand_mask = mask.unsqueeze(-1)
            valid = torch.sum(expand_mask).item()
            if valid > 0:
                mean_abs_d1 = (torch.sum(torch.abs(d1) * expand_mask) / torch.sum(expand_mask)).item()
                mean_abs_d2 = (torch.sum(torch.abs(d2) * expand_mask) / torch.sum(expand_mask)).item()
                ratio = mean_abs_d1 / max(mean_abs_d2, 1e-8)
                items['Recent RPE mean |D1|'] = f'{mean_abs_d1:.6f}'
                items['Recent RPE mean |D2|'] = f'{mean_abs_d2:.6f}'
                items['Recent RPE |D1|/|D2|'] = f'{ratio:.6f}'

        if 'Recent_RPE_Modulation' in results:
            value_mod = results['Recent_RPE_Modulation']
            mask = results['M']
            valid_steps = torch.sum(mask).item()
            if valid_steps > 0:
                mean_vmod = (torch.sum(value_mod * mask) / torch.sum(mask)).item()
                items['Recent RPE mean signal'] = f'{mean_vmod:+.6f}'

        return items

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
        policy_params = list(self.policy_net.get_trainable_params()) + self._policy_aux_params()
        policy_optimizer = optim.Adam(policy_params, lr=lr)
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
            best_value_pop_feedback_state = self.save.get('best_value_pop_feedback_state')

            training_history = self.save['training_history']
            trials_tot = self.save['trials_tot']
            self.recent_rpe_state = float(self.save.get('recent_rpe_state', 0.0))

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
            best_value_pop_feedback_state = self._value_pop_feedback_state()
            training_history = []
            trials_tot = 0
            self.recent_rpe_state = 0.0

        # Training loop
        if hasattr(self.task, 'start_session'):
            self.task.start_session(self.rng)

        tstart = datetime.datetime.now()

        try:
            for iter_ in range(iter_start, max_iter + 1):
                if self.config.get('use_value_modulation', False):
                    if iter_ < self.value_modulation_start_iter:
                        self.value_modulation_scale = 0.0
                    elif self.value_modulation_ramp_iters > 0:
                        ramp_progress = (
                            (iter_ - self.value_modulation_start_iter) /
                            float(self.value_modulation_ramp_iters)
                        )
                        self.value_modulation_scale = float(np.clip(ramp_progress, 0.0, 1.0))
                    else:
                        self.value_modulation_scale = 1.0
                    self.value_modulation_active = self.value_modulation_scale > 0.0

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
                        val_results = self.run_trials(
                            val_trials,
                            progress_bar=True,
                            return_states=True
                        )
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
                            best_value_pop_feedback_state = self._value_pop_feedback_state()
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
                            'value_pop_feedback_state': self._value_pop_feedback_state(),
                            'best_iter': best_iter,
                            'best_reward': best_reward,
                            'best_perf': best_perf,
                            'best_policy_params': best_policy_params,
                            'best_baseline_params': best_baseline_params,
                            'best_value_pop_feedback_state': best_value_pop_feedback_state,
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
                            'kappa': self.kappa,  # Save scalar kappa value
                            'recent_rpe_state': self.recent_rpe_state,
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

                        items.update(self._policy_named_input_win_norms())
                        items.update(self._recent_rpe_modulation_diagnostics(val_results))

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

        if self.config.get('use_value_modulation', False) and 'Value_Modulation' in results:
            value_mod = results['Value_Modulation']
            mask = results['M']
            valid_steps = mask.sum(dim=0).clamp(min=1.0)
            context = (value_mod * mask).sum(dim=0) / valid_steps
            return context, value_mod

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

    def _choice_action_indices(self):
        """Return policy action indices representing value-bearing alternatives."""
        choice_indices = [
            index for name, index in self.config['actions'].items()
            if str(name).upper().startswith('CHOOSE')
        ]
        if not choice_indices:
            raise ValueError(
                "pathway_specific_plasticity requires actions named CHOOSE-*"
            )
        return choice_indices

    def _value_choice_mask(self, results, choice_indices=None):
        """Return timesteps where a non-aborted value-bearing choice was made."""
        if choice_indices is None:
            choice_indices = self._choice_action_indices()
        A = results['A']
        R = results['R']
        M = results['M']
        chose = torch.sum(A[..., choice_indices], dim=-1) > 0
        # In this task, choice rewards are nonnegative; a negative CHOOSE
        # reward identifies a premature fixation/stimulus abort.
        valid_choice = chose & (R >= 0)
        return valid_choice.to(dtype=M.dtype) * M

    def _opal_choice_advantages(self, delta, eta_plus, eta_minus):
        """
        Return policy-objective coefficients for OpAL choice plasticity.

        D2 already enters the logits as ``-N``. It therefore uses a positive
        delta coefficient here: autograd through the subtraction turns
        positive prediction errors into reduced N and negative errors into
        increased N.
        """
        alpha_d1 = float(self.config.get('opal_alpha_d1', 1.0))
        alpha_d2 = float(self.config.get('opal_alpha_d2', 1.0))
        d1_negative_scale = float(self.config.get('opal_d1_negative_scale', 1.0))
        d2_positive_scale = float(self.config.get('opal_d2_positive_scale', 1.0))

        positive_mask = (delta > 0).to(delta.dtype)
        negative_mask = (delta < 0).to(delta.dtype)
        zero_mask = 1.0 - positive_mask - negative_mask

        d1_scale = positive_mask + d1_negative_scale * negative_mask + zero_mask
        d2_scale = negative_mask + d2_positive_scale * positive_mask + zero_mask
        return (
            alpha_d1 * d1_scale * eta_plus * delta,
            alpha_d2 * d2_scale * eta_minus * delta,
        )

    def _apply_actor_weight_learning_modulation(self):
            """
            Multiply actor output-weight gradients by current actor weight strength.

            This optional three-factor modifier keeps the existing policy-gradient
            update, then adds a current-actor-weight factor on Wout. Positive
            OpAL readouts use their effective G/N strengths, softplus(Wout_raw).
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
                if self.policy_net.config.get('positive_policy_readout', False):
                    strength = self.policy_net._effective_output_weights().detach() + floor
                else:
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

            r_0_unmod = self.policy_net.policy_rates(x0)
            r_0 = self._modulate_policy_rates(
                r_0_unmod,
                context_signal,
                context_timeseries=context_timeseries,
                start=0
            )
            if self.policy_dropout is not None:
                r_0 = self.policy_dropout(r_0)

            r_pred_unmod = self.policy_net.policy_rates(states)
            r_pred = self._modulate_policy_rates(
                r_pred_unmod,
                context_signal,
                context_timeseries=context_timeseries,
                start=1
            )
            if self.policy_dropout is not None:
                r_pred = self.policy_dropout(r_pred)

            logit_signal_0 = context_timeseries[0] if context_timeseries is not None else context_signal
            logit_signal_t = context_timeseries[1:] if context_timeseries is not None else context_signal

            def chosen_logpi(pathway_gradient=None):
                log_z0 = self.policy_net.log_output(
                    r_0,
                    pathway_gradient=pathway_gradient,
                    control_r=r_0_unmod,
                    modulation_signal=logit_signal_0
                )
                log_z = self.policy_net.log_output(
                    r_pred,
                    pathway_gradient=pathway_gradient,
                    control_r=r_pred_unmod,
                    modulation_signal=logit_signal_t
                )
                logpi_0 = torch.sum(log_z0 * A[0], dim=-1)
                logpi_t = torch.sum(log_z * A[1:], dim=-1)
                return torch.cat([logpi_0.unsqueeze(0), logpi_t], dim=0)

            if self.config.get('pathway_specific_plasticity', False):
                if not self.policy_net.uses_opponent_readout():
                    raise ValueError(
                        "pathway_specific_plasticity requires use_opponent_modulation "
                        "or positive_policy_readout"
                    )

                # Both pathways receive every PE. Because D2 is subtracted
                # from logits, the same PE coefficient induces the opposite
                # N-weight update required by OpAL.
                d1_advantage, d2_advantage = self._opal_choice_advantages(
                    delta, eta_plus, eta_minus
                )
                choice_indices = self._choice_action_indices()
                non_choice_indices = [
                    i for i in range(self.n_actions) if i not in choice_indices
                ]
                if len(non_choice_indices) != 1:
                    raise ValueError(
                        "pathway_specific_plasticity currently requires exactly "
                        "one control action outside CHOOSE-*"
                )
                control_index = non_choice_indices[0]
                choice_mask = self._value_choice_mask(results, choice_indices)

                def policy_logits(pathway_gradient=None):
                    logits_0 = self.policy_net.output_layer(
                        r_0,
                        return_logits=True,
                        pathway_gradient=pathway_gradient,
                        control_r=r_0_unmod,
                        modulation_signal=logit_signal_0
                    )
                    logits_t = self.policy_net.output_layer(
                        r_pred,
                        return_logits=True,
                        pathway_gradient=pathway_gradient,
                        control_r=r_pred_unmod,
                        modulation_signal=logit_signal_t
                    )
                    return torch.cat([logits_0.unsqueeze(0), logits_t], dim=0)

                def conditional_choice_logpi(pathway_gradient=None):
                    choice_logits = policy_logits(pathway_gradient)[..., choice_indices]
                    log_choice = F.log_softmax(choice_logits, dim=-1)
                    return torch.sum(log_choice * A[..., choice_indices], dim=-1)

                logits = policy_logits()
                log_norm = torch.logsumexp(logits, dim=-1)
                chose = torch.sum(A[..., choice_indices], dim=-1) > 0
                log_p_choose = torch.logsumexp(logits[..., choice_indices], dim=-1) - log_norm
                log_p_control = logits[..., control_index] - log_norm
                timing_logpi = torch.where(chose, log_p_choose, log_p_control)

                # Keep action-timing/control learning ordinary. Restrict opponent
                # plasticity to which value alternative is selected once choosing.
                timing_weighted_logpi = timing_logpi * scaled_advantage * M
                choice_weighted_logpi = (
                    conditional_choice_logpi('d1') * d1_advantage
                    + conditional_choice_logpi('d2') * d2_advantage
                    + conditional_choice_logpi('bias') * scaled_advantage
                ) * choice_mask
                weighted_logpi = timing_weighted_logpi + choice_weighted_logpi
            else:
                weighted_logpi = chosen_logpi() * scaled_advantage * M

            # 7. REINFORCE objective with dopamine/context-scaled plasticity
            J = torch.sum(weighted_logpi) / B_size

            reg = self.policy_net.get_regs(x0, states, M[:-1])
            r_readout = torch.cat([r_0.unsqueeze(0), r_pred], dim=0)
            reg = reg + self.policy_net.get_readout_regs(r_readout, M)
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
