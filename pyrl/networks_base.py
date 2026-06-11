"""
Base class for recurrent neural networks in PyTorch.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np


class RecurrentNetwork(nn.Module):
    """Base class for recurrent neural networks."""

    def __init__(self, network_type, name=''):
        super().__init__()
        self.network_type = network_type
        self.network_name = network_type
        if name:
            self.network_name += f'-{name}'

        self.config = {}
        self.masks = {}
        self._fixed_params = []

    @property
    def noise_dim(self):
        """Dimension of noise input."""
        return self.N

    def get_trainable_params(self):
        """Get list of trainable parameters (excluding fixed ones)."""
        trainable = []
        for name, param in self.named_parameters():
            if not any(name.startswith(fixed) for fixed in self._fixed_params):
                trainable.append(param)
        return trainable

    def get_state_dict_numpy(self):
        """Get state dict with numpy arrays instead of tensors."""
        return OrderedDict([
            (k, v.detach().cpu().numpy())
            for k, v in self.state_dict().items()
        ])

    def firing_rate(self, x):
        """Convert states to firing rates."""
        return torch.tanh(x) 

    def policy_rates(self, x):
        """Return rates used by the output readout."""
        return self.firing_rate(x)

    def _policy_readout_rates(self, r):
        """
        Map activity already prepared for the policy readout to the exact
        quantity consumed by the output layer.

        The base implementation is identity. Networks with a distinct
        readout-space transform can override this helper.
        """
        return r

    def uses_opponent_readout(self):
        """Return whether the output readout subtracts the second pathway."""
        return self.config.get('use_opponent_modulation', False)

    def control_action_index(self):
        """
        Return the single non-CHOOSE action index when configured.

        This is used to keep the control/wait action on an unmodulated
        readout path while still allowing dopamine to reshape the value-bearing
        CHOOSE actions.
        """
        if not self.config.get('exclude_control_action_from_dopamine_modulation', False):
            return None

        actions = self.config.get('actions')
        if not actions:
            return None

        choice_indices = [
            index for name, index in actions.items()
            if str(name).upper().startswith('CHOOSE')
        ]
        non_choice_indices = [
            i for i in range(len(actions)) if i not in choice_indices
        ]
        if len(non_choice_indices) != 1:
            raise ValueError(
                "exclude_control_action_from_dopamine_modulation requires exactly "
                "one control action outside CHOOSE-*"
            )
        return non_choice_indices[0]

    def _setup_dopamine_sensitivity(self, seed=1, params=None):
        """Initialize per-neuron dopamine receptor sensitivity."""
        enabled = self.config.get('dopamine_heterogeneous_sensitivity', False)
        learned = self.config.get('dopamine_sensitivity_learned', False)

        if params is not None and 'dopamine_sensitivity' in params:
            sensitivity = torch.FloatTensor(params['dopamine_sensitivity'])
        elif enabled:
            low = self.config.get('dopamine_sensitivity_min', 0.3)
            high = self.config.get('dopamine_sensitivity_max', 1.0)
            if high < low:
                raise ValueError("dopamine_sensitivity_max must be >= dopamine_sensitivity_min")

            gen = torch.Generator()
            gen.manual_seed(int(self.config.get('dopamine_sensitivity_seed', seed)))
            sensitivity = low + (high - low) * torch.rand(self.N, generator=gen)
        else:
            sensitivity = torch.ones(self.N)

        if learned:
            self.dopamine_sensitivity = nn.Parameter(sensitivity)
        else:
            self.register_buffer('dopamine_sensitivity', sensitivity)

    def _setup_dopamine_bias(self, params=None):
        """Initialize dopamine-dependent per-neuron current/bias."""
        enabled = self.config.get('dopamine_bias_enabled', False)
        learned = self.config.get('dopamine_bias_learned', True)

        if params is not None and 'dopamine_bias' in params:
            bias = torch.FloatTensor(params['dopamine_bias'])
        else:
            init = self.config.get('dopamine_bias_init', 0.0) if enabled else 0.0
            bias = torch.full((self.N,), float(init))

        if learned and enabled:
            self.dopamine_bias = nn.Parameter(bias)
        else:
            self.register_buffer('dopamine_bias', bias)

    def _setup_value_modulation(self, params=None):
        """Initialize optional learned affine couplings from scalar value to D1/D2 modulation."""
        enabled = self.config.get('use_value_modulation', False)
        shared = self.config.get('use_value_modulation_shared_gain', False)
        if params is not None and 'value_modulation_shared_weight' in params:
            shared_weight = torch.FloatTensor(params['value_modulation_shared_weight'])
            shared_bias = torch.FloatTensor(params['value_modulation_shared_bias'])
        elif params is not None and 'value_modulation_d1_weight' in params:
            d1_weight = torch.FloatTensor(params['value_modulation_d1_weight'])
            d1_bias = torch.FloatTensor(params['value_modulation_d1_bias'])
            d2_weight = torch.FloatTensor(params['value_modulation_d2_weight'])
            d2_bias = torch.FloatTensor(params['value_modulation_d2_bias'])
        else:
            if shared:
                shared_weight = torch.tensor([0.0])
                shared_bias = torch.tensor([0.0])
            else:
                d1_weight = torch.tensor([0.0])
                d1_bias = torch.tensor([0.0])
                d2_weight = torch.tensor([0.0])
                d2_bias = torch.tensor([0.0])

        if shared:
            if enabled:
                self.value_modulation_shared_weight = nn.Parameter(shared_weight)
                self.value_modulation_shared_bias = nn.Parameter(shared_bias)
            else:
                self.register_buffer('value_modulation_shared_weight', shared_weight)
                self.register_buffer('value_modulation_shared_bias', shared_bias)
        else:
            if enabled:
                self.value_modulation_d1_weight = nn.Parameter(d1_weight)
                self.value_modulation_d1_bias = nn.Parameter(d1_bias)
                self.value_modulation_d2_weight = nn.Parameter(d2_weight)
                self.value_modulation_d2_bias = nn.Parameter(d2_bias)
            else:
                self.register_buffer('value_modulation_d1_weight', d1_weight)
                self.register_buffer('value_modulation_d1_bias', d1_bias)
                self.register_buffer('value_modulation_d2_weight', d2_weight)
                self.register_buffer('value_modulation_d2_bias', d2_bias)

    def _value_modulation_drive(self, value_signal):
        """Return learned signed D1/D2 drives from scalar critic value."""
        signal = value_signal
        if signal.dim() == 0:
            signal = signal.unsqueeze(0)
        if self.config.get('use_value_modulation_shared_gain', False):
            signal = signal.to(dtype=self.value_modulation_shared_weight.dtype)
            shared_drive = signal * self.value_modulation_shared_weight + self.value_modulation_shared_bias
            d1_drive = shared_drive
            d2_drive = shared_drive
        else:
            signal = signal.to(dtype=self.value_modulation_d1_weight.dtype)
            d1_drive = signal * self.value_modulation_d1_weight + self.value_modulation_d1_bias
            d2_drive = signal * self.value_modulation_d2_weight + self.value_modulation_d2_bias
        return d1_drive, d2_drive

    def value_modulation_context_signal(self, value_signal):
        """Return a signed scalar summary of learned value modulation for training updates."""
        d1_drive, d2_drive = self._value_modulation_drive(value_signal)
        return 0.5 * (d1_drive + d2_drive)

    def step_0(self, x0=None, temperature=None):
        """
        Initial step (t=0).

        Parameters
        ----------
        x0 : tensor, optional
            Initial state. If None, uses learned initial state.
        temperature : tensor, optional
            Temperature for softmax output (policy networks only).
            If None, uses default temperature (1.0).

        Returns
        -------
        z : tensor
            Output (policy or value).
        x : tensor
            State.
        """
        if x0 is None:
            x0 = self.x0

        r = self.policy_rates(x0)
        z = self.output_layer(r, temperature=temperature)
        return z, x0

    def _apply_dopamine_modulation(self, r, dopamine_signal=None):
        """Apply push-pull D1/D2 gain modulation to firing rates."""
        if dopamine_signal is None:
            return r

        if dopamine_signal.dim() == 0:
            dopamine_signal = dopamine_signal.unsqueeze(0)

        mode = getattr(self, 'inference_pathway_mode', 'symmetric')
        if mode == 'symmetric':
            signal = torch.clamp(dopamine_signal, -0.9, 0.9)
        else:
            signal = torch.clamp(dopamine_signal, 0.0, 2.0)
        sensitivity = torch.clamp(self.dopamine_sensitivity.to(r.device), min=0.0)
        half_N = r.shape[-1] // 2
        sens_D1 = sensitivity[:half_N]
        sens_D2 = sensitivity[half_N:]

        if r.dim() == 3:
            signal = signal.view(1, -1, 1)
            sens_D1 = sens_D1.view(1, 1, -1)
            sens_D2 = sens_D2.view(1, 1, -1)
        elif r.dim() == 2:
            signal = signal.view(-1, 1)
            sens_D1 = sens_D1.view(1, -1)
            sens_D2 = sens_D2.view(1, -1)
        elif r.dim() == 1:
            signal = signal.view(1)
            sens_D1 = sens_D1.view(-1)
            sens_D2 = sens_D2.view(-1)

        if mode == 'symmetric' and self.config.get('dopamine_modulation_mode', 'linear') == 'hill':
            gain_D1, gain_D2 = self._dopamine_hill_gains(signal, sens_D1, sens_D2)
        else:
            if mode == 'symmetric':
                gain_D1 = 1.0 + signal * sens_D1
                gain_D2 = 1.0 - signal * sens_D2
            elif mode == 'd1_only_stim':
                gain_D1 = 1.0 + signal * sens_D1
                gain_D2 = torch.ones_like(r[..., half_N:])
            elif mode == 'd2_only_stim':
                gain_D1 = torch.ones_like(r[..., :half_N])
                gain_D2 = 1.0 + signal * sens_D2
            elif mode == 'd1_only_suppress':
                gain_D1 = 1.0 - signal * sens_D1
                gain_D2 = torch.ones_like(r[..., half_N:])
            elif mode == 'd2_only_suppress':
                gain_D1 = torch.ones_like(r[..., :half_N])
                gain_D2 = 1.0 - signal * sens_D2
            else:
                raise ValueError(f"Unknown inference_pathway_mode: {mode}")
        gain_D1 = torch.clamp(gain_D1, min=0.1, max=1.9)
        gain_D2 = torch.clamp(gain_D2, min=0.1, max=1.9)
        if mode == 'symmetric' and not getattr(self, 'disable_inference_dopamine_bias', False):
            bias = torch.clamp(
                self.dopamine_bias.to(r.device),
                -self.config.get('dopamine_bias_max_abs', 0.3),
                self.config.get('dopamine_bias_max_abs', 0.3)
            )
        else:
            bias = torch.zeros_like(self.dopamine_bias.to(r.device))
        bias_D1 = bias[:half_N]
        bias_D2 = bias[half_N:]
        if r.dim() == 3:
            bias_D1 = bias_D1.view(1, 1, -1)
            bias_D2 = bias_D2.view(1, 1, -1)
        elif r.dim() == 2:
            bias_D1 = bias_D1.view(1, -1)
            bias_D2 = bias_D2.view(1, -1)
        elif r.dim() == 1:
            bias_D1 = bias_D1.view(-1)
            bias_D2 = bias_D2.view(-1)

        return torch.cat([
            r[..., :half_N] * gain_D1 + signal * bias_D1,
            r[..., half_N:] * gain_D2 + signal * bias_D2
        ], dim=-1)

    def _apply_value_modulation(self, r, value_signal=None):
        """Apply learned D1/D2 gain modulation driven by critic scalar value."""
        if value_signal is None or not self.config.get('use_value_modulation', False):
            return r

        d1_drive, d2_drive = self._value_modulation_drive(value_signal)
        sensitivity = torch.clamp(self.dopamine_sensitivity.to(r.device), min=0.0)
        half_N = r.shape[-1] // 2
        sens_D1 = sensitivity[:half_N]
        sens_D2 = sensitivity[half_N:]

        if r.dim() == 3:
            d1_drive = d1_drive.view(1, -1, 1)
            d2_drive = d2_drive.view(1, -1, 1)
            sens_D1 = sens_D1.view(1, 1, -1)
            sens_D2 = sens_D2.view(1, 1, -1)
        elif r.dim() == 2:
            d1_drive = d1_drive.view(-1, 1)
            d2_drive = d2_drive.view(-1, 1)
            sens_D1 = sens_D1.view(1, -1)
            sens_D2 = sens_D2.view(1, -1)
        elif r.dim() == 1:
            d1_drive = d1_drive.view(1)
            d2_drive = d2_drive.view(1)
            sens_D1 = sens_D1.view(-1)
            sens_D2 = sens_D2.view(-1)

        gain_D1 = torch.clamp(1.0 + d1_drive * sens_D1, min=0.1, max=1.9)
        gain_D2 = torch.clamp(1.0 - d2_drive * sens_D2, min=0.1, max=1.9)
        return torch.cat([
            r[..., :half_N] * gain_D1,
            r[..., half_N:] * gain_D2
        ], dim=-1)

    def _apply_decision_precision_gain(self, logits, dopamine_signal=None):
        """
        Optionally scale final action logits with an extra context-dependent
        precision gain after D1/D2 balance has already been computed.
        """
        if dopamine_signal is None:
            return logits

        if not self.config.get('decision_precision_compensation', False):
            return logits

        sensitivity = float(self.config.get('decision_precision_sensitivity', 0.0))
        if sensitivity <= 0.0:
            return logits

        signal = dopamine_signal.to(device=logits.device, dtype=logits.dtype)
        if signal.dim() == 0:
            signal = signal.unsqueeze(0)

        if self.config.get('decision_precision_negative_only', True):
            gain_drive = torch.clamp(-signal, min=0.0)
        else:
            gain_drive = torch.abs(signal)

        precision_gain = 1.0 + sensitivity * gain_drive
        max_gain = self.config.get('decision_precision_gain_max', None)
        if max_gain is not None:
            precision_gain = torch.clamp(precision_gain, max=float(max_gain))

        if logits.dim() == 3:
            if precision_gain.dim() == 1:
                precision_gain = precision_gain.view(1, -1, 1)
            elif precision_gain.dim() == 2:
                precision_gain = precision_gain.unsqueeze(-1)
        elif logits.dim() == 2:
            if precision_gain.dim() == 1:
                precision_gain = precision_gain.view(-1, 1)
        elif logits.dim() == 1:
            precision_gain = precision_gain.view(1)

        return logits * precision_gain

    def _hill_occupancy(self, da, ec50, hill_n):
        da = torch.clamp(da, min=1e-6)
        ec50_t = torch.as_tensor(ec50, dtype=da.dtype, device=da.device).clamp_min(1e-6)
        da_n = da ** hill_n
        return da_n / (da_n + ec50_t ** hill_n)

    def _dopamine_hill_gains(self, signal, sens_D1, sens_D2):
        """
        Convert signed dopamine signal to D1/D2 gains through receptor occupancy.

        D2 has lower EC50, so it is high-affinity and more saturated at baseline.
        Positive dopamine increases D1 occupancy strongly and D2 occupancy weakly;
        D2 MSN excitability decreases as D2 receptor occupancy increases.
        """
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

        delta_D1 = occ_D1 - occ0_D1
        delta_D2 = occ_D2 - occ0_D2

        gain_D1 = 1.0 + scale * delta_D1 * sens_D1
        gain_D2 = 1.0 - scale * delta_D2 * sens_D2
        return gain_D1, gain_D2

    def step_t(self, u, q, x_tm1, temperature=None, dopamine_signal=None):
        """
        Time step t > 0.

        Parameters
        ----------
        u : tensor
            Input at time t.
        q : tensor
            Noise at time t.
        x_tm1 : tensor
            State at time t-1.
        temperature : tensor, optional
            Temperature for softmax output (policy networks only).
            If None, uses default temperature (1.0).

        Returns
        -------
        z : tensor
            Output (policy or value).
        x : tensor
            State at time t.
        """
        x_t = self.recurrent_step(u, q, x_tm1, dopamine_signal=dopamine_signal)
        r_t = self.policy_rates(x_t)
        z_t = self.output_layer(r_t, temperature=temperature)
        return z_t, x_t

    def forward(self, inputs, noise, x0=None, dopamine_signal=None):
        """
        Run network for full sequence.

        Parameters
        ----------
        inputs : tensor (T, B, Nin)
            Input sequence.
        noise : tensor (T, B, N)
            Noise sequence.
        x0 : tensor (B, N), optional
            Initial state.

        Returns
        -------
        outputs : tensor (T, B, Nout)
            Output sequence.
        states : tensor (T, B, N)
            State sequence.
        """
        T, B, _ = inputs.shape

        if x0 is None:
            x0 = self.x0.unsqueeze(0).expand(B, -1)

        states = []
        outputs = []

        x_t = x0
        for t in range(T):
            u_t = inputs[t]
            q_t = noise[t]
            da_t = None
            if dopamine_signal is not None:
                da_t = dopamine_signal[t]
            x_t = self.recurrent_step(u_t, q_t, x_t, dopamine_signal=da_t)
            r_t = self.policy_rates(x_t)
            z_t = self.output_layer(r_t)

            states.append(x_t)
            outputs.append(z_t)

        states = torch.stack(states, dim=0)
        outputs = torch.stack(outputs, dim=0)

        return outputs, states

    def output_layer(self, r, temperature=None):
        """
        Apply output transformation.

        Parameters
        ----------
        r : tensor
            Firing rates.
        temperature : tensor, optional
            Temperature for softmax (policy networks only).
            Ignored for linear outputs (value networks).
        """
        raise NotImplementedError

    def recurrent_step(self, u, q, x_tm1, dopamine_signal=None):
        """Single recurrent step."""
        raise NotImplementedError

    def get_regs(self, x0, x, M):
        """Get regularization terms."""
        return torch.tensor(0.0, device=x.device)

    def get_readout_regs(self, r, M):
        """Get policy readout regularization terms for prepared output rates."""
        return torch.tensor(0.0, device=r.device)
