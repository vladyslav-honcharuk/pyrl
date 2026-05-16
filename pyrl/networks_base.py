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

        r = self.firing_rate(x0)
        z = self.output_layer(r, temperature=temperature)
        return z, x0

    def _apply_dopamine_modulation(self, r, dopamine_signal=None):
        """Apply push-pull D1/D2 gain modulation to firing rates."""
        if dopamine_signal is None:
            return r

        if dopamine_signal.dim() == 0:
            dopamine_signal = dopamine_signal.unsqueeze(0)

        signal = torch.clamp(dopamine_signal, -0.9, 0.9)
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

        if self.config.get('dopamine_modulation_mode', 'linear') == 'hill':
            gain_D1, gain_D2 = self._dopamine_hill_gains(signal, sens_D1, sens_D2)
        else:
            gain_D1 = 1.0 + signal * sens_D1
            gain_D2 = 1.0 - signal * sens_D2
        gain_D1 = torch.clamp(gain_D1, min=0.1, max=1.9)
        gain_D2 = torch.clamp(gain_D2, min=0.1, max=1.9)
        bias = torch.clamp(
            self.dopamine_bias.to(r.device),
            -self.config.get('dopamine_bias_max_abs', 0.3),
            self.config.get('dopamine_bias_max_abs', 0.3)
        )
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
        r_t = self.firing_rate(x_t)
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
            r_t = self.firing_rate(x_t)
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
