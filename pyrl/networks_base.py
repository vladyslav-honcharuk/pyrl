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

    def step_0(self, x0=None):
        """
        Initial step (t=0).

        Parameters
        ----------
        x0 : tensor, optional
            Initial state. If None, uses learned initial state.

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
        z = self.output_layer(r)
        return z, x0

    def step_t(self, u, q, x_tm1):
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

        Returns
        -------
        z : tensor
            Output (policy or value).
        x : tensor
            State at time t.
        """
        x_t = self.recurrent_step(u, q, x_tm1)
        r_t = self.firing_rate(x_t)
        z_t = self.output_layer(r_t)
        return z_t, x_t

    def forward(self, inputs, noise, x0=None):
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
            x_t = self.recurrent_step(u_t, q_t, x_t)
            r_t = self.firing_rate(x_t)
            z_t = self.output_layer(r_t)

            states.append(x_t)
            outputs.append(z_t)

        states = torch.stack(states, dim=0)
        outputs = torch.stack(outputs, dim=0)

        return outputs, states

    def output_layer(self, r):
        """Apply output transformation."""
        raise NotImplementedError

    def recurrent_step(self, u, q, x_tm1):
        """Single recurrent step."""
        raise NotImplementedError

    def get_regs(self, x0, x, M):
        """Get regularization terms."""
        return torch.tensor(0.0, device=x.device)
