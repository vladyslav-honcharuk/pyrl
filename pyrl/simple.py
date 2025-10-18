"""
Simple recurrent neural network implementation in PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .networks_base import RecurrentNetwork
from .nptools import get_rng
from .matrixtools import spectral_radius


class SimpleRNN(RecurrentNetwork):
    """Simple recurrent network with ReLU activation."""

    def __init__(self, config, params=None, masks=None, seed=1, name=''):
        super().__init__('simple', name)

        # Store config
        required = ['Nin', 'Nout']
        for k in required:
            if k not in config:
                raise ValueError(f"SimpleRNN requires config key: {k}")

        defaults = {
            'alpha': 1.0,
            'N': 50,
            'rho': 1.5,
            'f_out': 'softmax',
            'L2_r': 0.002,
            'L1_Wrec': 0,
            'L2_Wrec': 0,
            'fix': []
        }

        self.config = {**defaults, **config}

        # Network dimensions
        self.Nin = self.config['Nin']
        self.N = self.config['N']
        self.Nout = self.config['Nout']
        self.alpha = self.config['alpha']

        # Fixed parameters
        self._fixed_params = self.config['fix']

        print(f"[ {self.network_name} ] alpha = {self.alpha}")
        print(f"[ {self.network_name} ] L2_r = {self.config['L2_r']}")

        # Initialize or load parameters
        if params is None:
            self._initialize_params(seed)
        else:
            self._load_params(params)

        # Set output activation
        self.f_out = self.config['f_out']

    def _initialize_params(self, seed):
        """Initialize network parameters."""
        rng = get_rng(seed, __name__)
        print(f"Seed = {seed}")

        # Input weights
        Win = rng.normal(size=(self.Nin, self.N))
        self.Win = nn.Parameter(torch.FloatTensor(Win))

        # Input biases
        self.bin = nn.Parameter(torch.zeros(self.N))

        # Recurrent weights
        Wrec = rng.normal(size=(self.N, self.N))
        rho = self.config['rho']
        rho0 = spectral_radius(Wrec)
        Wrec *= rho / rho0
        self.Wrec = nn.Parameter(torch.FloatTensor(Wrec))

        # Output weights
        Wout = self.config.get('Wout_init', np.zeros((self.N, self.Nout)))
        self.Wout = nn.Parameter(torch.FloatTensor(Wout))

        # Output biases
        self.bout = nn.Parameter(torch.zeros(self.Nout))

        # Initial state
        states_0_init = self.config.get('states_0_init', np.arctanh(0.5))
        self.x0 = nn.Parameter(torch.FloatTensor(
            states_0_init * np.ones(self.N)
        ))

    def _load_params(self, params):
        """Load parameters from saved values."""
        self.Win = nn.Parameter(torch.FloatTensor(params['Win']))
        self.bin = nn.Parameter(torch.FloatTensor(params['bin']))
        self.Wrec = nn.Parameter(torch.FloatTensor(params['Wrec']))
        self.Wout = nn.Parameter(torch.FloatTensor(params['Wout']))
        self.bout = nn.Parameter(torch.FloatTensor(params['bout']))
        self.x0 = nn.Parameter(torch.FloatTensor(params['states_0']))

    def recurrent_step(self, u, q, x_tm1):
        """
        Single RNN step.

        Parameters
        ----------
        u : tensor (B, Nin)
            Input.
        q : tensor (B, N)
            Noise.
        x_tm1 : tensor (B, N)
            Previous state.

        Returns
        -------
        x_t : tensor (B, N)
            Current state.
        """
        # Input transformation
        inputs_t = torch.matmul(u, self.Win) + self.bin

        # Firing rate from previous state
        r_tm1 = torch.relu(x_tm1)

        # State update
        next_states = torch.matmul(r_tm1, self.Wrec) + inputs_t + q
        x_t = (1 - self.alpha) * x_tm1 + self.alpha * next_states

        return x_t

    def output_layer(self, r):
        """Apply output transformation."""
        logits = torch.matmul(r, self.Wout) + self.bout

        if self.f_out == 'softmax':
            return F.softmax(logits, dim=-1)
        elif self.f_out == 'linear':
            return logits
        else:
            raise ValueError(f"Unknown output activation: {self.f_out}")

    def log_output(self, r):
        """Apply log output transformation."""
        logits = torch.matmul(r, self.Wout) + self.bout

        if self.f_out == 'softmax':
            return F.log_softmax(logits, dim=-1)
        elif self.f_out == 'linear':
            return torch.log(logits)
        else:
            raise ValueError(f"Unknown output activation: {self.f_out}")

    def get_regs(self, x0, x, M):
        """
        Compute regularization terms.

        Parameters
        ----------
        x0 : tensor (B, N)
            Initial states.
        x : tensor (T, B, N)
            State trajectory.
        M : tensor (T, B)
            Mask indicating valid timesteps.

        Returns
        -------
        regs : tensor (scalar)
            Total regularization.
        """
        regs = torch.tensor(0.0, device=x.device)

        # L1 recurrent weights
        if self.config['L1_Wrec'] > 0:
            regs += self.config['L1_Wrec'] * torch.mean(torch.abs(self.Wrec))

        # L2 recurrent weights
        if self.config['L2_Wrec'] > 0:
            regs += self.config['L2_Wrec'] * torch.mean(self.Wrec ** 2)

        # L2 firing rate
        if self.config['L2_r'] > 0:
            # Expand mask to match state dimensions
            M_expanded = M.unsqueeze(-1).expand_as(x)

            # Combine t=0 with t>0
            x_all = torch.cat([x0.unsqueeze(0), x], dim=0)

            # Firing rates
            r = torch.relu(x_all)

            # Regularization
            regs += self.config['L2_r'] * torch.sum((r ** 2) * M_expanded) / torch.sum(M_expanded)

        return regs
