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
            'fix': [],
            'dopamine_heterogeneous_sensitivity': False,
            'dopamine_sensitivity_min': 0.3,
            'dopamine_sensitivity_max': 1.0,
            'dopamine_sensitivity_learned': False,
            'dopamine_sensitivity_seed': seed,
            'dopamine_bias_enabled': False,
            'dopamine_bias_learned': True,
            'dopamine_bias_init': 0.0,
            'dopamine_bias_max_abs': 0.3,
            'dopamine_modulation_mode': 'linear',
            'dopamine_hill_base_da': 1.0,
            'dopamine_hill_da_range': 1.0,
            'dopamine_hill_ec50_d1': 1.0,
            'dopamine_hill_ec50_d2': 0.07,
            'dopamine_hill_coefficient': 1.0,
            'dopamine_hill_gain_scale': 2.0,
        }

        self.config = {**defaults, **config}

        # Network dimensions
        self.Nin = self.config['Nin']
        self.N = self.config['N']
        self.Nout = self.config['Nout']
        self.alpha = self.config['alpha']

        # Fixed parameters
        self._fixed_params = self.config['fix']


        # Initialize or load parameters
        if params is None:
            self._initialize_params(seed)
        else:
            self._load_params(params)
        self._setup_dopamine_sensitivity(seed, params=params)
        self._setup_dopamine_bias(params=params)

        # Set output activation
        self.f_out = self.config['f_out']

    def _initialize_params(self, seed):
        """Initialize network parameters."""
        rng = get_rng(seed, __name__)

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

    def recurrent_step(self, u, q, x_tm1, dopamine_signal=None):
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

        # Firing rate from previous state, optionally modulated by dopamine.
        r_tm1 = torch.relu(x_tm1)
        r_tm1 = self._apply_dopamine_modulation(r_tm1, dopamine_signal)

        # State update
        next_states = torch.matmul(r_tm1, self.Wrec) + inputs_t + q
        x_t = (1 - self.alpha) * x_tm1 + self.alpha * next_states

        return x_t

    def _policy_readout_rates(self, r):
        """Map activity to nonnegative readout rates when configured."""
        if self.config.get('positive_policy_readout', False) and self.f_out == 'softmax':
            return 0.5 * (r + 1.0)
        return r

    def _effective_output_weights(self):
        """Return output weights after optional positivity constraint."""
        if self.config.get('positive_policy_readout', False) and self.f_out == 'softmax':
            return F.softplus(self.Wout)
        return self.Wout

    def output_layer(self, r, temperature=None, return_logits=False):
        """
        Apply output transformation with optional temperature scaling.

        Parameters
        ----------
        r : tensor
            Firing rates.
        temperature : tensor, optional
            Temperature for softmax. Shape: (B,) where B is batch size.
            Only used if f_out='softmax'. Higher temperature → flatter distribution.
            If None, uses standard softmax (temperature=1.0).
        return_logits : bool
            If True, return raw logits before softmax. Default: False.
        """
        r_out = self._policy_readout_rates(r)
        Wout = self._effective_output_weights()

        # Compute logits with opponent modulation if enabled
        if self.config.get('use_opponent_modulation', False):
            # Direct (D1/Go) - Indirect (D2/No-Go) opponent computation
            # This implements the biological mechanism where D1 and D2 pathways
            # have opponent effects on action selection through GPi
            half_N = self.N // 2
            d1_rates = r_out[..., :half_N]
            d2_rates = r_out[..., half_N:]
            d1_logits = torch.matmul(d1_rates, Wout[:half_N, :])
            d2_logits = torch.matmul(d2_rates, Wout[half_N:, :])
            logits = d1_logits - d2_logits + self.bout
        else:
            # Standard computation: all neurons contribute additively
            logits = torch.matmul(r_out, Wout) + self.bout

        if return_logits:
            return logits

        if self.f_out == 'softmax':
            if temperature is not None:
                # Apply temperature scaling: softmax(logits / T)
                # Reshape temperature for broadcasting: (B,) → (B, 1)
                if temperature.dim() == 1:
                    temperature = temperature.unsqueeze(-1)
                logits = logits / temperature
            return F.softmax(logits, dim=-1)
        elif self.f_out == 'linear':
            # Linear output (value networks) - temperature not applicable
            return logits
        else:
            raise ValueError(f"Unknown output activation: {self.f_out}")

    def log_output(self, r, temperature=None):
        """
        Apply log output transformation with optional temperature scaling.

        Parameters
        ----------
        r : tensor
            Firing rates.
        temperature : tensor, optional
            Temperature for log_softmax. Shape: (B,) where B is batch size.
            Only used if f_out='softmax'.
            If None, uses standard log_softmax (temperature=1.0).
        """
        logits = self.output_layer(r, return_logits=True)

        if self.f_out == 'softmax':
            if temperature is not None:
                # Apply temperature scaling
                if temperature.dim() == 1:
                    temperature = temperature.unsqueeze(-1)
                logits = logits / temperature
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
        if self.config.get('L1_Wrec', 0) > 0:
            regs += self.config['L1_Wrec'] * torch.mean(torch.abs(self.Wrec))

        # L2 recurrent weights
        if self.config.get('L2_Wrec', 0) > 0:
            regs += self.config['L2_Wrec'] * torch.mean(self.Wrec ** 2)

        if self.config.get('L2_r', 0) > 0 or self.config.get('activity_balance', 0) > 0:
            x_all = torch.cat([x0.unsqueeze(0), x], dim=0)
            r = torch.relu(x_all)
            M_all = torch.cat([torch.ones_like(M[:1]), M], dim=0)
            M_expanded = M_all.unsqueeze(-1).expand_as(r)

        # L2 firing rate
        if self.config.get('L2_r', 0) > 0:
            regs += self.config['L2_r'] * torch.sum((r ** 2) * M_expanded) / torch.sum(M_expanded)

        # Penalize concentrating activity in only a few neurons.
        if self.config.get('activity_balance', 0) > 0:
            valid = torch.sum(M_all).clamp_min(1.0)
            mean_abs = torch.sum(torch.abs(r) * M_expanded, dim=(0, 1)) / valid
            p = mean_abs / mean_abs.sum().clamp_min(1e-8)
            concentration = torch.sum(p ** 2) - (1.0 / r.shape[-1])
            regs += self.config['activity_balance'] * concentration

        return regs
