"""
Modified Gated Recurrent Unit (GRU) implementation in PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .networks_base import RecurrentNetwork
from .nptools import get_rng
from .matrixtools import spectral_radius


class GRU(RecurrentNetwork):
    """Modified Gated Recurrent Unit network."""

    def __init__(self, config, params=None, masks=None, seed=1, name=''):
        super().__init__('gru', name)

        # Store config
        required = ['Nin', 'Nout']
        for k in required:
            if k not in config:
                raise ValueError(f"GRU requires config key: {k}")

        defaults = {
            'alpha': 1.0,
            'N': 50,
            'p0': 1.0,
            'rho': 1.5,
            'f_out': 'softmax',
            'L2_r': 0,
            'Win': 1.0,
            'Win_mask': None,
            'Wout': 0,
            'bout': 0,
            'x0': 0.5,
            'L1_Wrec': 0,
            'L2_Wrec': 0,
            'fix': [],
            'ei': None,
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
            self._initialize_params(seed, masks)
        else:
            self._load_params(params, masks)
        self._setup_dopamine_sensitivity(seed, params=params)
        self._setup_dopamine_bias(params=params)
        self._setup_value_modulation(params=params)

        # Set output activation
        self.f_out = self.config['f_out']

    def _initialize_params(self, seed, masks):
        """Initialize network parameters."""
        rng = get_rng(seed, __name__)

        # Connection masks
        if masks is None:
            masks = {}

        # Input mask
        if self.config['Win_mask'] is not None:
            masks['Win'] = self.config['Win_mask']

        # Sparse recurrent connectivity
        if self.config['p0'] < 1:
            K = int(self.config['p0'] * self.N)
            idx = np.arange(self.N)

            # Wrec mask
            M_wrec = np.zeros((self.N, self.N))
            for j in range(self.N):
                M_wrec[rng.permutation(idx)[:K], j] = 1
            masks['Wrec'] = M_wrec

            # Wrec_gates mask
            M_gates = np.zeros((self.N, 2*self.N))
            for j in range(2*self.N):
                M_gates[rng.permutation(idx)[:K], j] = 1
            masks['Wrec_gates'] = M_gates

        # Input weights
        Win = self.config['Win'] * rng.normal(size=(self.Nin, 3*self.N))
        self.Win = nn.Parameter(torch.FloatTensor(Win))

        # Input biases
        self.bin = nn.Parameter(torch.zeros(3*self.N))

        # Recurrent weights (gates)
        k = 4
        Wrec_gates = rng.gamma(k, 1/k, size=(self.N, 2*self.N))
        Wrec_gates *= 2*rng.randint(2, size=(self.N, 2*self.N)) - 1
        self.Wrec_gates = nn.Parameter(torch.FloatTensor(Wrec_gates))

        # Recurrent weights (states)
        Wrec = rng.gamma(k, 1/k, size=(self.N, self.N))
        Wrec *= 2*rng.randint(2, size=(self.N, self.N)) - 1
        self.Wrec = nn.Parameter(torch.FloatTensor(Wrec))

        # Apply masks and normalize spectral radius
        rho = self.config['rho']

        Wrec_gates_masked = self.Wrec_gates.data.numpy().copy()
        if 'Wrec_gates' in masks:
            Wrec_gates_masked *= masks['Wrec_gates']

        Wrec_masked = self.Wrec.data.numpy().copy()
        if 'Wrec' in masks:
            Wrec_masked *= masks['Wrec']

        # Normalize spectral radius for each gate component
        rho0 = spectral_radius(Wrec_gates_masked[:, :self.N])
        self.Wrec_gates.data[:, :self.N] *= rho / rho0

        rho0 = spectral_radius(Wrec_gates_masked[:, self.N:])
        self.Wrec_gates.data[:, self.N:] *= rho / rho0

        rho0 = spectral_radius(Wrec_masked)
        self.Wrec.data *= rho / rho0

        # Output weights
        if self.config['Wout'] > 0:
            Wout = self.config['Wout'] * rng.normal(size=(self.N, self.Nout))
        else:
            Wout = np.zeros((self.N, self.Nout))
        self.Wout = nn.Parameter(torch.FloatTensor(Wout))

        # Output biases
        self.bout = nn.Parameter(torch.FloatTensor(
            self.config['bout'] * np.ones(self.Nout)
        ))

        # Initial state
        self.x0 = nn.Parameter(torch.FloatTensor(
            self.config['x0'] * np.ones(self.N)
        ))

        # Store masks
        for k, v in masks.items():
            self.masks[k] = torch.FloatTensor(v)

    def _load_params(self, params, masks):
        """Load parameters from saved values."""
        self.Win = nn.Parameter(torch.FloatTensor(params['Win']))
        self.bin = nn.Parameter(torch.FloatTensor(params['bin']))
        self.Wrec_gates = nn.Parameter(torch.FloatTensor(params['Wrec_gates']))
        self.Wrec = nn.Parameter(torch.FloatTensor(params['Wrec']))
        self.Wout = nn.Parameter(torch.FloatTensor(params['Wout']))
        self.bout = nn.Parameter(torch.FloatTensor(params['bout']))
        self.x0 = nn.Parameter(torch.FloatTensor(params['x0']))

        # Load masks
        if masks:
            for k, v in masks.items():
                self.masks[k] = torch.FloatTensor(v)

    def _apply_mask(self, param_name):
        """Apply mask to parameter if it exists."""
        param = getattr(self, param_name)
        if param_name in self.masks:
            mask = self.masks[param_name].to(param.device)
            return param * mask
        return param

    def policy_rates(self, x):
        """Return the activity representation used only by the policy readout."""
        if self.config.get('positive_policy_readout', False) and self.f_out == 'softmax':
            return F.softplus(x)
        return self.firing_rate(x)

    def _effective_output_weights(self):
        """Return output weights after applying any readout constraint."""
        if self.config.get('positive_policy_readout', False) and self.f_out == 'softmax':
            return F.softplus(self.Wout)
        return self.Wout

    def uses_opponent_readout(self):
        """Return whether D2 is subtracted in the policy readout."""
        return (
            self.config.get('use_opponent_modulation', False) or
            (self.config.get('positive_policy_readout', False) and self.f_out == 'softmax')
        )

    def recurrent_step(self, u, q, x_tm1, dopamine_signal=None):
        """
        Single GRU step.

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
        # Apply masks to recurrent weights
        Wrec_gates = self._apply_mask('Wrec_gates')
        Wrec = self._apply_mask('Wrec')

        # Input transformation
        inputs_t = torch.matmul(u, self.Win) + self.bin
        pop_input_dim = int(getattr(self, 'value_pop_input_dim', 0) or 0)
        d1_gain = float(getattr(self, 'value_pop_current_d1_gain', 1.0))
        d2_gain = float(getattr(self, 'value_pop_current_d2_gain', 1.0))
        if pop_input_dim > 0 and (d1_gain != 1.0 or d2_gain != 1.0):
            pop_u = u[:, -pop_input_dim:]
            pop_current = torch.matmul(pop_u, self.Win[-pop_input_dim:, :])
            scaled_pop_current = pop_current.clone()
            half_N = self.N // 2
            d1_cols = np.concatenate([
                np.arange(0, half_N),
                np.arange(self.N, self.N + half_N),
                np.arange(2 * self.N, 2 * self.N + half_N),
            ])
            d2_cols = np.concatenate([
                np.arange(half_N, self.N),
                np.arange(self.N + half_N, 2 * self.N),
                np.arange(2 * self.N + half_N, 3 * self.N),
            ])
            scaled_pop_current[:, d1_cols] *= d1_gain
            scaled_pop_current[:, d2_cols] *= d2_gain
            inputs_t = inputs_t + (scaled_pop_current - pop_current)
        state_inputs = inputs_t[:, :self.N]
        gate_inputs = inputs_t[:, self.N:]

        # Firing rate from previous state, optionally modulated by dopamine.
        r_tm1 = torch.tanh(x_tm1)
        r_tm1 = self._apply_dopamine_modulation(r_tm1, dopamine_signal)

        # Gate values
        gate_values = torch.sigmoid(torch.matmul(r_tm1, Wrec_gates) + gate_inputs)
        update_values = gate_values[:, :self.N]
        g = gate_values[:, self.N:]

        # State update
        x_t = ((1 - self.alpha * update_values) * x_tm1 +
               self.alpha * update_values * (torch.matmul(g * r_tm1, Wrec) + state_inputs + q))

        return x_t

    def output_layer(self, r, temperature=None, return_logits=False, pathway_gradient=None,
                     control_r=None, modulation_signal=None):
        """
        Apply output transformation with optional temperature scaling.

        Parameters
        ----------
        r : tensor
            Rates already prepared for policy readout by ``policy_rates`` and
            any action-time dopamine modulation.
        temperature : tensor, optional
            Temperature for softmax. Shape: (B,) where B is batch size.
            Only used if f_out='softmax'. Higher temperature → flatter distribution.
            If None, uses standard softmax (temperature=1.0).
        return_logits : bool
            If True, return raw logits before softmax. Default: False.
        pathway_gradient : {'d1', 'd2', 'bias'}, optional
            Restrict autograd to one opponent readout pathway or only the
            output bias. Used for pathway-specific plasticity during training.
        control_r : tensor, optional
            Unmodulated rates used only for the single control action logit
            when exclude_control_action_from_dopamine_modulation is enabled.
        """
        Wout = self._effective_output_weights()
        if self.config.get('positive_policy_readout', False) and self.f_out == 'softmax':
            # OpAL-style readout: nonneg rates from pre-tanh state, nonneg weights,
            # D1 (first half) promotes actions, D2 (second half) suppresses them.
            half_N = self.N // 2
            d1_logits = torch.matmul(r[..., :half_N], Wout[:half_N, :])
            d2_logits = torch.matmul(r[..., half_N:], Wout[half_N:, :])
        elif self.uses_opponent_readout():
            # Existing opponent path (unconstrained rates/weights, kept for compat)
            half_N = self.N // 2
            d1_rates = r[..., :half_N]
            d2_rates = r[..., half_N:]
            d1_logits = torch.matmul(d1_rates, Wout[:half_N, :])
            d2_logits = torch.matmul(d2_rates, Wout[half_N:, :])
        else:
            if pathway_gradient is not None:
                raise ValueError("pathway_gradient requires an opponent policy readout")
            logits = torch.matmul(r, Wout) + self.bout

        if self.uses_opponent_readout():
            if pathway_gradient == 'd1':
                logits = d1_logits - d2_logits.detach() + self.bout.detach()
            elif pathway_gradient == 'd2':
                logits = d1_logits.detach() - d2_logits + self.bout.detach()
            elif pathway_gradient == 'bias':
                logits = d1_logits.detach() - d2_logits.detach() + self.bout
            elif pathway_gradient is None:
                logits = d1_logits - d2_logits + self.bout
            else:
                raise ValueError(f"Unknown pathway_gradient: {pathway_gradient}")

        control_index = self.control_action_index()
        if control_index is not None and control_r is not None:
            control_r = self._policy_readout_rates(control_r)
            if self.uses_opponent_readout():
                half_N = self.N // 2
                control_d1 = torch.matmul(control_r[..., :half_N], Wout[:half_N, control_index:control_index + 1])
                control_d2 = torch.matmul(control_r[..., half_N:], Wout[half_N:, control_index:control_index + 1])

                if pathway_gradient == 'd1':
                    control_logit = (
                        control_d1
                        - control_d2.detach()
                        + self.bout[control_index:control_index + 1].detach()
                    )
                elif pathway_gradient == 'd2':
                    control_logit = (
                        control_d1.detach()
                        - control_d2
                        + self.bout[control_index:control_index + 1].detach()
                    )
                elif pathway_gradient == 'bias':
                    control_logit = (
                        control_d1.detach()
                        - control_d2.detach()
                        + self.bout[control_index:control_index + 1]
                    )
                else:
                    control_logit = control_d1 - control_d2 + self.bout[control_index:control_index + 1]
            else:
                control_logit = (
                    torch.matmul(control_r, Wout[:, control_index:control_index + 1])
                    + self.bout[control_index:control_index + 1]
                )

            logits = logits.clone()
            logits[..., control_index:control_index + 1] = control_logit

        logits = self._apply_decision_precision_gain(logits, modulation_signal)

        if return_logits:
            return logits

        if self.f_out == 'softmax':
            if temperature is not None:
                if temperature.dim() == 1:
                    temperature = temperature.unsqueeze(-1)
                logits = logits / temperature
            return F.softmax(logits, dim=-1)
        elif self.f_out == 'linear':
            return logits
        else:
            raise ValueError(f"Unknown output activation: {self.f_out}")

    def log_output(self, r, temperature=None, pathway_gradient=None, control_r=None,
                   modulation_signal=None):
        """Apply log output transformation with optional temperature scaling."""
        logits = self.output_layer(
            r,
            return_logits=True,
            pathway_gradient=pathway_gradient,
            control_r=control_r,
            modulation_signal=modulation_signal
        )

        if self.f_out == 'softmax':
            if temperature is not None:
                if temperature.dim() == 1:
                    temperature = temperature.unsqueeze(-1)
                logits = logits / temperature
            return F.log_softmax(logits, dim=-1)
        elif self.f_out == 'linear':
            return torch.log(logits)
        else:
            raise ValueError(f"Unknown output activation: {self.f_out}")

    def get_readout_regs(self, r, M):
        """Penalize large positive actor strengths and cancelling opponent pulls."""
        if not (self.config.get('positive_policy_readout', False) and self.f_out == 'softmax'):
            return torch.tensor(0.0, device=r.device)

        regs = torch.tensor(0.0, device=r.device)
        Wout = self._effective_output_weights()
        weight_l2 = float(self.config.get('positive_readout_weight_l2', 0.0))
        if weight_l2 > 0:
            regs = regs + weight_l2 * torch.mean(Wout ** 2)

        pull_l2 = float(self.config.get('opponent_pull_l2', 0.0))
        if pull_l2 > 0:
            half_N = self.N // 2
            d1_pull = torch.matmul(r[..., :half_N], Wout[:half_N, :])
            d2_pull = torch.matmul(r[..., half_N:], Wout[half_N:, :])
            M_expanded = M.unsqueeze(-1).expand_as(d1_pull)
            n_valid = torch.sum(M_expanded).clamp_min(1.0)
            pull_sq = (d1_pull ** 2 + d2_pull ** 2) * M_expanded
            regs = regs + pull_l2 * torch.sum(pull_sq) / n_valid

        return regs

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
            W = self._apply_mask('Wrec')
            regs += self.config['L1_Wrec'] * torch.mean(torch.abs(W))

        # L2 recurrent weights
        if self.config.get('L2_Wrec', 0) > 0:
            W_gates = self._apply_mask('Wrec_gates')
            W = self._apply_mask('Wrec')
            reg = torch.sum(W_gates ** 2) + torch.sum(W ** 2)
            size = W_gates.numel() + W.numel()
            regs += self.config['L2_Wrec'] * reg / size

        if self.config.get('L2_r', 0) > 0 or self.config.get('activity_balance', 0) > 0:
            x_all = torch.cat([x0.unsqueeze(0), x], dim=0)
            r = torch.tanh(x_all)
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
