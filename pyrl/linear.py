from collections import OrderedDict
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import matrixtools, nptools, torchtools
from .debug import DEBUG
from .recurrent import Recurrent

from . import tasktools

configs_required = ['Nin', 'Nout']
configs_default = {
    'alpha': 1,
    'N': 50,
    'p0': 1,
    'rho': 1.5,
    'f_out': 'softmax',
    'L2_r': 0,
    'Win': 1,
    'Win_mask': None,
    'Wout': 0,
    'bout': 0,
    'x0': 0.5,
    'L1_Wrec': 0,
    'L2_Wrec': 0,
    'fix': [],
    'ei': None
}


def random_sign(rng, size):
    return 2 * rng.randint(2, size=size) - 1


class Linear(Recurrent):
    """
    Linear readout units.
    """
    def get_dim(self, name):
        if name == 'Win':
            return (self.Nin, 3 * self.N)
        if name == 'bin':
            return 3 * self.N
        if name == 'Wrec_gates':
            return (self.N, 2 * self.N)
        if name == 'Wrec':
            return (self.N, self.N)
        if name == 'Wout':
            return (self.N, self.Nout)
        if name == 'bout':
            return self.Nout
        if name == 'x0':
            return self.N

        raise ValueError(name)

    def __init__(self, config, params=None, masks=None, seed=1, name=''):
        super(Linear, self).__init__('linear', name)

        # Config
        self.config = {}

        # Required
        for k in configs_required:
            if k not in config:
                print(f"[ {self.name} ] Error: {k} is required.")
                sys.exit()
            self.config[k] = config[k]

        # Defaults available
        for k in configs_default:
            if k in config:
                self.config[k] = config[k]
            else:
                self.config[k] = configs_default[k]

        # Activations
        # Hidden
        self.f_hidden = lambda x: 1 * x
        self.firing_rate = lambda x: 1 * x

        # Output
        if self.config['f_out'] == 'softmax':
            self.f_out = torchtools.softmax
            self.f_log_out = torchtools.log_softmax
        elif self.config['f_out'] == 'linear':
            self.f_out = lambda x: x
            self.f_log_out = torch.log
        else:
            raise ValueError(self.config['f_out'])

        # Network shape
        self.Nin = self.config['Nin']
        self.N = self.config['Nin']
        self.Nout = self.config['Nout']

        # Initialize parameters
        if params is None:
            # Random number generator
            rng = nptools.get_rng(seed, __name__)

            # Connection masks
            masks = {}

            # Input masks
            if self.config['Win_mask'] is not None:
                print(f"[ {self.name} ] Setting mask for Win.")
                masks['Win'] = self.config['Win_mask']

            if self.config['p0'] < 1:
                # Recurrent in-degree
                K = int(self.config['p0'] * self.N)
                idx = np.arange(self.N)

                # Wrec
                M = np.zeros(self.get_dim('Wrec'))
                for j in range(M.shape[1]):
                    M[rng.permutation(idx)[:K], j] = 1
                masks['Wrec'] = M

                # Wrec (gates)
                M = np.zeros(self.get_dim('Wrec_gates'))
                for j in range(M.shape[1]):
                    M[rng.permutation(idx)[:K], j] = 1
                masks['Wrec_gates'] = M

            # Network parameters
            params = OrderedDict()
            if self.config['ei'] is None:
                # Input weights
                params['Win'] = self.config['Win'] * rng.normal(size=self.get_dim('Win'))

                # Input biases
                params['bin'] = np.zeros(self.get_dim('bin'))

                # Recurrent weights
                k = 4
                params['Wrec_gates'] = rng.gamma(k, 1/k, self.get_dim('Wrec_gates'))
                params['Wrec'] = rng.gamma(k, 1/k, self.get_dim('Wrec'))
                params['Wrec_gates'] *= random_sign(rng, self.get_dim('Wrec_gates'))
                params['Wrec'] *= random_sign(rng, self.get_dim('Wrec'))

                # Output weights
                if self.config['Wout'] > 0:
                    print(f"[ {self.name} ] Initialize Wout to random normal.")
                    params['Wout'] = self.config['Wout'] * rng.normal(size=self.get_dim('Wout'))
                else:
                    print(f"[ {self.name} ] Initialize Wout to zeros.")
                    params['Wout'] = np.zeros(self.get_dim('Wout'))

                # Output biases
                params['bout'] = self.config['bout'] * np.ones(self.get_dim('bout'))

                # Initial condition
                params['x0'] = self.config['x0'] * np.ones(self.get_dim('x0'))
            else:
                raise NotImplementedError

            # Desired spectral radius
            rho = self.config['rho']

            Wrec_gates = params['Wrec_gates'].copy()
            if 'Wrec_gates' in masks:
                Wrec_gates *= masks['Wrec_gates']
            Wrec = params['Wrec'].copy()
            if 'Wrec' in masks:
                Wrec *= masks['Wrec']

            rho0 = matrixtools.spectral_radius(Wrec_gates[:, :self.N])
            params['Wrec_gates'][:, :self.N] *= rho / rho0

            rho0 = matrixtools.spectral_radius(Wrec_gates[:, self.N:])
            params['Wrec_gates'][:, self.N:] *= rho / rho0

            rho0 = matrixtools.spectral_radius(Wrec)
            params['Wrec'] *= rho / rho0

        # Give to PyTorch
        # Share
        for k, v in params.items():
            self.params[k] = torchtools.shared(v, k)
            self.register_parameter(k, self.params[k])

        for k, v in masks.items():
            self.masks[k] = torchtools.shared(v)
            self.register_buffer(k + '_mask', self.masks[k])

        # Trainable parameters
        self.trainables = [self.params[k] for k in ['x0', 'Wout', 'bout']]

        # Setup
        # Leak
        self.alpha = self.config['alpha']
        print(f"[ {self.name} ] alpha = {self.alpha}")

        # Define a step
        def step(u, q, x_tm1, alpha, Win, bin, Wrec_gates, Wrec):
            return u + 0 * x_tm1 + 0 * q

        self.step = step
        self.step_params = [self.alpha]
        self.step_params += [self.get(k)
                            for k in ['Win', 'bin', 'Wrec_gates', 'Wrec']]

    def get_regs(self, x0_, x, M):
        """
        Regularization terms.
        """
        regs = 0

        # L1 recurrent weights
        L1_Wrec = self.config['L1_Wrec']
        if L1_Wrec > 0:
            print(f"L1_Wrec = {L1_Wrec}")

            W = self.get('Wrec')
            reg = torch.sum(torch.abs(W))
            size = torch.prod(torch.tensor(W.shape))

            regs += L1_Wrec * reg / size

        # L2 recurrent weights
        L2_Wrec = self.config['L2_Wrec']
        if L2_Wrec > 0:
            print(f"L2_Wrec = {L2_Wrec}")

            W = self.get('Wrec')
            reg = torch.sum(W ** 2)
            size = torch.prod(torch.tensor(W.shape))

            W = self.get('Wrec_gates')
            reg += torch.sum(W ** 2)
            size += torch.prod(torch.tensor(W.shape))

            regs += L2_Wrec * reg / size

        # Firing rates
        L2_r = self.config['L2_r']
        if L2_r > 0:
            # Repeat (T, B) -> (T, B, N)
            M_ = M.T.unsqueeze(-1).repeat(1, 1, x.shape[-1]).permute(1, 0, 2)

            # Combine t=0 with t>0
            x_all = torch.cat([x0_.unsqueeze(0), x], dim=0)

            # Firing rate
            r = self.f_hidden(x_all)

            # Regularization
            regs += L2_r * torch.sum((r ** 2) * M_) / torch.sum(M_)

        return regs
