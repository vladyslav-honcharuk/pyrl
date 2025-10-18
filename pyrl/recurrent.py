from collections import OrderedDict
import sys

import torch
import torch.nn as nn


class Recurrent(nn.Module):
    """
    Generic recurrent unit.
    """
    def __init__(self, type_, name=''):
        super(Recurrent, self).__init__()
        self.type = type_
        self.name = self.type
        self.params = OrderedDict()
        self.masks = {}

        if name != '':
            self.name += '-' + name

        # Subclasses should define:
        # self.N
        # self.trainables
        # self.f_hidden
        # self.f_out
        # self.f_log_out
        # self.step

    @property
    def noise_dim(self):
        return self.N

    def get_dim(self, name):
        raise NotImplementedError

    def index(self, name):
        for i, (param_name, _) in enumerate(self.named_parameters()):
            if param_name == name:
                return i
        return None

    def get_masks(self):
        return {k: v.detach().cpu().numpy() for k, v in self.masks.items()}

    def get_values(self):
        return OrderedDict([(k, v.detach().cpu().numpy())
                           for k, v in self.params.items()])

    def get(self, name):
        p = self.params[name]
        if name in self.masks:
            return self.masks[name] * p
        return p

    def func_step_0(self, use_x0=False):
        """
        Returns a function for step 0.
        """
        def step_0(x0=None):
            if use_x0:
                x0_val = x0
            else:
                x0_val = self.get('x0')

            Wout = self.get('Wout')
            bout = self.get('bout')

            r = self.f_hidden(x0_val)
            z = self.f_out(r @ Wout + bout)

            return z, x0_val

        return step_0

    def func_step_t(self):
        """
        Returns a function for step t.
        """
        def step_t(inputs, noise, x_tm1):
            Wout = self.get('Wout')
            bout = self.get('bout')

            x_t = self.step(inputs, noise, x_tm1, *self.step_params)
            r_t = self.f_hidden(x_t)
            z_t = self.f_out(r_t @ Wout + bout)

            return z_t[0], x_t[0]

        return step_t

    def get_outputs_0(self, x0, log=False):
        Wout = self.get('Wout')
        bout = self.get('bout')
        r0 = self.f_hidden(x0)

        if log:
            return self.f_log_out(r0 @ Wout + bout)
        return self.f_out(r0 @ Wout + bout)

    def get_outputs(self, inputs, noise, x0, log=False):
        """
        Get outputs for a sequence of inputs.

        Args:
            inputs: (T, B, input_dim)
            noise: (T, B, noise_dim)
            x0: (B, N) initial state
            log: whether to return log outputs

        Returns:
            x: (T, B, N) hidden states
            outputs: (T, B, output_dim) outputs
        """
        Wout = self.get('Wout')
        bout = self.get('bout')

        T = inputs.shape[0]
        B = x0.shape[0]

        x_list = []
        x_t = x0

        for t in range(T):
            x_t = self.step(inputs[t], noise[t], x_t, *self.step_params)
            x_list.append(x_t)

        x = torch.stack(x_list, dim=0)  # (T, B, N)
        r = self.f_hidden(x)

        if log:
            return x, self.f_log_out(r @ Wout + bout)
        return x, self.f_out(r @ Wout + bout)

    def get_regs(self, x0_, x, M):
        return 0
