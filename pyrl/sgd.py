import numpy as np
import torch
import torch.nn as nn

import torchtools


class Adam(object):
    """
    Adam optimizer for PyTorch.
    """
    def __init__(self, trainables, accumulators=None):
        self.trainables = trainables

        if accumulators is None:
            self.means = [torchtools.shared(torch.zeros_like(x.data)) for x in trainables]
            self.vars = [torchtools.shared(torch.zeros_like(x.data)) for x in trainables]
            self.time = torchtools.shared(torch.tensor(0))
        else:
            self.means = [torchtools.shared(torch.from_numpy(x)) for x in accumulators[0]]
            self.vars = [torchtools.shared(torch.from_numpy(x)) for x in accumulators[1]]
            self.time = torchtools.shared(torch.tensor(accumulators[2]))

    def get_values(self):
        means = [x.detach().cpu().numpy() for x in self.means]
        vars_ = [x.detach().cpu().numpy() for x in self.vars]
        time = self.time.item()

        return [means, vars_, time]

    def get_updates(self, loss, lr, max_norm=1, beta1=0.9, beta2=0.999,
                    epsilon=1e-8, grads=None):
        """
        Get Adam updates.

        Args:
            loss: loss tensor
            lr: learning rate
            max_norm: maximum gradient norm for clipping
            beta1: exponential decay rate for first moment
            beta2: exponential decay rate for second moment
            epsilon: small constant for numerical stability
            grads: optional pre-computed gradients

        Returns:
            norm: gradient norm
            grads: clipped gradients
            updates: list of (parameter, new_value) tuples
        """
        # Gradients
        if grads is None:
            grads = torch.autograd.grad(loss, self.trainables, create_graph=True)

        # Clipping
        norm = torch.sqrt(sum([torch.sum(g ** 2) for g in grads]))
        m = torchtools.clipping_multiplier(norm, max_norm)
        grads = [m * g for g in grads]

        # Safeguard against numerical instability
        new_cond = torch.logical_or(
            torch.logical_or(torch.isnan(norm), torch.isinf(norm)),
            torch.logical_or(norm < 0, norm > 1e10)
        )
        grads = [torch.where(new_cond, torch.tensor(0.0, dtype=g.dtype, device=g.device), g)
                for g in grads]

        # New values
        t = self.time + 1
        lr_t = lr * torch.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)
        means_t = [beta1 * m + (1.0 - beta1) * g for g, m in zip(grads, self.means)]
        vars_t = [beta2 * v + (1.0 - beta2) * (g ** 2) for g, v in zip(grads, self.vars)]
        steps = [lr_t * m_t / (torch.sqrt(v_t) + epsilon)
                for m_t, v_t in zip(means_t, vars_t)]

        # Updates
        updates = [(x, x - step) for x, step in zip(self.trainables, steps)]
        updates += [(m, m_t) for m, m_t in zip(self.means, means_t)]
        updates += [(v, v_t) for v, v_t in zip(self.vars, vars_t)]
        updates += [(self.time, t)]

        return norm, grads, updates
