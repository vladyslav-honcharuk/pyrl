"""Smoke checks for the current scalar actor-critic trainer."""

import sys

import numpy as np
import torch
import torch.nn.functional as F

from pyrl import configs
from pyrl.performance import Performance2AFC
from pyrl.actor_critic import ActorCriticTrainer
import tasks.gambling as gambling_task


class DummyTask:
    """Minimal task with the same input/output surface as the gambling task."""

    def get_condition(self, rng, dt):
        return {
            'durations': {'fixation': (0, 250), 'tmax': 760},
            'time': np.arange(0, 760, dt),
            'epochs': {'fixation': range(25)},
        }

    def get_step(self, rng, dt, trial, t, a):
        u = np.zeros(len(gambling_task.inputs), dtype=np.float32)
        r = 0.0
        return u, r, {'continue': t < 75, 'reward': r}


def make_config():
    config = {**configs.default}
    config.update({
        'inputs': gambling_task.inputs,
        'actions': gambling_task.actions,
        'Nin': len(gambling_task.inputs),
        'Nout': len(gambling_task.actions),
        'tmax': 760,
        'n_gradient': 2,
        'n_validation': 2,
        'N': 50,
        'baseline_N': 50,
        'max_iter': 1,
        'Performance': Performance2AFC,
    })
    return config


def test_scalar_policy_gradient():
    config = make_config()
    pg = ActorCriticTrainer(DummyTask, config, seed=1, device='cpu')

    if pg.baseline_net.Nout != 1:
        raise AssertionError(f"baseline Nout should be 1, got {pg.baseline_net.Nout}")

    trial = pg.task.get_condition(np.random.RandomState(1), config['dt'])
    results = pg.run_trials([trial], return_states=True)
    z_b = results['Z_b']
    if z_b.dim() != 2:
        raise AssertionError(f"Z_b should have shape (T, B), got {tuple(z_b.shape)}")

    if not torch.isfinite(z_b).all():
        raise AssertionError("Z_b contains non-finite values")


def test_positive_opal_plasticity_components():
    config = make_config()
    config.update({
        'use_opponent_modulation': True,
        'positive_policy_readout': True,
        'pathway_specific_plasticity': True,
        'opal_alpha_d1': 0.5,
        'opal_alpha_d2': 2.0,
        'actor_weight_learning_modulation': True,
        'actor_weight_learning_floor': 0.0,
        'actor_weight_learning_max': None,
        'actor_weight_learning_normalize': False,
        'positive_readout_weight_l2': 1e-4,
        'opponent_pull_l2': 1e-5,
    })
    pg = ActorCriticTrainer(DummyTask, config, seed=1, device='cpu')

    delta = torch.tensor([[2.0, -3.0]])
    eta_plus = torch.ones_like(delta)
    eta_minus = torch.ones_like(delta)
    d1_advantage, d2_advantage = pg._opal_choice_advantages(
        delta, eta_plus, eta_minus
    )
    torch.testing.assert_close(d1_advantage, torch.tensor([[1.0, -1.5]]))
    torch.testing.assert_close(d2_advantage, torch.tensor([[4.0, -6.0]]))

    # D2 is subtracted in the logits: positive PE must reduce the selected
    # NoGo weight; negative PE must increase it.
    n_selected = torch.tensor(0.2, requires_grad=True)
    n_other = torch.tensor(0.2, requires_grad=True)
    logpi = torch.log_softmax(
        torch.stack([torch.tensor(1.0) - n_selected, torch.tensor(0.0) - n_other]),
        dim=0,
    )[0]
    for pe, expected_direction in [(1.0, -1.0), (-1.0, 1.0)]:
        n_selected.grad = None
        (logpi * pe).backward(retain_graph=True)
        if torch.sign(n_selected.grad).item() != expected_direction:
            raise AssertionError(
                f"D2 selected-weight gradient has wrong direction for PE={pe}: "
                f"{n_selected.grad.item()}"
            )

    with torch.no_grad():
        pg.policy_net.Wout.copy_(
            torch.linspace(-2.0, 2.0, pg.policy_net.Wout.numel()).reshape_as(pg.policy_net.Wout)
        )
    pg.policy_net.Wout.grad = torch.ones_like(pg.policy_net.Wout)
    pg._apply_actor_weight_learning_modulation()
    torch.testing.assert_close(
        pg.policy_net.Wout.grad,
        F.softplus(pg.policy_net.Wout.detach()),
    )

    rates = torch.ones(2, 1, pg.policy_net.N)
    mask = torch.ones(2, 1)
    effective_weights = F.softplus(pg.policy_net.Wout)
    half_n = pg.policy_net.N // 2
    d1_pull = torch.matmul(rates[..., :half_n], effective_weights[:half_n])
    d2_pull = torch.matmul(rates[..., half_n:], effective_weights[half_n:])
    expected_reg = (
        1e-4 * torch.mean(effective_weights ** 2)
        + 1e-5 * torch.mean(d1_pull ** 2 + d2_pull ** 2)
    )
    torch.testing.assert_close(pg.policy_net.get_readout_regs(rates, mask), expected_reg)


def main():
    test_scalar_policy_gradient()
    test_positive_opal_plasticity_components()
    print("ActorCriticTrainer scalar smoke test passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
