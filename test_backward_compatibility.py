"""Smoke checks for the current scalar actor-critic trainer."""

import sys

import numpy as np
import torch

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


def main():
    test_scalar_policy_gradient()
    print("ActorCriticTrainer scalar smoke test passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
