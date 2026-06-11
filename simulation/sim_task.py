"""Gambling task spec tuned for the live 3D simulation.

Re-exports the real gambling task (same inputs, actions, options, colors and
trial dynamics as ``tasks/gambling.py``) but shrinks the network and training
budget so the learning process is watchable in real time inside the browser.
The behaviour the agent learns is identical in kind to the full model; only
the capacity and iteration count are reduced for responsiveness.

Bump ``max_iter`` / ``N`` toward the values in ``pyrl/configs.py`` if you want
fully-converged behaviour at the cost of a longer (still streamed) training run.
"""

from tasks.gambling import *  # noqa: F401,F403  (inputs, actions, get_condition, get_step, value_vector, color_vector, ...)

# Smaller networks train fast enough to watch live.
N = 50
baseline_N = 50

# Frequent validation so the learning curve updates often.
max_iter = 600
n_gradient = 24
n_validation = 100
checkfreq = 10
