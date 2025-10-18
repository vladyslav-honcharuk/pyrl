"""Visualization tools for trials and network activity."""

import numpy as np


def plot_trial(trial_info, trial, figspath, name):
    """
    Plot a single trial showing observables, policy, actions, and rewards.

    NOTE: This function requires figtools.py to be ported for full functionality.
    It is included here as a placeholder for compatibility.

    Parameters
    ----------
    trial_info : tuple
        (U, Z, A, R, M, init, states_0, perf)
    trial : dict
        Trial specification
    figspath : str
        Path to save figure
    name : str
        Figure name
    """
    try:
        from .figtools import Figure
    except ImportError:
        print("Warning: figtools.py not available. Cannot plot trial.")
        print("Please port figtools.py from the original pyrl for full visualization support.")
        return

    U, Z, A, R, M, init, states_0, perf = trial_info
    U = U[:, 0, :]
    Z = Z[:, 0, :]
    A = A[:, 0, :]
    R = R[:, 0]
    M = M[:, 0]
    t = int(np.sum(M))

    w = 0.65
    h = 0.18
    x = 0.17
    dy = h + 0.05
    y0 = 0.08
    y1 = y0 + dy
    y2 = y1 + dy
    y3 = y2 + dy

    fig = Figure(h=6)
    plots = {'observables': fig.add([x, y3, w, h]),
             'policy': fig.add([x, y2, w, h]),
             'actions': fig.add([x, y1, w, h]),
             'rewards': fig.add([x, y0, w, h])}

    time = trial['time']
    dt = time[1] - time[0]
    act_time = time[:t]
    obs_time = time[:t-1] + dt
    reward_time = act_time + dt
    xlim = (0, max(time))

    # Observables
    plot = plots['observables']
    plot.plot(obs_time, U[:t-1, 0], 'o', ms=5, mew=0, mfc=Figure.colors('blue'))
    plot.plot(obs_time, U[:t-1, 0], lw=1.25, color=Figure.colors('blue'), label='Input 0')
    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Observables')

    # Policy
    plot = plots['policy']
    for i in range(Z.shape[1]):
        plot.plot(act_time, Z[:t, i], lw=1.25, label=f'Action {i}')
    plot.xlim(*xlim)
    plot.ylim(0, 1)
    plot.ylabel('Action probabilities')

    # Actions
    plot = plots['actions']
    actions = [np.argmax(a) for a in A[:t]]
    plot.plot(act_time, actions, 'o', ms=5, mew=0)
    plot.plot(act_time, actions, lw=1.25)
    plot.xlim(*xlim)
    plot.ylabel('Action')

    # Rewards
    plot = plots['rewards']
    plot.plot(reward_time, R[:t], 'o', ms=5, mew=0)
    plot.plot(reward_time, R[:t], lw=1.25)
    plot.xlim(*xlim)
    plot.xlabel('Time (ms)')
    plot.ylabel('Reward')

    fig.save(path=figspath, name=name)
    fig.close()
