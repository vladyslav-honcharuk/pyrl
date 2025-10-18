"""Tools for running and saving trial results."""

import os
from . import utils


def behaviorfile(path):
    """Get path for behavior-only trial file."""
    return os.path.join(path, 'trials_behavior.pkl')


def activityfile(path):
    """Get path for behavior+activity trial file."""
    return os.path.join(path, 'trials_activity.pkl')


def run(action, trials, pg, scratchpath, dt_save=None):
    """
    Run trials and save results.

    Parameters
    ----------
    action : str
        'trials-b' for behavior only, 'trials-a' for behavior+activity
    trials : list
        List of trial specifications
    pg : PolicyGradient
        PolicyGradient instance
    scratchpath : str
        Path to save results
    dt_save : float, optional
        Timestep for saving (will downsample if different from pg.dt)
    """
    if dt_save is not None:
        dt = pg.dt
        inc = int(dt_save / dt)
    else:
        inc = 1
    print("Saving in increments of {}".format(inc))

    # Run trials
    if action == 'trials-b':
        print("Saving behavior only.")
        trialsfile = behaviorfile(scratchpath)

        results = pg.run_trials(trials, progress_bar=True)
        U = results['U']
        Z = results['Z']
        Z_b = results['Z_b']
        A = results['A']
        R = results['R']
        M = results['M']
        init = results['init']
        init_b = results['init_b']
        states_0 = results['states_0']
        states_0_b = results['states_0_b']
        perf = results['perf']

        for trial in trials:
            trial['time'] = trial['time'][::inc]
        save = [trials, A[::inc], R[::inc], M[::inc], perf]
    elif action == 'trials-a':
        print("Saving behavior + activity.")
        trialsfile = activityfile(scratchpath)

        results = pg.run_trials(trials, return_states=True, progress_bar=True)
        U = results['U']
        Z = results['Z']
        Z_b = results['Z_b']
        A = results['A']
        R = results['R']
        M = results['M']
        init = results['init']
        init_b = results['init_b']
        states_0 = results['states_0']
        states_0_b = results['states_0_b']
        perf = results['perf']
        states = results['states']
        states_b = results['states_b']

        for trial in trials:
            trial['time'] = trial['time'][::inc]
        save = [trials, U[::inc], Z[::inc], Z_b[::inc], A[::inc], R[::inc],
                M[::inc], perf, states[::inc], states_b[::inc]]
    else:
        raise ValueError(action)

    # Performance
    perf.display()

    # Save
    utils.save(trialsfile, save)

    # File size
    size_in_bytes = os.path.getsize(trialsfile)
    print("File size: {:.1f} MB".format(size_in_bytes / 2**20))
