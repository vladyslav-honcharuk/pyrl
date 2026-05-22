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
    pg : ActorCriticTrainer
        Trainer instance
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
        A = results['A']
        R = results['R']
        M = results['M']
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
        perf = results['perf']
        r_policy = results['r_policy']
        r_value = results['r_value']
        
        # Include RPE signals if available (added in updated run_trials)
        RPE_objective = results.get('RPE_objective', None)
        RPE_subjective = results.get('RPE_subjective', None)
        Policy_Values = results.get('Policy_Values', None)
        Policy_D1_Pull = results.get('Policy_D1_Pull', None)
        Policy_D2_Pull = results.get('Policy_D2_Pull', None)
        r_policy_mod = results.get('r_policy_mod', None)

        for trial in trials:
            trial['time'] = trial['time'][::inc]
        
        # Save with RPE signals if available
        if RPE_objective is not None and RPE_subjective is not None:
            save = [trials, U[::inc], Z[::inc], Z_b[::inc], A[::inc], R[::inc],
                    M[::inc], perf, r_policy[::inc], r_value[::inc],
                    RPE_objective[::inc], RPE_subjective[::inc]]
            print("Including RPE signals in saved data.")
        else:
            # Backward compatibility: save without RPE if not available
            save = [trials, U[::inc], Z[::inc], Z_b[::inc], A[::inc], R[::inc],
                    M[::inc], perf, r_policy[::inc], r_value[::inc]]
            print("Warning: RPE signals not found in results (using older version?)")

        if (
            Policy_Values is not None
            and Policy_D1_Pull is not None
            and Policy_D2_Pull is not None
        ):
            save.extend([
                Policy_Values[::inc],
                Policy_D1_Pull[::inc],
                Policy_D2_Pull[::inc],
                (r_policy_mod if r_policy_mod is not None else r_policy)[::inc]
            ])
            print("Including policy split logits in saved data.")
    else:
        raise ValueError(action)

    # Performance
    perf.display()

    # Save
    utils.save(trialsfile, save)

    # File size
    size_in_bytes = os.path.getsize(trialsfile)
    print("File size: {:.1f} MB".format(size_in_bytes / 2**20))
