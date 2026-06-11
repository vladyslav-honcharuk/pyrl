#!/usr/bin/env python3
"""Compare direct-context and fake-RPE runtime modulation for OpAL04 models."""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

from pyrl import utils
from pyrl.model import Model


def first_choices(actions):
    action_idx = actions.cpu().numpy().argmax(axis=2)
    choices = np.full(action_idx.shape[1], -1, dtype=int)
    for trial_i in range(action_idx.shape[1]):
        choice_times = np.where(
            (action_idx[:, trial_i] == 1) | (action_idx[:, trial_i] == 2)
        )[0]
        if choice_times.size:
            choices[trial_i] = action_idx[choice_times[0], trial_i]
    return choices


def controlled_trials(trials):
    selected = []
    for trial in trials:
        prob_l = float(trial['prob_l'])
        prob_r = float(trial['prob_r'])
        ev_l = prob_l * float(trial['size_l'])
        ev_r = prob_r * float(trial['size_r'])
        matched_ev = abs(ev_r - ev_l) < 0.01 and prob_l != prob_r
        matched_prob = prob_l == prob_r and abs(ev_r - ev_l) > 0.01
        if matched_ev or matched_prob:
            selected.append(trial)
    return selected


def trial_arrays(trials):
    prob_l = np.array([float(t['prob_l']) for t in trials])
    prob_r = np.array([float(t['prob_r']) for t in trials])
    ev_l = np.array([float(t['prob_l'] * t['size_l']) for t in trials])
    ev_r = np.array([float(t['prob_r'] * t['size_r']) for t in trials])
    matched_ev = (np.abs(ev_r - ev_l) < 0.01) & (prob_l != prob_r)
    matched_prob = (prob_l == prob_r) & (np.abs(ev_r - ev_l) > 0.01)
    return prob_l, prob_r, ev_l, ev_r, matched_ev, matched_prob


def summarize(results, trials):
    prob_l, prob_r, ev_l, ev_r, matched_ev, matched_prob = trial_arrays(trials)
    decisions = np.asarray(results['perf'].decisions, dtype=bool)
    choices = first_choices(results['A'])
    right = choices == 2
    risky = np.where(prob_r < prob_l, right, ~right)
    high_ev = np.where(ev_r > ev_l, right, ~right)
    risk_mask = matched_ev & decisions
    ev_mask = matched_prob & decisions
    return {
        'completion': float(np.mean(decisions)),
        'risk': float(np.mean(risky[risk_mask])) if np.any(risk_mask) else np.nan,
        'high_ev': float(np.mean(high_ev[ev_mask])) if np.any(ev_mask) else np.nan,
        'risk_n': int(np.sum(risk_mask)),
        'ev_n': int(np.sum(ev_mask)),
    }


def make_pg(model_file, savefile, seed, device):
    model = Model(model_file)
    return model.get_pg(savefile, seed=seed, load='best', device=device)


def run_direct(model_file, savefile, trials, level, phase, seed, device):
    pg = make_pg(model_file, savefile, seed, device)
    pg.rng = np.random.RandomState(seed)
    phases = None if phase == 'all' else (phase,)
    return summarize(
        pg.run_trials(
            trials,
            context_input=level,
            context_phases=phases,
            return_states=False,
            collect_policy_diagnostics=False,
        ),
        trials,
    )


def run_fake_rpe(model_file, savefile, trials, level, phase, seed, device,
                 natural_gain, uniform_fake):
    pg = make_pg(model_file, savefile, seed, device)
    pg.config['use_rpe_modulation'] = True
    pg.use_rpe_modulation = True
    pg.config['rpe_modulation_gain'] = float(natural_gain)
    pg.rpe_modulation_gain = float(natural_gain)
    pg.config['rpe_modulation_clamp'] = 0.9
    pg.rpe_modulation_clamp = 0.9

    if uniform_fake:
        pg.config['vta_training_context'] = True
        pg.config['vta_context_distribution'] = 'uniform'
        pg.config['vta_context_low'] = -0.9
        pg.config['vta_context_high'] = 0.9
        pg.config['vta_context_weight'] = 1.0
        pg.rng = np.random.RandomState(seed)
        results = pg.run_trials(
            trials,
            training=True,
            return_states=False,
            collect_policy_diagnostics=False,
        )
    else:
        pg.opto_stim_offset = level
        pg.opto_stim_gain = 1.0
        pg.opto_stim_phase = phase
        pg.rng = np.random.RandomState(seed)
        results = pg.run_trials(
            trials,
            return_states=False,
            collect_policy_diagnostics=False,
        )
    return summarize(results, trials)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', default='tasks/gambling.py')
    parser.add_argument('--vanilla-savefile', required=True)
    parser.add_argument('--natural-savefile', required=True)
    parser.add_argument('--trials-file', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--phase', default='decision',
                        choices=['all', 'fixation', 'cue', 'decision'])
    parser.add_argument('--uniform-fake-rpe', action='store_true',
                        help='Sample trial-wise fake RPE uniformly from [-0.9, 0.9].')
    args = parser.parse_args()

    trial_data = utils.load(args.trials_file)
    trials = controlled_trials(trial_data[0])
    levels = (-1.0, 0.0, 1.0)
    rows = []

    for model_name, savefile, natural_gain in (
        ('vanilla', args.vanilla_savefile, 0.0),
        ('natural_rpe', args.natural_savefile, 3.0),
    ):
        for channel in ('direct', 'fake_rpe'):
            for level in levels:
                if channel == 'direct':
                    stats = run_direct(
                        args.model_file, savefile, trials, level, args.phase,
                        args.seed, args.device
                    )
                else:
                    stats = run_fake_rpe(
                        args.model_file, savefile, trials, level, args.phase,
                        args.seed, args.device, natural_gain,
                        args.uniform_fake_rpe
                    )
                rows.append((model_name, channel, level, stats))

    print('model channel level completion risk_matchedEV highEV_matchedP nRisk nEV')
    for model_name, channel, level, stats in rows:
        print(
            f"{model_name} {channel} {level:+.1f} "
            f"{stats['completion']:.3f} {stats['risk']:.3f} "
            f"{stats['high_ev']:.3f} {stats['risk_n']} {stats['ev_n']}"
        )


if __name__ == '__main__':
    main()
