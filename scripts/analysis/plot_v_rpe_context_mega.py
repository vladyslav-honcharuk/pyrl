#!/usr/bin/env python3
"""Create V/RPE dynamics mega-figures across context or fake-VTA levels."""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

from pyrl import utils


LEVELS = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
          0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def context_tag(level):
    return f'{level:+.2f}'.replace('.', 'p').replace('+', 'pos').replace('-', 'neg')


def opto_tag(level):
    return f'{level:+.3f}'.replace('.', 'p').replace('+', 'pos').replace('-', 'neg')


def to_numpy(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return np.asarray(x)


def load_trial_file(path):
    data = utils.load(path)
    result = {
        'trials': data[0],
        'Z_b': to_numpy(data[3]),
        'A': to_numpy(data[4]),
        'R': to_numpy(data[5]),
        'M': to_numpy(data[6]),
    }
    if len(data) >= 12:
        result['RPE_objective'] = to_numpy(data[10])
        result['RPE_subjective'] = to_numpy(data[11])
    return result


def first_choices(A):
    action_idx = A.argmax(axis=2)
    choices = np.full(action_idx.shape[1], -1, dtype=int)
    for trial_i in range(action_idx.shape[1]):
        times = np.where((action_idx[:, trial_i] == 1) | (action_idx[:, trial_i] == 2))[0]
        if times.size:
            choices[trial_i] = action_idx[times[0], trial_i]
    return choices


def compute_online_dopamine_signal(td, level, mode):
    """Approximate the action-time dopamine/RPE signal used by rollout modulation."""
    V = td['Z_b']
    R = td['R']
    M = td['M']
    signal = np.zeros_like(V)

    if mode == 'direct_context':
        # Direct context is not a TD error. Use the saved TD error for dynamics.
        if 'RPE_objective' in td:
            return td['RPE_objective'], 'Objective TD RPE'
        return np.zeros_like(V), 'Objective TD RPE unavailable'

    natural_gain = 0.0 if mode == 'fake_vta_zero_rpe' else 3.0
    for t in range(1, V.shape[0]):
        td_error = R[t - 1] + V[t] - V[t - 1]
        signal[t] = np.clip(natural_gain * td_error + level, -0.9, 0.9)

    label = 'Fake VTA dopamine offset' if natural_gain == 0.0 else 'Natural RPE + fake VTA dopamine'
    return signal * (M > 0), label


def trial_selection(td, n_trials):
    R = td['R']
    M = td['M']
    total_reward = (R * M).sum(axis=0)
    valid_trials = np.where(M.sum(axis=0) > 0)[0]
    if valid_trials.size <= n_trials:
        return valid_trials

    order = valid_trials[np.argsort(total_reward[valid_trials])]
    positions = np.linspace(0, len(order) - 1, n_trials).round().astype(int)
    return order[positions]


def trial_label(td, trial_idx, choice):
    trial = td['trials'][trial_idx]
    ev_l = float(trial['prob_l'] * trial['size_l'])
    ev_r = float(trial['prob_r'] * trial['size_r'])
    side = 'L' if choice == 1 else 'R' if choice == 2 else '-'
    reward = float((td['R'][:, trial_idx] * td['M'][:, trial_idx]).sum())
    return f'{side} R={reward:.1f} EV={ev_l:.1f}/{ev_r:.1f}'


def plot_level(ax_v, ax_rpe, td, level, mode, n_trials, colors):
    V = td['Z_b']
    M = td['M']
    signal, signal_label = compute_online_dopamine_signal(td, level, mode)
    choices = first_choices(td['A'])
    selected = trial_selection(td, n_trials)

    for color_idx, trial_idx in enumerate(selected):
        valid = np.where(M[:, trial_idx] > 0)[0]
        if valid.size == 0:
            continue
        color = colors[color_idx % len(colors)]
        label = trial_label(td, trial_idx, choices[trial_idx]) if color_idx < 3 else None
        ax_v.plot(valid, V[valid, trial_idx], color=color, linewidth=1.0, alpha=0.82, label=label)
        ax_rpe.plot(valid, signal[valid, trial_idx], color=color, linewidth=1.0, alpha=0.82)

    for ax in (ax_v, ax_rpe):
        ax.axvspan(0, 25, color='#4c78a8', alpha=0.08)
        ax.axvspan(25, 50, color='#59a14f', alpha=0.08)
        ax.axvspan(50, 77, color='#f28e2b', alpha=0.08)
        ax.axvline(25, color='0.65', linestyle='--', linewidth=0.6)
        ax.axvline(50, color='0.65', linestyle='--', linewidth=0.6)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        ax.tick_params(labelsize=7, width=0.7, length=2.5)
        ax.spines[['top', 'right']].set_visible(False)

    ax_rpe.axhline(0, color='0.2', linewidth=0.6, alpha=0.45)
    ax_v.set_ylabel(f'{level:+.1f}\nV', fontsize=8)
    ax_rpe.set_ylabel('RPE', fontsize=8)
    return signal_label


def make_mega(args):
    os.makedirs(args.output_dir, exist_ok=True)
    levels = LEVELS
    n_rows = len(levels)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(13.5, 2.05 * n_rows),
        sharex=True,
        constrained_layout=False,
    )
    colors = plt.cm.tab10(np.linspace(0, 1, args.n_trials))
    signal_label = 'RPE'

    for row, level in enumerate(levels):
        tag = context_tag(level) if args.kind == 'ctx' else opto_tag(level)
        path = os.path.join(args.trials_dir, f'trials_activity_{args.kind}{tag}.pkl')
        td = load_trial_file(path)
        signal_label = plot_level(
            axes[row, 0], axes[row, 1], td, level, args.mode, args.n_trials, colors
        )

    axes[0, 0].set_title('Critic value V(t), 10 trials per level', fontsize=12, weight='bold')
    axes[0, 1].set_title(signal_label, fontsize=12, weight='bold')
    axes[-1, 0].set_xlabel('Timestep', fontsize=11)
    axes[-1, 1].set_xlabel('Timestep', fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False, fontsize=6.5, loc='upper right')

    fig.suptitle(args.title, fontsize=16, weight='bold', y=0.998)
    fig.text(0.012, 0.5, args.level_label, rotation=90, va='center', ha='left', fontsize=11)
    fig.tight_layout(rect=[0.035, 0.01, 1, 0.992])

    outfile = os.path.join(args.output_dir, args.output_name)
    fig.savefig(outfile, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--kind', choices=['ctx', 'opto'], required=True)
    parser.add_argument('--mode', choices=['direct_context', 'fake_vta_zero_rpe', 'natural_fake_vta'], required=True)
    parser.add_argument('--level-label', default='Context / VTA level')
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    make_mega(args)


if __name__ == '__main__':
    main()
