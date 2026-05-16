#!/usr/bin/env python
"""
Extract EV and RPE for all timesteps for a few trials from a trained model.
"""
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from pyrl import utils
from pyrl.model import Model


def extract_ev_rpe(modelfile, n_trials=5):
    """
    Run trials and extract Expected Value and RPE at each timestep.

    Returns
    -------
    results : dict
        - trials: list of trial specs
        - V: (T, n_trials) - Critic's value predictions
        - R: (T, n_trials) - Immediate rewards
        - EV_left: (n_trials,) - Expected value of left option
        - EV_right: (n_trials,) - Expected value of right option
        - RPE: (T, n_trials) - Reward prediction error (TD error)
        - choices: (n_trials,) - Agent's choices ('L', 'R', or None)
    """
    # Load model
    model = Model('tasks/gambling.py')
    pg = model.get_pg(modelfile, seed=100, load='best')

    # Generate trials
    trials = [pg.task.get_condition(pg.rng, pg.dt) for _ in range(n_trials)]

    # Run trials
    results = pg.run_trials(trials, return_states=True)

    # Extract data
    V = results['Z_b'].cpu().numpy()  # (T, B) - Value predictions
    R = results['R'].cpu().numpy()    # (T, B) - Rewards
    M = results['M'].cpu().numpy()    # (T, B) - Mask
    T, B = R.shape

    # Check if continuous RPE was computed (when use_rpe_modulation=True)
    if 'RPE_continuous' in results:
        RPE = results['RPE_continuous'].cpu().numpy()
        rpe_source = "continuous (online)"
    else:
        # Compute RPE (TD error): δ(t) = r(t) + γ*V(t+1) - V(t)
        gamma = pg.gamma
        RPE = np.zeros_like(R)

        for t in range(T):
            if t < T - 1:
                RPE[t] = R[t] + gamma * V[t+1] * M[t+1] - V[t]
            else:
                RPE[t] = R[t] - V[t]  # Terminal state
        rpe_source = "post-hoc"

    # Extract trial info
    EV_left = []
    EV_right = []
    choices = []

    for trial in trials:
        EV_left.append(trial['prob_l'] * trial['size_l'])
        EV_right.append(trial['prob_r'] * trial['size_r'])
        choices.append(trial.get('choice', None))

    return {
        'trials': trials,
        'V': V,
        'R': R,
        'M': M,
        'EV_left': np.array(EV_left),
        'EV_right': np.array(EV_right),
        'RPE': RPE,
        'rpe_source': rpe_source,
        'choices': choices,
        'timesteps': T
    }


def print_trial_summary(data, trial_idx=0):
    """Print detailed summary for one trial."""
    T = data['timesteps']
    trial = data['trials'][trial_idx]

    print(f"\n{'='*80}")
    print(f"TRIAL {trial_idx + 1}")
    print(f"{'='*80}")
    print(f"Left:  p={trial['prob_l']:.1f}, size={trial['size_l']:.1f}, EV={data['EV_left'][trial_idx]:.2f}")
    print(f"Right: p={trial['prob_r']:.1f}, size={trial['size_r']:.1f}, EV={data['EV_right'][trial_idx]:.2f}")
    print(f"Choice: {data['choices'][trial_idx]}")
    print(f"RPE computed: {data.get('rpe_source', 'unknown')}")
    print(f"\n{'Time':>4} {'V(s)':>8} {'R(t)':>8} {'RPE(t)':>10} {'Epoch':<12}")
    print('-' * 80)

    # Define epochs (from gambling.py)
    fixation = range(0, 25)
    stimulus = range(25, 50)
    decision = range(50, 77)

    for t in range(T):
        if data['M'][t, trial_idx] == 0:
            break  # End of trial

        V_t = data['V'][t, trial_idx]
        R_t = data['R'][t, trial_idx]
        RPE_t = data['RPE'][t, trial_idx]

        # Determine epoch
        if t in fixation:
            epoch = "Fixation"
        elif t in stimulus:
            epoch = "Stimulus"
        elif t in decision:
            epoch = "Decision"
        else:
            epoch = "Post-trial"

        print(f"{t:4d} {V_t:8.4f} {R_t:8.4f} {RPE_t:10.4f} {epoch:<12}")


def append_terminal_rpe_timestep(data):
    """
    Add one post-trial timestep containing the terminal reward RPE.

    Online RPE is computed before the current timestep's reward is observed, so
    the final reward can be missing from the plotted RPE trace. This appends a
    terminal next-state point with V=0 and RPE = R(final) - V(final).
    """
    V = data['V']
    R = data['R']
    M = data['M']
    RPE = data['RPE']
    T, B = R.shape

    V_ext = np.zeros((T + 1, B), dtype=V.dtype)
    R_ext = np.zeros((T + 1, B), dtype=R.dtype)
    M_ext = np.zeros((T + 1, B), dtype=M.dtype)
    M_value_ext = np.zeros((T + 1, B), dtype=M.dtype)
    RPE_ext = np.zeros((T + 1, B), dtype=RPE.dtype)
    terminal_rpe_mask = np.zeros((T + 1, B), dtype=bool)

    V_ext[:T] = V
    R_ext[:T] = R
    M_ext[:T] = M
    M_value_ext[:T] = M
    RPE_ext[:T] = RPE

    for i in range(B):
        valid = np.where(M[:, i] > 0)[0]
        if len(valid) == 0:
            continue
        final_t = valid[-1]
        terminal_t = final_t + 1

        V_ext[terminal_t, i] = 0.0
        R_ext[terminal_t, i] = 0.0
        M_ext[terminal_t, i] = 1.0
        RPE_ext[terminal_t, i] = R[final_t, i] - V[final_t, i]
        terminal_rpe_mask[terminal_t, i] = True

    data = dict(data)
    data['V'] = V_ext
    data['R'] = R_ext
    data['M'] = M_ext
    data['M_value'] = M_value_ext
    data['RPE'] = RPE_ext
    data['terminal_rpe_mask'] = terminal_rpe_mask
    data['timesteps'] = T + 1
    data['rpe_source'] = f"{data.get('rpe_source', 'unknown')} + terminal next-step"
    return data


def plot_v_rpe(data, save_path=None):
    """
    Plot V(t) and RPE(t) for all trials.

    Creates two subplots:
    - Top: Value function V(t) across time
    - Bottom: RPE(t) across time

    Vertical lines mark task epochs (fixation, stimulus, decision).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    T = data['timesteps']
    n_trials = data['V'].shape[1]

    # Define epoch boundaries
    fixation_end = 25
    stimulus_end = 50
    decision_end = 77

    # Plot V(t)
    ax = axes[0]
    for i in range(n_trials):
        trial = data['trials'][i]
        choice = data['choices'][i]
        total_reward = np.sum(data['R'][:, i] * data['M'][:, i])

        # Find valid timesteps
        value_mask = data.get('M_value', data['M'])
        valid = value_mask[:, i] > 0
        t_vals = np.arange(T)[valid]
        V_vals = data['V'][valid, i]

        # Label with trial info
        label = f"T{i+1}: {choice}, R={total_reward:.1f}"
        ax.plot(t_vals, V_vals, '-o', markersize=2, alpha=0.7, label=label)

    # Mark epochs
    ax.axvline(fixation_end, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(stimulus_end, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvspan(0, fixation_end, alpha=0.1, color='blue', label='Fixation')
    ax.axvspan(fixation_end, stimulus_end, alpha=0.1, color='green', label='Stimulus')
    ax.axvspan(stimulus_end, decision_end, alpha=0.1, color='orange', label='Decision')

    ax.set_ylabel('V(t) - Value Prediction', fontsize=12)
    ax.set_title('Value Function Dynamics Across Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot RPE(t)
    ax = axes[1]
    rpe_source_str = data.get('rpe_source', 'unknown')
    terminal_mask = data.get('terminal_rpe_mask', np.zeros_like(data['M'], dtype=bool))
    nonterminal_rpe = data['RPE'][(data['M'] > 0) & (~terminal_mask)]
    if nonterminal_rpe.size > 0:
        y_min, y_max = np.nanmin(nonterminal_rpe), np.nanmax(nonterminal_rpe)
        y_pad = max(0.05, 0.12 * (y_max - y_min))
        rpe_ylim = (y_min - y_pad, y_max + y_pad)
    else:
        rpe_ylim = None

    for i in range(n_trials):
        trial = data['trials'][i]
        choice = data['choices'][i]
        total_reward = np.sum(data['R'][:, i] * data['M'][:, i])

        # Find valid timesteps
        valid = (data['M'][:, i] > 0) & (~terminal_mask[:, i])
        t_vals = np.arange(T)[valid]
        RPE_vals = data['RPE'][valid, i]

        # Color based on win/loss
        color = 'green' if total_reward > 0 else 'red'
        label = f"T{i+1}: {choice}, R={total_reward:.1f}"
        ax.plot(t_vals, RPE_vals, '-o', markersize=2, alpha=0.7, color=color, label=label)

        terminal_idx = np.where(terminal_mask[:, i])[0]
        if len(terminal_idx) > 0 and rpe_ylim is not None:
            t_terminal = terminal_idx[0]
            rpe_terminal = data['RPE'][t_terminal, i]
            y0 = 0.0
            y1 = np.clip(rpe_terminal, rpe_ylim[0], rpe_ylim[1])
            ax.plot([t_terminal, t_terminal], [y0, y1], '--', color=color, alpha=0.65, linewidth=1.4)
            ax.plot(t_terminal, y1, 'o', color=color, markersize=4, clip_on=False)
            text_y = rpe_ylim[1] if rpe_terminal >= 0 else rpe_ylim[0]
            va = 'top' if rpe_terminal >= 0 else 'bottom'
            ax.annotate(
                f'{rpe_terminal:+.2f}',
                xy=(t_terminal, y1),
                xytext=(4, -4 if rpe_terminal >= 0 else 4),
                textcoords='offset points',
                fontsize=8,
                color=color,
                va=va,
                ha='left',
                clip_on=False
            )

    # Mark epochs
    ax.axvline(fixation_end, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(stimulus_end, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvspan(0, fixation_end, alpha=0.1, color='blue')
    ax.axvspan(fixation_end, stimulus_end, alpha=0.1, color='green')
    ax.axvspan(stimulus_end, decision_end, alpha=0.1, color='orange')
    if rpe_ylim is not None:
        ax.set_ylim(rpe_ylim)

    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('RPE(t) - Reward Prediction Error', fontsize=12)
    title = f'Reward Prediction Error Dynamics ({rpe_source_str})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract EV and RPE from model')
    parser.add_argument('modelfile', help='Path to model .pkl file')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--trial_idx', type=int, default=None, help='Which trial to print in detail (default: all)')
    parser.add_argument('--save', help='Save results to .npz file')
    parser.add_argument('--plot', help='Save plot to file (e.g., output.png)')
    parser.add_argument('--append_terminal_rpe', action='store_true',
                        help='Append one post-trial timestep with terminal reward RPE = R(final) - V(final)')
    parser.add_argument('--no_print', action='store_true', help='Skip text output, only plot')

    args = parser.parse_args()

    # Extract data
    print(f"Loading model: {args.modelfile}")
    data = extract_ev_rpe(args.modelfile, n_trials=args.n_trials)
    if args.append_terminal_rpe:
        data = append_terminal_rpe_timestep(data)

    # Print summary for one or all trials (unless --no_print)
    if not args.no_print:
        if args.trial_idx is not None:
            print_trial_summary(data, trial_idx=args.trial_idx)
        else:
            # Print all trials
            for i in range(args.n_trials):
                print_trial_summary(data, trial_idx=i)

        # Print all trials overview
        print(f"\n{'='*80}")
        print("ALL TRIALS OVERVIEW")
        print(f"{'='*80}")
        print(f"{'Trial':>5} {'EV_L':>6} {'EV_R':>6} {'Choice':>7} {'Total R':>8} {'Final V':>8}")
        print('-' * 80)

        for i in range(args.n_trials):
            total_reward = np.sum(data['R'][:, i] * data['M'][:, i])
            final_V = data['V'][-1, i]
            print(f"{i+1:5d} {data['EV_left'][i]:6.2f} {data['EV_right'][i]:6.2f} "
                  f"{str(data['choices'][i]):>7} {total_reward:8.2f} {final_V:8.4f}")

    # Generate plot
    if args.plot:
        plot_v_rpe(data, save_path=args.plot)

    # Save if requested
    if args.save:
        np.savez(args.save, **data)
        print(f"\nSaved results to: {args.save}")
