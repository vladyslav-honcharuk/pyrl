#!/usr/bin/env python
"""
Test if cumulative RPE during stimulus period correlates with final RPE at reward.
"""
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from pyrl.model import Model


def analyze_rpe_correlation(modelfile, n_trials=100):
    """
    Compute correlation between cumulative stimulus RPE and final RPE.

    Returns
    -------
    results : dict
        - cumulative_stim_rpe: (n_trials,) sum of RPE during stimulus
        - final_rpe: (n_trials,) RPE at reward delivery
        - correlation: Pearson correlation coefficient
        - p_value: Statistical significance
    """
    # Load model
    model = Model('tasks/gambling.py')
    pg = model.get_pg(modelfile, seed=100, load='best')

    # Generate trials
    trials = [pg.task.get_condition(pg.rng, pg.dt) for _ in range(n_trials)]

    # Run trials
    results = pg.run_trials(trials, return_states=True)

    # Extract data
    V = results['Z_b'].cpu().numpy()  # (T, B)
    R = results['R'].cpu().numpy()    # (T, B)
    M = results['M'].cpu().numpy()    # (T, B)

    # Compute RPE (TD error): δ(t) = r(t) + γ*V(t+1) - V(t)
    gamma = pg.gamma
    T, B = R.shape
    RPE = np.zeros_like(R)

    for t in range(T):
        if t < T - 1:
            RPE[t] = R[t] + gamma * V[t+1] * M[t+1] - V[t]
        else:
            RPE[t] = R[t] - V[t]

    # Define epochs (from gambling.py)
    stimulus_period = range(25, 50)  # Stimulus presentation

    # Storage
    cumulative_stim_rpe = []
    final_rpe = []
    reward_received = []
    ev_chosen = []

    for i in range(n_trials):
        # Sum RPE during stimulus period
        stim_rpe = np.sum(RPE[stimulus_period, i])
        cumulative_stim_rpe.append(stim_rpe)

        # Find when trial ended (first invalid timestep)
        trial_end = np.where(M[:, i] == 0)[0]
        if len(trial_end) > 0:
            last_valid = trial_end[0] - 1
        else:
            last_valid = T - 1

        # Final RPE (at reward delivery or last timestep)
        final_rpe.append(RPE[last_valid, i])

        # Reward received
        reward_received.append(R[last_valid, i])

        # EV of chosen option
        trial = trials[i]
        choice = trial.get('choice', None)
        if choice == 'L':
            ev_chosen.append(trial['prob_l'] * trial['size_l'])
        elif choice == 'R':
            ev_chosen.append(trial['prob_r'] * trial['size_r'])
        else:
            ev_chosen.append(np.nan)

    cumulative_stim_rpe = np.array(cumulative_stim_rpe)
    final_rpe = np.array(final_rpe)
    reward_received = np.array(reward_received)
    ev_chosen = np.array(ev_chosen)

    # Compute correlation
    valid = ~np.isnan(cumulative_stim_rpe) & ~np.isnan(final_rpe)
    if np.sum(valid) > 3:
        r, p = stats.pearsonr(cumulative_stim_rpe[valid], final_rpe[valid])
    else:
        r, p = np.nan, np.nan

    return {
        'cumulative_stim_rpe': cumulative_stim_rpe,
        'final_rpe': final_rpe,
        'reward_received': reward_received,
        'ev_chosen': ev_chosen,
        'correlation': r,
        'p_value': p,
        'n_trials': n_trials
    }


def plot_correlation(data, outfile=None):
    """Plot cumulative stimulus RPE vs final RPE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Scatter with regression line
    ax = axes[0]
    ax.scatter(data['cumulative_stim_rpe'], data['final_rpe'],
               alpha=0.5, s=30, color='steelblue')

    # Add regression line
    valid = ~np.isnan(data['cumulative_stim_rpe']) & ~np.isnan(data['final_rpe'])
    if np.sum(valid) > 3:
        z = np.polyfit(data['cumulative_stim_rpe'][valid],
                       data['final_rpe'][valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(data['cumulative_stim_rpe'][valid].min(),
                             data['cumulative_stim_rpe'][valid].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7)

    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Cumulative RPE (Stimulus Period)', fontsize=12)
    ax.set_ylabel('Final RPE (At Reward)', fontsize=12)
    ax.set_title(f'r = {data["correlation"]:.3f}, p = {data["p_value"]:.4f}',
                 fontsize=13)
    ax.grid(True, alpha=0.3)

    # Plot 2: Colored by reward received
    ax = axes[1]
    scatter = ax.scatter(data['cumulative_stim_rpe'], data['final_rpe'],
                        c=data['reward_received'], cmap='RdYlGn',
                        alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax, label='Reward Received')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Cumulative RPE (Stimulus Period)', fontsize=12)
    ax.set_ylabel('Final RPE (At Reward)', fontsize=12)
    ax.set_title('Colored by Reward Magnitude', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {outfile}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test RPE correlation')
    parser.add_argument('modelfile', help='Path to model .pkl file')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of trials to analyze')
    parser.add_argument('--plot', help='Save plot to file')

    args = parser.parse_args()

    print(f"Loading model: {args.modelfile}")
    print(f"Analyzing {args.n_trials} trials...")

    data = analyze_rpe_correlation(args.modelfile, n_trials=args.n_trials)

    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Cumulative Stimulus RPE range: [{data['cumulative_stim_rpe'].min():.3f}, "
          f"{data['cumulative_stim_rpe'].max():.3f}]")
    print(f"Final RPE range:                [{data['final_rpe'].min():.3f}, "
          f"{data['final_rpe'].max():.3f}]")
    print(f"\nPearson correlation:  r = {data['correlation']:.4f}")
    print(f"P-value:              p = {data['p_value']:.6f}")

    if data['p_value'] < 0.001:
        sig = "*** (highly significant)"
    elif data['p_value'] < 0.01:
        sig = "** (very significant)"
    elif data['p_value'] < 0.05:
        sig = "* (significant)"
    else:
        sig = "(not significant)"

    print(f"Significance:         {sig}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Mean cumulative stimulus RPE: {np.mean(data['cumulative_stim_rpe']):.4f}")
    print(f"Mean final RPE:               {np.mean(data['final_rpe']):.4f}")
    print(f"Mean reward received:         {np.mean(data['reward_received']):.4f}")

    if args.plot:
        plot_correlation(data, outfile=args.plot)
