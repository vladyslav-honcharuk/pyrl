"""
Analysis script for gambling task - generates heatmaps like Nakazawa poster.

Creates behavioral heatmaps showing:
- Proportion of rightward choices as a function of ΔHH-LL and ΔEV
- ΔHH-LL: log probability difference (log(prob_R) - log(prob_L))
- ΔEV: log expected value difference (log(EV_R) - log(EV_L))
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pyrl import utils

def load_gambling_data(datapath, name):
    """Load gambling task trial data."""
    trialsfile = os.path.join(datapath, 'gambling.pkl')

    if not os.path.exists(trialsfile):
        print(f"File not found: {trialsfile}")
        print("Run the model first with: python do.py models/gambling.py run")
        return None

    data = utils.load(trialsfile)

    # Check what's in the data
    print(f"Data contains {len(data)} elements")

    # Handle both behavior-only and activity formats
    if len(data) == 5:
        # Behavior only: [trials, A, R, M, perf]
        trials, A, R, M, perf = data
    elif len(data) == 10:
        # Activity format: [trials, U, Z, Z_b, A, R, M, perf, r_policy, r_value]
        trials, U, Z, Z_b, A, R, M, perf, r_policy, r_value = data
    else:
        print(f"Unexpected data format with {len(data)} elements")
        return None

    return trials, A, R, M, perf


def analyze_choices(trials, A):
    """
    Analyze gambling choices and compute behavioral heatmap.

    Parameters
    ----------
    trials : list
        List of trial dictionaries with prob_l, prob_r, size_l, size_r
    A : ndarray
        Actions over time (timesteps x trials)

    Returns
    -------
    heatmap : ndarray
        Proportion of rightward choices in each (ΔHH-LL, ΔEV) bin
    delta_hh_ll_bins : ndarray
        Bin edges for ΔHH-LL
    delta_ev_bins : ndarray
        Bin edges for ΔEV
    """
    n_trials = len(trials)

    # Extract choice from each trial (first non-fixate action)
    choices = []
    delta_hh_ll = []
    delta_ev = []

    for i, trial in enumerate(trials):
        # Get actions for this trial
        trial_actions = A[:, i]

        # Find first choice (action 1=LEFT, 2=RIGHT)
        choice_idx = np.where((trial_actions == 1) | (trial_actions == 2))[0]

        if len(choice_idx) > 0:
            choice = trial_actions[choice_idx[0]]

            # Get trial parameters
            prob_l = trial['prob_l']
            prob_r = trial['prob_r']
            size_l = trial['size_l']
            size_r = trial['size_r']

            # Calculate expected values
            ev_l = prob_l * size_l
            ev_r = prob_r * size_r

            # Calculate deltas (avoid log(0))
            if prob_l > 0 and prob_r > 0 and ev_l > 0 and ev_r > 0:
                dhh_ll = np.log(prob_r) - np.log(prob_l)
                dev = np.log(ev_r) - np.log(ev_l)

                choices.append(choice)
                delta_hh_ll.append(dhh_ll)
                delta_ev.append(dev)

    choices = np.array(choices)
    delta_hh_ll = np.array(delta_hh_ll)
    delta_ev = np.array(delta_ev)

    print(f"\nAnalyzed {len(choices)}/{n_trials} trials with valid choices")
    print(f"Right choices: {np.sum(choices == 2)} ({100*np.mean(choices == 2):.1f}%)")
    print(f"Left choices: {np.sum(choices == 1)} ({100*np.mean(choices == 1):.1f}%)")

    # Create bins for heatmap (7x7 like in poster)
    hh_ll_bins = np.linspace(-2.2, 2.2, 8)  # 7 bins
    ev_bins = np.linspace(-0.9, 0.9, 8)      # 7 bins

    # Create heatmap
    heatmap = np.full((7, 7), np.nan)

    for i in range(7):
        for j in range(7):
            # Find trials in this bin
            mask = ((delta_hh_ll >= hh_ll_bins[i]) & (delta_hh_ll < hh_ll_bins[i+1]) &
                    (delta_ev >= ev_bins[j]) & (delta_ev < ev_bins[j+1]))

            if np.sum(mask) > 0:
                # Proportion choosing right (action=2)
                heatmap[j, i] = np.mean(choices[mask] == 2)

    return heatmap, hh_ll_bins, ev_bins


def plot_heatmap(heatmap, hh_ll_bins, ev_bins, savefile=None):
    """
    Plot behavioral heatmap like Nakazawa poster.

    Parameters
    ----------
    heatmap : ndarray
        Proportion of rightward choices in each bin
    hh_ll_bins : ndarray
        Bin edges for ΔHH-LL axis
    ev_bins : ndarray
        Bin edges for ΔEV axis
    savefile : str, optional
        Path to save figure
    """
    # Calculate aspect ratio for square cells
    x_range = hh_ll_bins[-1] - hh_ll_bins[0]  # 4.4
    y_range = ev_bins[-1] - ev_bins[0]         # 1.8
    aspect_ratio = y_range / x_range

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)

    # Main heatmap
    ax_main = fig.add_subplot(gs[0])
    im = ax_main.imshow(
        heatmap,
        origin='lower',
        cmap='plasma',
        vmin=0, vmax=1,
        extent=[hh_ll_bins[0], hh_ll_bins[-1], ev_bins[0], ev_bins[-1]],
        aspect=1/aspect_ratio,
        interpolation='nearest'
    )

    ax_main.set_title('Proportion of Rightward Choices', fontsize=14, pad=20)
    ax_main.set_xlabel('ΔHH-LL (log probability difference)', fontsize=12)
    ax_main.set_ylabel('ΔEV (log expected value difference)', fontsize=12)
    ax_main.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax_main.axvline(0, color='white', linestyle='--', alpha=0.3, linewidth=1)

    # Colorbar
    ax_cbar = fig.add_subplot(gs[1])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('P(Right Choice)', fontsize=10)

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {savefile}")

    plt.show()


def main():
    """Main analysis function."""
    # Setup paths
    here = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.join(here, '..','examples', 'work', 'data', 'gambling')
    figspath = os.path.join(here, '..','examples', 'work', 'figs', 'gambling')

    # Create output directory if needed
    os.makedirs(figspath, exist_ok=True)

    # Model name
    name = 'gambling'

    print("="*60)
    print("Gambling Task Analysis")
    print("="*60)

    # Load data
    print(f"\nLoading data from {datapath}...")
    result = load_gambling_data(datapath, name)

    if result is None:
        return

    trials, A, R, M, perf = result
    print(f"Loaded {len(trials)} trials")

    # Analyze choices
    print("\nAnalyzing choice behavior...")
    heatmap, hh_ll_bins, ev_bins = analyze_choices(trials, A)

    # Print summary statistics
    print("\nHeatmap statistics:")
    valid_cells = ~np.isnan(heatmap)
    print(f"  Valid cells: {np.sum(valid_cells)}/49")
    print(f"  Mean P(right): {np.nanmean(heatmap):.3f}")
    print(f"  Min P(right): {np.nanmin(heatmap):.3f}")
    print(f"  Max P(right): {np.nanmax(heatmap):.3f}")

    # Plot
    print("\nGenerating heatmap plot...")
    savefile = os.path.join(figspath, 'gambling_behavior.png')
    plot_heatmap(heatmap, hh_ll_bins, ev_bins, savefile=savefile)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
