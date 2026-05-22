"""
Analysis for gambling task.

Data Format:
- Behavior only (5 elements): [trials, A, R, M, perf]
- Activity (10 elements): [trials, U, Z, Z_b, A, R, M, perf, r_policy, r_value]
- Activity + RPE (12 elements): [trials, U, Z, Z_b, A, R, M, perf, r_policy, r_value, RPE_objective, RPE_subjective]
"""

import os
import numpy as np
import torch
from pyrl import runtools, utils
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

# Disable LaTeX rendering to avoid Unicode issues
plt.rcParams['text.usetex'] = False

# ============================================================
# Helper Functions
# ============================================================


def to_numpy(x):
    """Convert torch tensor to numpy array if needed."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


def load_trial_data(trialsfile):
    """Load and unpack trial data into a consistent dict."""
    data = utils.load(trialsfile)
    result = {}

    if len(data) == 5:
        result['trials'], result['A'], result['R'], result['M'], _ = data
        result['format'] = 'behavior'
    elif len(data) >= 10:
        (result['trials'], _, _, result['Z_b'], result['A'], result['R'],
         result['M'], _, result['r_policy'], result['r_value']) = data[:10]
        result['r_policy_raw'] = result['r_policy']
        result['format'] = 'activity'
        
        if len(data) >= 12:
            result['RPE_objective'] = data[10]
            result['RPE_subjective'] = data[11]
            result['format'] = 'rpe'
            
        if len(data) >= 15:
            result['Policy_Values'] = data[12]
            result['Policy_D1_Pull'] = data[13]
            result['Policy_D2_Pull'] = data[14]
            result['format'] = 'subjective_values'

        if len(data) >= 16:
            result['r_policy_mod'] = data[15]
            result['r_policy'] = result['r_policy_mod']
            result['format'] = 'modulated_activity'

    return result


def convert_actions(A):
    """Convert actions from torch/one-hot to numpy action indices."""
    A_np = to_numpy(A)
    if A_np.ndim == 3:
        return np.argmax(A_np, axis=2)
    return A_np


def compute_deltas(trials):
    """
    Compute ΔHH-LL and ΔEV for each trial.

    Returns (delta_hh_lls, delta_evs, valid) arrays.
    Invalid trials (zero prob/EV) get delta=0 and valid=False.
    """
    delta_hh_lls, delta_evs, valid = [], [], []
    for trial in trials:
        prob_l, prob_r = trial['prob_l'], trial['prob_r']
        size_l, size_r = trial['size_l'], trial['size_r']
        ev_l, ev_r = prob_l * size_l, prob_r * size_r

        if prob_l > 0 and prob_r > 0 and ev_l > 0 and ev_r > 0:
            delta_hh_lls.append(np.log(prob_r) - np.log(prob_l))
            delta_evs.append(np.log(ev_r) - np.log(ev_l))
            valid.append(True)
        else:
            delta_hh_lls.append(0)
            delta_evs.append(0)
            valid.append(False)

    return np.array(delta_hh_lls), np.array(delta_evs), np.array(valid)


def extract_choices(trials, action_indices, delta_hh_lls=None, delta_evs=None, valid_mask=None):
    """
    Extract first choice (action 1 or 2) per trial.

    If delta arrays and valid_mask are provided, only includes trials
    where both a choice was made AND the delta is valid.

    Returns dict with 'choices' and optionally 'delta_hh_lls', 'delta_evs'.
    """
    choices, choice_hh, choice_ev = [], [], []
    filter_valid = delta_hh_lls is not None

    for i in range(len(trials)):
        trial_actions = action_indices[:, i]
        choice_idx = np.where((trial_actions == 1) | (trial_actions == 2))[0]

        if len(choice_idx) > 0:
            if filter_valid and valid_mask is not None and not valid_mask[i]:
                continue
            choices.append(trial_actions[choice_idx[0]])
            if filter_valid:
                choice_hh.append(delta_hh_lls[i])
                choice_ev.append(delta_evs[i])

    result = {'choices': np.array(choices)}
    if filter_valid:
        result['delta_hh_lls'] = np.array(choice_hh)
        result['delta_evs'] = np.array(choice_ev)
    return result


def compute_behavior_heatmap(choices, delta_hh_lls, delta_evs):
    """Compute 7×7 P(Right Choice) heatmap from choices and deltas."""
    hh_ll_bins = np.linspace(-2.2, 2.2, 8)
    ev_bins = np.linspace(-0.9, 0.9, 8)
    heatmap = np.full((7, 7), np.nan)

    for i in range(7):
        for j in range(7):
            mask = ((delta_hh_lls >= hh_ll_bins[i]) & (delta_hh_lls < hh_ll_bins[i+1]) &
                    (delta_evs >= ev_bins[j]) & (delta_evs < ev_bins[j+1]))
            if np.sum(mask) > 0:
                heatmap[j, i] = np.mean(choices[mask] == 2)

    return heatmap, hh_ll_bins, ev_bins


def regress_neurons(neural_data, delta_hh_lls, delta_evs, time_slice=slice(25, 50)):
    """
    Perform multiple regression for each neuron.

    Returns (beta_hh_ll, beta_ev) arrays of shape (n_neurons,).
    """
    T, n_trials, n_neurons = neural_data.shape
    beta_hh_ll, beta_ev = [], []

    for n in range(n_neurons):
        activity = np.mean(neural_data[time_slice, :, n], axis=0)
        X = np.column_stack([delta_hh_lls, delta_evs, np.ones(len(activity))])
        coeffs, _, _, _ = np.linalg.lstsq(X, activity, rcond=None)
        beta_hh_ll.append(coeffs[0])
        beta_ev.append(coeffs[1])

    return np.array(beta_hh_ll), np.array(beta_ev)


def compute_value_grid(trials, action_indices, Z_b_np):
    """Compute 5x5 predicted value grid from chosen options."""
    value_grid = np.zeros((5, 5))
    count_grid = np.zeros((5, 5))

    if Z_b_np.ndim == 3:
        Z_b_np = np.mean(Z_b_np, axis=-1)

    for i, trial in enumerate(trials):
        trial_actions = action_indices[:, i]
        choice_idx = np.where((trial_actions == 1) | (trial_actions == 2))[0]
        if len(choice_idx) > 0:
            choice = trial_actions[choice_idx[0]]
            pred_value = Z_b_np[choice_idx[0], i]
            target = trial['target_l'] if choice == 1 else trial['target_r']
            row, col = target // 5, target % 5
            value_grid[row, col] += pred_value
            count_grid[row, col] += 1

    return np.where(count_grid > 0, value_grid / count_grid, np.nan)


def _format_ev_label(ev):
    """Format EV labels from the actual trial values without image-specific scaling."""
    if abs(ev) >= 10:
        return f'{ev:g}'
    return f'{ev:.2f}'.rstrip('0').rstrip('.')


def plot_gambling_reward_probability_design(figspath):
    """
    Plot the gambling task option design as reward size vs probability.

    The task stores options as probability-major rows in value_vector:
    five reward/EV levels for 10%, then five for 30%, etc. This plot
    reconstructs the intended design directly from value_vector and colors
    each point with the matching color_vector entry.
    """
    from tasks.gambling import value_vector, color_vector

    os.makedirs(figspath, exist_ok=True)

    probabilities = value_vector[:, 0] * 100
    reward_ul = value_vector[:, 1] * 100
    expected_values = value_vector[:, 0] * value_vector[:, 1] * 100

    prob_levels = np.unique(probabilities)
    ev_levels = np.array(sorted(np.unique(np.round(expected_values, 6))))

    fig, ax = plt.subplots(figsize=(3.8, 3.4))

    # Constant-EV curves: reward = EV / probability.
    x_smooth = np.linspace(prob_levels.min(), prob_levels.max(), 300)
    for ev in ev_levels:
        y_smooth = ev / (x_smooth / 100)
        ax.plot(x_smooth, y_smooth, color='0.75', linewidth=2.4, zorder=1)

    ax.scatter(
        probabilities,
        reward_ul,
        s=190,
        c=color_vector,
        edgecolors='none',
        zorder=3,
        clip_on=False,
    )

    ax.set_xlabel('Probability (%)', fontsize=18)
    ax.set_ylabel('Reward size (µL)', fontsize=18)
    ax.set_xlim(6, 94)
    ax.set_ylim(0, 2700)
    ax.set_xticks(prob_levels)
    ax.set_xticklabels([f'{int(p)}' for p in prob_levels], fontsize=15)
    ax.set_yticks([0, 500, 1000, 1500, 2000, 2500])
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(width=3, length=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.2)
    ax.spines['bottom'].set_linewidth(3.2)

    fig.tight_layout()

    outfile = os.path.join(figspath, 'gambling_reward_probability_design.png')
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved gambling reward-probability design plot: {outfile}")
    return outfile


def plot_gambling_expected_value_probability_design(figspath):
    """
    Plot the task option grid as expected value vs probability.

    This reproduces the paper-style 5x5 color layout:
    x-axis is HH→LL based on reward probability, y-axis is raw expected
    value in µL, and colors come from the task color_vector.
    """
    from tasks.gambling import value_vector, color_vector

    os.makedirs(figspath, exist_ok=True)

    probabilities = value_vector[:, 0] * 100
    prob_levels = np.unique(probabilities)
    ev_levels = np.array([100.0, 137.5, 175.0, 212.5, 250.0])
    expected_value_ul = np.tile(ev_levels, len(prob_levels))

    fig, ax = plt.subplots(figsize=(5.0, 4.8))
    fig.subplots_adjust(left=0.27, right=0.96, bottom=0.27, top=0.88)

    ax.scatter(
        probabilities,
        expected_value_ul,
        s=230,
        c=color_vector,
        edgecolors='none',
        zorder=3,
        clip_on=False,
    )

    ax.set_xlabel('HH-LL\nbased on Probability (%)', fontsize=18, labelpad=18)
    ax.set_ylabel('Expected value (µL)', fontsize=18, labelpad=18)
    ax.set_xlim(6, 96)
    ax.set_ylim(86, 266)
    ax.set_xticks(prob_levels)
    ax.set_xticklabels([f'{int(p)}' for p in prob_levels], fontsize=16)
    ax.set_yticks(ev_levels)
    ax.set_yticklabels([f'{ev:.1f}' for ev in ev_levels], fontsize=16)
    ax.text(-0.18, -0.07, 'HH', transform=ax.transAxes,
            fontsize=18, ha='left', va='top')
    ax.text(1.04, -0.07, 'LL', transform=ax.transAxes,
            fontsize=18, ha='left', va='top')

    ax.tick_params(width=3, length=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3.2)
    ax.spines['bottom'].set_linewidth(3.2)

    fig.text(0.08, 0.94, 'B', fontsize=34, weight='bold')

    outfile = os.path.join(figspath, 'gambling_expected_value_probability_design.png')
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved gambling expected-value probability design plot: {outfile}")
    return outfile

def compute_option_choice_frequency(trials, action_indices):
    """
    Compute relative preference within each EV group (probability weighting).

    For each EV level, computes: P(choose option with probability p | both options have same EV)
    This reveals how the agent weights different probabilities when EV is held constant.

    Returns a dict mapping EV level to {probabilities, choice_proportions, counts}
    where each entry has 5 points (one for each probability level at that EV).
    """
    from tasks.gambling import value_vector, REWARD_SCALE

    probabilities = value_vector[:, 0]
    magnitudes = value_vector[:, 1] / REWARD_SCALE
    evs = probabilities * magnitudes

    # For each option: count how many times it was chosen vs presented
    # Structure: [col][row] = (times_chosen, times_presented)
    option_chosen = {col: np.zeros(5) for col in range(5)}
    option_presented = {col: np.zeros(5) for col in range(5)}

    for i, trial in enumerate(trials):
        trial_actions = action_indices[:, i]
        choice_idx = np.where((trial_actions == 1) | (trial_actions == 2))[0]
        if len(choice_idx) == 0:
            continue

        choice = trial_actions[choice_idx[0]]
        chosen_target = trial['target_l'] if choice == 1 else trial['target_r']

        target_l = trial['target_l']
        target_r = trial['target_r']

        # Extract position in grid for both options
        col_l = target_l % 5
        row_l = target_l // 5
        col_r = target_r % 5
        row_r = target_r // 5

        # Count ALL trials (both within-EV and cross-EV)
        # Both options were presented
        option_presented[col_l][row_l] += 1
        option_presented[col_r][row_r] += 1

        # One option was chosen
        col_chosen = chosen_target % 5
        row_chosen = chosen_target // 5
        option_chosen[col_chosen][row_chosen] += 1

    # Compute choice rate: P(choose this option | it was presented)
    results = {}
    for col in range(5):
        ev_level = round(evs[col], 2)  # EV is same for all in column
        probs = []
        choice_rates = []
        counts_chosen = []
        counts_presented = []

        for row in range(5):
            option_idx = col + 5 * row
            probs.append(probabilities[option_idx] * 100)  # Convert to percentage

            # Choice rate = times chosen / times presented
            n_chosen = option_chosen[col][row]
            n_presented = option_presented[col][row]

            if n_presented > 0:
                choice_rates.append(n_chosen / n_presented)
            else:
                choice_rates.append(0.0)

            counts_chosen.append(n_chosen)
            counts_presented.append(n_presented)


        # Sort by probability
        order = np.argsort(probs)
        results[ev_level] = {
            'probabilities': np.array(probs)[order],
            'choice_proportions': np.array(choice_rates)[order],
            'counts': np.array(counts_chosen)[order]
        }

    return results

def generate_psychometric_trial_set(trials_per_comparison=20):
    """
    Generate trials designed for probability weighting curves.
    Creates ALL pairwise comparisons (both within-EV and cross-EV) to allow
    the agent to demonstrate both:
    1. EV preference (higher EV chosen more often)
    2. Probability weighting (within same EV, probability preferences)
    """
    trials = []

    # Generate ALL possible pairs of the 25 options
    for opt1 in range(25):
        for opt2 in range(opt1 + 1, 25):  # Avoid duplicates and self-pairs
            # Generate trials for this specific comparison
            for _ in range(trials_per_comparison):
                # Add both orderings (left-right and right-left)
                trials.append({'target_l': opt1, 'target_r': opt2})
                trials.append({'target_l': opt2, 'target_r': opt1})

    return trials

def plot_context_choice_probability_curves(contexts, trialsfiles, figspath, baseline_context=0.0):
    """
    Plot choice frequency vs probability for each of the 25 gambling options.
    This reveals probability weighting curves showing how often each option is chosen.
    """
    os.makedirs(figspath, exist_ok=True)

    # Compute choice frequencies for each context
    choice_data = {}
    for ctx in contexts:
        td = load_trial_data(trialsfiles[ctx])
        action_indices = convert_actions(td['A'])
        choice_data[ctx] = compute_option_choice_frequency(td['trials'], action_indices)

    if baseline_context not in choice_data:
        baseline_context = min(contexts, key=lambda c: abs(c))

    # Get all EV levels
    unique_evs = sorted(choice_data[baseline_context].keys())
    if len(unique_evs) == 0:
        raise SystemExit("No choice data found. Check trial files.")

    # Color maps
    grey_colors = plt.cm.Greys(np.linspace(0.85, 0.35, len(unique_evs)))
    red_colors = plt.cm.Reds(np.linspace(0.45, 0.85, len(unique_evs)))

    comparison_contexts = [ctx for ctx in contexts if ctx != baseline_context]

    for ctx in comparison_contexts:
        if ctx not in choice_data:
            continue

        fig, ax = plt.subplots(figsize=(10.8, 6.0))
        fig.subplots_adjust(left=0.08, right=0.67, bottom=0.12, top=0.88)

        baseline_handles = []
        context_handles = []

        # Plot each EV group (5 points per group)
        for ev_idx, ev_level in enumerate(unique_evs):
            if ev_level not in choice_data[baseline_context]:
                continue
            if ev_level not in choice_data[ctx]:
                continue

            # Get baseline data
            base_data = choice_data[baseline_context][ev_level]
            probs_base = base_data['probabilities']
            props_base = base_data['choice_proportions']

            # Get context data
            ctx_data = choice_data[ctx][ev_level]
            probs_ctx = ctx_data['probabilities']
            props_ctx = ctx_data['choice_proportions']

            # Connect dots with straight lines
            if len(probs_base) >= 2:
                from scipy.interpolate import interp1d
                # Use linear interpolation to connect dots
                f_base = interp1d(probs_base, props_base, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                f_ctx = interp1d(probs_ctx, props_ctx, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
                
                x_smooth = np.linspace(probs_base.min(), probs_base.max(), 100)

                # Clip to valid range [0, 1]
                y_base_smooth = np.clip(f_base(x_smooth), 0, 1)
                y_ctx_smooth = np.clip(f_ctx(x_smooth), 0, 1)

                # Plot smooth fitted curves
                ax.plot(x_smooth, y_base_smooth, '-', color=grey_colors[ev_idx],
                       linewidth=2.2, alpha=0.8, zorder=2)
                ax.plot(x_smooth, y_ctx_smooth, '-', color=red_colors[ev_idx],
                       linewidth=2.2, alpha=0.8, zorder=2)

            # Plot data points on top
            ax.plot(probs_base, props_base, 'o', color=grey_colors[ev_idx],
                   markerfacecolor='white', markeredgewidth=1.8, markersize=7,
                   zorder=4, clip_on=False)
            ax.plot(probs_ctx, props_ctx, 'o', color=red_colors[ev_idx],
                   markerfacecolor=red_colors[ev_idx], markeredgecolor=red_colors[ev_idx],
                   markersize=7, zorder=5, clip_on=False)

            # Create legend handles
            ev_label = f'EV {_format_ev_label(ev_level)}'
            baseline_handles.append(Line2D([0], [0], color=grey_colors[ev_idx],
                                          marker='o', markerfacecolor='white',
                                          markeredgewidth=1.8, linewidth=2.2,
                                          label=ev_label))
            context_handles.append(Line2D([0], [0], color=red_colors[ev_idx],
                                         marker='o', markerfacecolor=red_colors[ev_idx],
                                         markeredgecolor=red_colors[ev_idx],
                                         linewidth=2.2, label=ev_label))

        # Formatting
        title = f'Probability weighting curve: c={ctx:+.2f} vs c={baseline_context:+.2f}'
        ax.set_title(title, fontsize=15, pad=12)
        ax.set_xlabel('Probability (%)', fontsize=14)
        ax.set_ylabel('Proportion chosen', fontsize=14)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.set_yticks([0, 0.5, 1.0])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=12)

        handles = baseline_handles + context_handles
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, ncol=2, fontsize=8.5, frameon=False,
                 title=f'c={baseline_context:+.2f}                  c={ctx:+.2f}',
                 title_fontsize=10,
                 loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

        ctx_str = format_context_str(ctx)
        outfile = os.path.join(figspath, f'context_choice_probability_curves_ctx{ctx_str}.png')
        fig.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved probability weighting curve: {outfile}")


def plot_context_choice_probability_mega(contexts, trialsfiles, figspath, baseline_context=0.0):
    """
    Create a compact 2-row mega plot of probability weighting curves.
    """
    os.makedirs(figspath, exist_ok=True)

    choice_data = {}
    for ctx in contexts:
        td = load_trial_data(trialsfiles[ctx])
        action_indices = convert_actions(td['A'])
        choice_data[ctx] = compute_option_choice_frequency(td['trials'], action_indices)

    if baseline_context not in choice_data:
        baseline_context = min(contexts, key=lambda c: abs(c))

    unique_evs = sorted(choice_data[baseline_context].keys())
    if len(unique_evs) == 0:
        raise SystemExit("No choice data found. Check trial files.")

    negative_contexts = sorted([ctx for ctx in contexts if ctx < baseline_context])
    positive_contexts = sorted([ctx for ctx in contexts if ctx > baseline_context])
    n_cols = max(len(negative_contexts), len(positive_contexts))

    grey_colors = plt.cm.Greys(np.linspace(0.78, 0.35, len(unique_evs)))
    red_colors = plt.cm.Reds(np.linspace(0.45, 0.85, len(unique_evs)))

    fig, axes = plt.subplots(2, n_cols, figsize=(2.35 * n_cols, 5.2),
                             sharex=True, sharey=True)
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    def draw_panel(ax, ctx):
        if ctx not in choice_data:
            return

        for ev_idx, ev_level in enumerate(unique_evs):
            if ev_level not in choice_data[baseline_context]:
                continue
            if ev_level not in choice_data[ctx]:
                continue

            base_data = choice_data[baseline_context][ev_level]
            ctx_data = choice_data[ctx][ev_level]

            base_probs = base_data['probabilities']
            base_props = base_data['choice_proportions']
            ctx_probs = ctx_data['probabilities']
            ctx_props = ctx_data['choice_proportions']

            # Connect dots with straight lines
            if len(base_probs) >= 2:
                from scipy.interpolate import interp1d
                f_base = interp1d(base_probs, base_props, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
                f_ctx = interp1d(ctx_probs, ctx_props, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
                
                x_smooth = np.linspace(base_probs.min(), base_probs.max(), 100)

                y_base_smooth = np.clip(f_base(x_smooth), 0, 1)
                y_ctx_smooth = np.clip(f_ctx(x_smooth), 0, 1)

                # Plot smooth curves
                ax.plot(x_smooth, y_base_smooth, '-', color=grey_colors[ev_idx],
                        linewidth=1.1, alpha=0.55, zorder=1)
                ax.plot(x_smooth, y_ctx_smooth, '-', color=red_colors[ev_idx],
                        linewidth=1.5, alpha=0.9, zorder=2)

            # Plot data points
            ax.plot(base_probs, base_props, 'o', color=grey_colors[ev_idx],
                    markerfacecolor='white', markeredgewidth=0.9,
                    markersize=3.2, zorder=3, clip_on=False)
            ax.plot(ctx_probs, ctx_props, 'o', color=red_colors[ev_idx],
                    markerfacecolor=red_colors[ev_idx],
                    markeredgecolor=red_colors[ev_idx],
                    markersize=3.2, zorder=4, clip_on=False)

        ax.set_title(f'c={ctx:+.1f}', fontsize=10, pad=4)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([10, 50, 90])
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    for col, ctx in enumerate(negative_contexts):
        draw_panel(axes[0, col], ctx)
    for col, ctx in enumerate(positive_contexts):
        draw_panel(axes[1, col], ctx)

    for col in range(len(negative_contexts), n_cols):
        axes[0, col].axis('off')
    for col in range(len(positive_contexts), n_cols):
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Negative c\nProportion chosen', fontsize=10)
    axes[1, 0].set_ylabel('Positive c\nProportion chosen', fontsize=10)
    for ax in axes[1, :]:
        ax.set_xlabel('Probability (%)', fontsize=9)

    # Create legend handles for baseline (grey) and context (red) for each EV
    baseline_handles = [
        Line2D([0], [0], color=grey_colors[i], marker='o', markerfacecolor='white',
               markeredgewidth=1.2, linewidth=1.5, label=f'EV {_format_ev_label(ev)}')
        for i, ev in enumerate(unique_evs)
    ]
    context_handles = [
        Line2D([0], [0], color=red_colors[i], marker='o',
               markerfacecolor=red_colors[i], markeredgecolor=red_colors[i],
               linewidth=1.5, label=f'EV {_format_ev_label(ev)}')
        for i, ev in enumerate(unique_evs)
    ]

    # Combine handles and create custom labels with two columns
    all_handles = baseline_handles + context_handles

    # Place legend outside the plot area on the right, close to plots
    fig.legend(handles=all_handles, loc='center left',
               frameon=False, bbox_to_anchor=(0.92, 0.5),
               fontsize=9, ncol=2, columnspacing=1.5, handlelength=2.0,
               title=f'c={baseline_context:+.1f}                  c=dopamine',
               title_fontsize=10)

    fig.suptitle('Probability Weighting Curves Across Context', fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    outfile = os.path.join(figspath, 'context_choice_probability_curves_mega.png')
    fig.savefig(outfile, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved probability weighting mega plot: {outfile}")


def load_model_weights(modelfile):
    """Load output weights from a model file. Returns (Wout_policy, Wout_value)."""
    model_data = utils.load(modelfile)
    policy_params = model_data.get('best_policy_params', {})
    baseline_params = model_data.get('best_baseline_params', {})

    Wout_policy = to_numpy(policy_params.get('Wout', None))

    # For MLP critic, use Wout2 (final layer) if available, otherwise fall back to Wout
    if 'Wout2' in baseline_params:
        Wout_value = to_numpy(baseline_params.get('Wout2', None))
    else:
        Wout_value = to_numpy(baseline_params.get('Wout', None))

    return Wout_policy, Wout_value


def model_uses_dopamine_split(modelfile):
    """Return True when first/second policy halves have dopamine/opponent meaning."""
    try:
        model_data = utils.load(modelfile)
        cfg = model_data.get('config', {})
    except Exception:
        return False

    return bool(
        cfg.get('use_rpe_modulation', False) or
        cfg.get('use_opponent_modulation', False) or
        cfg.get('vta_training_context', False) or
        cfg.get('dopamine_heterogeneous_sensitivity', False) or
        cfg.get('dopamine_sensitivity_learned', False) or
        cfg.get('dopamine_bias_enabled', False)
    )


def policy_split_labels(dopamine_split=False):
    """Labels for the first and second halves of the policy network."""
    if dopamine_split:
        return {
            'half1_short': 'D1',
            'half2_short': 'D2',
            'half1_row': 'Policy D1\nGo Pull',
            'half2_row': 'Policy D2\nNoGo Pull',
            'half1_beta': 'Policy D1\nβEV',
            'half2_beta': 'Policy D2\nβEV',
            'half1_weight': 'Policy D1\nOutput Weight',
            'half2_weight': 'Policy D2\nOutput Weight',
            'half1_title': 'D1 (Go) Pull',
            'half2_title': 'D2 (NoGo) Pull',
            'total_title': 'Total Subjective Value ($V=P-N$)',
            'total_row': 'Policy Total\nV = P - N',
        }

    return {
        'half1_short': '1st half',
        'half2_short': '2nd half',
        'half1_row': 'Policy units\n1st half',
        'half2_row': 'Policy units\n2nd half',
        'half1_beta': 'Policy 1st half\nβEV',
        'half2_beta': 'Policy 2nd half\nβEV',
        'half1_weight': 'Policy 1st half\nOutput Weight',
        'half2_weight': 'Policy 2nd half\nOutput Weight',
        'half1_title': 'Policy units 1st half',
        'half2_title': 'Policy units 2nd half',
        'total_title': 'Total Policy Logits',
        'total_row': 'Policy Total\nLogits',
    }


def format_kappa_str(kappa):
    """Format kappa value to a filename-safe string."""
    return f"{kappa:+.1f}".replace('.', 'p').replace('-', 'neg').replace('+', 'pos')


def format_context_str(ctx):
    """Format context value to a filename-safe string, normalizing -0.0 to +0.0."""
    if abs(ctx) < 1e-12:
        ctx = 0.0
    return f"{ctx:+.2f}".replace('.', 'p').replace('-', 'neg').replace('+', 'pos')


def dense_context_values():
    """Dense context sweep from -1.0 to +1.0 in 0.1 steps."""
    return [0.0 if i == 0 else round(i / 10, 1) for i in range(-10, 11)]


def context_values_step(step=0.2):
    """Context sweep from -1.0 to +1.0 with a configurable step."""
    n_steps = int(round(2.0 / step))
    return [
        0.0 if abs(-1.0 + i * step) < 1e-9 else round(-1.0 + i * step, 2)
        for i in range(n_steps + 1)
    ]


def compute_theoretical_evs():
    """
    Compute theoretical expected values for all 25 gambling options.

    Returns ev_grid : ndarray (5, 5) with rows=probability, cols=magnitude.
    """
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    magnitudes = np.linspace(0.4, 2.0, 5)
    return np.outer(probs, magnitudes)


def create_extended_value_colormap():
    """Custom colormap with viridis reserved for the 0.39-1.01 value range."""
    viridis = plt.cm.viridis

    # LogNorm maps [vmin=0.01, vmax=10.0] to [0, 1].
    vmin_data = 0.01
    vmax_data = 10.0
    log_vmin = np.log(vmin_data)
    log_vmax = np.log(vmax_data)

    # Viridis is intentionally limited to the biologically relevant value
    # band around neutral expected values. Values below that use a distinct
    # magenta-to-blue extension so sub-neutral values remain distinguishable.
    vf = [0.0, 0.25, 0.5, 0.75, 1.0]
    viridis_anchors = [(0.39 + f * (1.01 - 0.39), viridis(f)) for f in vf]
    anchors = [
        (0.01, '#3b0f70'),
        (0.04, '#8c2981'),
        (0.10, '#de4968'),
        (0.20, '#fe9f6d'),
        (0.32, '#f6d746'),
        *viridis_anchors,
        (5.00, '#ff8c00'),
        (10.0, '#ff0000'),
    ]

    positions = [(np.log(p) - log_vmin) / (log_vmax - log_vmin) for p, _ in anchors]
    colors = [color for _, color in anchors]

    return LinearSegmentedColormap.from_list('value_extended', list(zip(positions, colors)))

TEAL_BROWN_CMAP = LinearSegmentedColormap.from_list(
    'teal_brown', ['#008080', '#40E0D0', '#FFD700', '#FF8C00', '#8B4513']
)


def _load_comparison_data(keys, trialsfiles, modelfiles):
    """
    Shared data loading for mega_comparison and distribution_comparison.

    Loads trials, computes actions, deltas, regression coefficients,
    choices, and value grids for each key.

    Returns (all_data dict, baseline_data or None).
    """
    all_data = {}
    baseline_key = 0.0 if 0.0 in keys else ('baseline' if 'baseline' in keys else None)

    for key in keys:
        if key not in trialsfiles:
            print(f"  ⚠️  {key}: Skipping (no trials file)")
            continue

        print(f"  📂 Loading {key}...")
        td = load_trial_data(trialsfiles[key])
        if td['format'] == 'behavior':
            print(f"  ⚠️  {key}: Behavior-only data, skipping")
            continue
        print(f"     ✓ Loaded {len(td['trials'])} trials")

        # Load model weights
        Wout_policy, Wout_value = None, None
        if key in modelfiles:
            Wout_policy, Wout_value = load_model_weights(modelfiles[key])
            if Wout_policy is not None:
                print(f"     Policy Wout shape: {Wout_policy.shape}, range: [{Wout_policy.min():.3f}, {Wout_policy.max():.3f}]")
            if Wout_value is not None:
                print(f"     Value Wout shape: {Wout_value.shape}, range: [{Wout_value.min():.3f}, {Wout_value.max():.3f}]")

        action_indices = convert_actions(td['A'])
        Z_b_np = to_numpy(td['Z_b'])
        r_policy_np = to_numpy(td['r_policy'])
        r_value_np = to_numpy(td['r_value'])

        delta_hh_lls, delta_evs, _ = compute_deltas(td['trials'])

        beta_hh_ll_policy, beta_ev_policy = regress_neurons(r_policy_np, delta_hh_lls, delta_evs)
        beta_hh_ll_value, beta_ev_value = regress_neurons(r_value_np, delta_hh_lls, delta_evs)

        half_N = r_policy_np.shape[2] // 2
        r_policy_d1 = r_policy_np[:, :, :half_N]
        r_policy_d2 = r_policy_np[:, :, half_N:]
        beta_hh_ll_policy_d1, beta_ev_policy_d1 = regress_neurons(
            r_policy_d1, delta_hh_lls, delta_evs
        )
        beta_hh_ll_policy_d2, beta_ev_policy_d2 = regress_neurons(
            r_policy_d2, delta_hh_lls, delta_evs
        )

        Wout_policy_d1 = None
        Wout_policy_d2 = None
        if Wout_policy is not None and Wout_policy.ndim >= 1:
            Wout_policy_d1 = Wout_policy[:half_N]
            Wout_policy_d2 = Wout_policy[half_N:]

        ch = extract_choices(td['trials'], action_indices, delta_hh_lls, delta_evs)

        value_grid = compute_value_grid(td['trials'], action_indices, Z_b_np)

        # --- NEW: Compute Policy Value Grids ---
        grid_V, grid_D1, grid_D2 = None, None, None
        pull_stats = None
        if 'Policy_Values' in td:
            grid_V, grid_D1, grid_D2 = compute_policy_value_grids(
                td['trials'], td['Policy_Values'], td['Policy_D1_Pull'], td['Policy_D2_Pull'], td['M']
            )
            d1_pull = to_numpy(td['Policy_D1_Pull'])
            d2_pull = to_numpy(td['Policy_D2_Pull'])
            mask = to_numpy(td['M']).astype(bool)
            if np.any(mask):
                d1_valid = d1_pull[mask]
                d2_valid = d2_pull[mask]
                pull_stats = {
                    'd1_abs_mean': float(np.mean(np.abs(d1_valid))),
                    'd2_abs_mean': float(np.mean(np.abs(d2_valid))),
                    'd1_std': float(np.std(d1_valid)),
                    'd2_std': float(np.std(d2_valid)),
                    'd1_choice_abs_mean': float(np.mean(np.abs(d1_valid[:, 1:3]))),
                    'd2_choice_abs_mean': float(np.mean(np.abs(d2_valid[:, 1:3]))),
                }
        # ---------------------------------------

        all_data[key] = {
            'choices': ch['choices'], 'delta_hh_lls': ch['delta_hh_lls'],
            'delta_evs': ch['delta_evs'], 'value_grid': value_grid,
            'beta_hh_ll_policy': beta_hh_ll_policy, 'beta_ev_policy': beta_ev_policy,
            'beta_hh_ll_value': beta_hh_ll_value, 'beta_ev_value': beta_ev_value,
            'beta_hh_ll_policy_d1': beta_hh_ll_policy_d1, 'beta_ev_policy_d1': beta_ev_policy_d1,
            'beta_hh_ll_policy_d2': beta_hh_ll_policy_d2, 'beta_ev_policy_d2': beta_ev_policy_d2,
            'Wout_policy': Wout_policy, 'Wout_value': Wout_value,
            'Wout_policy_d1': Wout_policy_d1, 'Wout_policy_d2': Wout_policy_d2,
            
            # --- NEW: Add them to the dictionary ---
            'grid_V': grid_V, 'grid_D1': grid_D1, 'grid_D2': grid_D2,
            'pull_stats': pull_stats
        }

    baseline_data = all_data.get(baseline_key)
    return all_data, baseline_data


def _compute_weight_limits(all_data, baseline_data, keys):
    """Compute symmetric weight limits for consistent axis scaling across columns."""
    all_pw, all_vw = [], []
    for key in keys:
        if key in all_data:
            wp = all_data[key].get('Wout_policy')
            wv = all_data[key].get('Wout_value')
            if wp is not None:
                all_pw.extend(wp.flatten())
            if wv is not None:
                all_vw.extend(wv.flatten())
    if baseline_data:
        if baseline_data.get('Wout_policy') is not None:
            all_pw.extend(baseline_data['Wout_policy'].flatten())
        if baseline_data.get('Wout_value') is not None:
            all_vw.extend(baseline_data['Wout_value'].flatten())

    policy_lim = max(abs(np.min(all_pw)), abs(np.max(all_pw))) * 1.1 if all_pw else 1.0
    value_lim = max(abs(np.min(all_vw)), abs(np.max(all_vw))) * 1.1 if all_vw else 1.0
    return policy_lim, value_lim


def _symmetric_axis_limit(values, min_limit=0.02, pad=1.25, percentile=99):
    """Robust symmetric axis limit for compact scatter rows."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return min_limit
    limit = np.nanpercentile(np.abs(values), percentile) * pad
    return max(float(limit), min_limit)


def _row_colorbar_axis(fig, ax, width=0.006):
    """Add a row colorbar axis next to the last panel."""
    pos = ax.get_position()
    return fig.add_axes([pos.x1 + 0.01, pos.y0, width, pos.height])


def _plot_row_behavior(fig, gs, row, col_keys, all_data, get_title):
    """Plot Row: Behavioral heatmaps."""
    axes = []
    im = None
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        axes.append(ax)
        if key not in all_data:
            ax.text(0.5, 0.5, f'No data\n{key}', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        d = all_data[key]
        heatmap, hh_ll_bins, ev_bins = compute_behavior_heatmap(
            d['choices'], d['delta_hh_lls'], d['delta_evs'])
        im = ax.imshow(heatmap, origin='lower', cmap='plasma', vmin=0, vmax=1,
                       extent=[hh_ll_bins[0], hh_ll_bins[-1], ev_bins[0], ev_bins[-1]],
                       aspect='auto', interpolation='nearest')
        ax.set_title(get_title(key), fontsize=11, weight='bold',
                     color='blue' if isinstance(key, (int, float)) and key < 0 else
                           ('red' if isinstance(key, (int, float)) and key > 0 else 'black'))
        ax.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-0.8, 0, 0.8])
        if idx == 0:
            ax.set_ylabel('ΔEV', fontsize=10); ax.set_xlabel('ΔHH-LL', fontsize=9)
    # Colorbar
    if im is not None:
        cax = _row_colorbar_axis(fig, axes[-1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('P(Right)', fontsize=9, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
    return axes


def _plot_row_values(fig, gs, row, col_keys, all_data):
    """Plot Row: Predicted value heatmaps (log-scaled)."""
    axes = []
    im = None
    value_cmap = create_extended_value_colormap()
    norm = LogNorm(vmin=0.01, vmax=10.0)
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        axes.append(ax)
        if key not in all_data:
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # Clip negative/zero values to vmin and apply log normalization
        data = np.clip(all_data[key]['value_grid'], 0.01, None)
        im = ax.imshow(data.T, cmap=value_cmap, norm=norm,
                       aspect='auto', origin='lower')
        prob_labels = ['10', '30', '50', '70', '90']
        ax.set_yticks(range(5)); ax.set_xticks(range(5))
        ax.set_yticklabels(prob_labels, fontsize=8)
        ax.set_xticklabels(prob_labels, fontsize=8)
        if idx == 0:
            ax.set_ylabel('EV', fontsize=10); ax.set_xlabel('HH-LL(%)', fontsize=9)
    if im is not None:
        cax = _row_colorbar_axis(fig, axes[-1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Predicted\nValue', fontsize=9, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
        # Ticks in data coordinates; LogNorm handles the scaling
        cbar.set_ticks([0.01, 0.39, 1.01, 5.0, 10.0])
        cbar.set_ticklabels(['0.01', '0.39', '1.01', '5.0', '10.0'])
    return axes

def _plot_row_regression(fig, gs, row, col_keys, all_data, network, ylabel):
    """Plot Row: Regression scatter (β_HH-LL vs β_EV)."""
    axes = []
    sc = None
    beta_key_hh = f'beta_hh_ll_{network}'
    beta_key_ev = f'beta_ev_{network}'
    if network == 'value':
        all_hh = [all_data[key][beta_key_hh] for key in col_keys if key in all_data]
        all_ev = [all_data[key][beta_key_ev] for key in col_keys if key in all_data]
        xlim = _symmetric_axis_limit(np.concatenate(all_hh) if all_hh else [], min_limit=0.03)
        ylim = _symmetric_axis_limit(np.concatenate(all_ev) if all_ev else [], min_limit=0.08)
    else:
        xlim = 0.5
        ylim = 0.8
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        axes.append(ax)
        if key not in all_data:
            continue
        beta_hh = all_data[key][beta_key_hh]
        beta_ev = all_data[key][beta_key_ev]
        sort_idx = np.argsort(beta_ev)
        neuron_colors = np.zeros(len(beta_ev))
        neuron_colors[sort_idx] = np.arange(len(beta_ev))
        sc = ax.scatter(beta_hh, beta_ev, c=neuron_colors, cmap=TEAL_BROWN_CMAP,
                        s=20, alpha=0.6, edgecolors='none', vmin=0, vmax=len(beta_ev)-1)
        ax.axhline(0, color='black', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.axvline(0, color='black', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.set_xlim([-xlim, xlim]); ax.set_ylim([-ylim, ylim])
        ax.set_xticks([-xlim, 0, xlim]); ax.set_yticks([-ylim, 0, ylim])
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=10); ax.set_xlabel('βHH-LL', fontsize=9)
    if sc is not None:
        cax = _row_colorbar_axis(fig, axes[-1])
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label('Neuron\n(sorted by βEV)', fontsize=9, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
        n = len(all_data[col_keys[-1] if col_keys[-1] in all_data else
                 next(k for k in reversed(col_keys) if k in all_data)][beta_key_ev])
        mid = (n + 1) // 2
        cbar.set_ticks([0, (n - 1) / 2, n - 1])
        cbar.set_ticklabels(['1', str(mid), str(n)])
    return axes


def _plot_row_regression_keys(fig, gs, row, col_keys, all_data, beta_key_hh, beta_key_ev, ylabel):
    """Plot Row: Regression scatter using explicit beta keys."""
    axes = []
    sc = None
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        axes.append(ax)
        if key not in all_data:
            continue
        beta_hh = all_data[key][beta_key_hh]
        beta_ev = all_data[key][beta_key_ev]
        sort_idx = np.argsort(beta_ev)
        neuron_colors = np.zeros(len(beta_ev))
        neuron_colors[sort_idx] = np.arange(len(beta_ev))
        sc = ax.scatter(beta_hh, beta_ev, c=neuron_colors, cmap=TEAL_BROWN_CMAP,
                        s=20, alpha=0.6, edgecolors='none', vmin=0, vmax=len(beta_ev)-1)
        ax.axhline(0, color='black', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.axvline(0, color='black', linestyle=':', alpha=0.4, linewidth=0.8)
        ax.set_xlim([-0.5, 0.5]); ax.set_ylim([-0.8, 0.8])
        ax.set_xticks([-0.4, 0, 0.4]); ax.set_yticks([-0.6, 0, 0.6])
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=10); ax.set_xlabel('βHH-LL', fontsize=9)
    if sc is not None:
        cax = _row_colorbar_axis(fig, axes[-1])
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label('Neuron\n(sorted by βEV)', fontsize=9, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
        n = len(all_data[col_keys[-1] if col_keys[-1] in all_data else
                 next(k for k in reversed(col_keys) if k in all_data)][beta_key_ev])
        mid = (n + 1) // 2
        cbar.set_ticks([0, (n - 1) / 2, n - 1])
        cbar.set_ticklabels(['1', str(mid), str(n)])
    return axes


def _plot_row_weights(fig, gs, row, col_keys, all_data, baseline_data,
                      network, color, lim, ylabel, xlabel):
    """Plot Row: Output weight comparison vs baseline."""
    wkey = f'Wout_{network}'
    baseline_w = baseline_data.get(wkey) if baseline_data else None
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        if key not in all_data:
            continue
        w = all_data[key].get(wkey)
        if baseline_w is not None and w is not None:
            # Check if shapes match (linear vs MLP baseline)
            if baseline_w.size != w.size:
                ax.text(0.5, 0.5, 'Shape\nMismatch', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim])
                continue
            ax.scatter(baseline_w.flatten(), w.flatten(), s=15, alpha=0.5, color=color)
            ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=1)
            ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_aspect('equal')
            ticks = [-lim, 0, lim]
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.tick_params(labelsize=8)
            fmt = lambda t: f'{t:.3f}' if abs(t) < 0.1 else f'{t:.2f}' if abs(t) < 1 else f'{t:.1f}'
            ax.set_xticklabels([fmt(t) for t in ticks])
            ax.set_yticklabels([fmt(t) for t in ticks])
            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=10); ax.set_xlabel(xlabel, fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No weights', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])


def _plot_row_beta_vs_weights(fig, gs, row, col_keys, all_data, network, color, wlim):
    """Plot Row: β_HH-LL vs output weights."""
    wkey = f'Wout_{network}'
    beta_key = f'beta_hh_ll_{network}'
    if network == 'value':
        all_betas = []
        all_weights = []
        for key in col_keys:
            if key not in all_data:
                continue
            w = all_data[key].get(wkey)
            if w is None:
                continue
            n_outputs = w.shape[1] if w.ndim > 1 else 1
            betas = all_data[key][beta_key]
            all_betas.extend(np.repeat(betas, n_outputs) if n_outputs > 1 else betas)
            all_weights.extend(w.flatten())
        xlim = _symmetric_axis_limit(all_betas, min_limit=0.03)
        ylim = _symmetric_axis_limit(all_weights, min_limit=0.03)
    else:
        xlim = 0.5
        ylim = wlim
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        if key not in all_data:
            continue
        betas = all_data[key][beta_key]
        w = all_data[key].get(wkey)
        if w is None:
            continue
        n_outputs = w.shape[1] if w.ndim > 1 else 1
        x_data = np.repeat(betas, n_outputs) if n_outputs > 1 else betas
        y_data = w.flatten()
        ax.scatter(x_data, y_data, s=15, alpha=0.5, color=color)
        ax.set_xlim([-xlim, xlim]); ax.set_ylim([-ylim, ylim])
        ax.set_xticks([-xlim, 0, xlim])
        ax.set_yticks([-ylim, 0, ylim])
        ax.tick_params(labelsize=8)
        fmt = lambda t: f'{t:.3f}' if abs(t) < 0.1 else f'{t:.2f}' if abs(t) < 1 else f'{t:.1f}'
        ax.set_xticklabels([fmt(t) for t in [-xlim, 0, xlim]])
        ax.set_yticklabels([fmt(t) for t in [-ylim, 0, ylim]])
        if idx == 0:
            ax.set_ylabel(f'{network.capitalize()}\nOutput Weight', fontsize=10)
            ax.set_xlabel('βHH-LL', fontsize=9)


def _plot_row_beta_vs_weights_keys(fig, gs, row, col_keys, all_data, wkey, beta_key, color, wlim, ylabel):
    """Plot Row: β_HH-LL vs output weights using explicit keys."""
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        if key not in all_data:
            continue
        betas = all_data[key][beta_key]
        w = all_data[key].get(wkey)
        if w is None:
            continue
        n_outputs = w.shape[1] if w.ndim > 1 else 1
        x_data = np.repeat(betas, n_outputs) if n_outputs > 1 else betas
        y_data = w.flatten()
        ax.scatter(x_data, y_data, s=15, alpha=0.5, color=color)
        ax.set_xlim([-0.5, 0.5]); ax.set_ylim([-wlim, wlim])
        ax.set_xticks([-0.4, 0, 0.4])
        ax.set_yticks([-wlim, 0, wlim])
        ax.tick_params(labelsize=8)
        fmt = lambda t: f'{t:.3f}' if abs(t) < 0.1 else f'{t:.2f}' if abs(t) < 1 else f'{t:.1f}'
        ax.set_xticklabels([f'{t:.1f}' for t in [-0.4, 0, 0.4]])
        ax.set_yticklabels([fmt(t) for t in [-wlim, 0, wlim]])
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel('βHH-LL', fontsize=9)

# ============================================================
# Plotting Functions
# ============================================================


def plot_heatmap(trialsfile, figspath):
    """
    Generate behavioral heatmap showing proportion of rightward choices
    as a function of ΔHH-LL (log probability ratio) and ΔEV (log EV ratio).
    """
    td = load_trial_data(trialsfile)
    trials = td['trials']
    n_trials = len(trials)
    print(f"\nAnalyzing {n_trials} trials...")

    action_indices = convert_actions(td['A'])
    delta_hh_lls, delta_evs, valid = compute_deltas(trials)
    ch = extract_choices(trials, action_indices, delta_hh_lls, delta_evs, valid)
    choices = ch['choices']

    print(f"Valid choices: {len(choices)}/{n_trials}")
    print(f"Right choices: {np.sum(choices == 2)} ({100*np.mean(choices == 2):.1f}%)")
    print(f"Left choices: {np.sum(choices == 1)} ({100*np.mean(choices == 1):.1f}%)")

    heatmap, hh_ll_bins, ev_bins = compute_behavior_heatmap(
        choices, ch['delta_hh_lls'], ch['delta_evs'])

    # Plot
    x_range = hh_ll_bins[-1] - hh_ll_bins[0]
    y_range = ev_bins[-1] - ev_bins[0]
    aspect_ratio = y_range / x_range

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)

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
    ax_main.set_xlabel('Delta HH-LL (log probability difference)', fontsize=12)
    ax_main.set_ylabel('Delta EV (log expected value difference)', fontsize=12)
    ax_main.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax_main.axvline(0, color='white', linestyle='--', alpha=0.3, linewidth=1)

    # Colorbar
    ax_cbar = fig.add_subplot(gs[1])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('P(Right Choice)', fontsize=10)

    plt.tight_layout()

    # Save
    savefile = os.path.join(figspath, 'gambling_behavior.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to {savefile}")
    plt.close()


def plot_kappa_comparison(kappa_values, trialsfiles, figspath):
    """
    Create multi-panel comparison plot showing how different kappa values
    affect choice behavior in the gambling task.

    Parameters
    ----------
    kappa_values : list of float
        List of kappa values (e.g., [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    trialsfiles : dict
        Dictionary mapping kappa values to trial data files
    figspath : str
        Path to save figures
    """
    n_kappas = len(kappa_values)
    ncols = n_kappas
    fig, axes = plt.subplots(1, ncols, figsize=(3*ncols, 4))
    if n_kappas == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, kappa in enumerate(kappa_values):
        ax = axes[idx]

        if kappa not in trialsfiles:
            ax.text(0.5, 0.5, f'No data\nκ={kappa}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        try:
            td = load_trial_data(trialsfiles[kappa])
        except ValueError:
            ax.text(0.5, 0.5, f'Invalid data\nκ={kappa}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        action_indices = convert_actions(td['A'])
        delta_hh_lls, delta_evs, valid = compute_deltas(td['trials'])
        ch = extract_choices(td['trials'], action_indices, delta_hh_lls, delta_evs, valid)
        choices = ch['choices']

        heatmap, hh_ll_bins, ev_bins = compute_behavior_heatmap(
            choices, ch['delta_hh_lls'], ch['delta_evs'])

        x_range = hh_ll_bins[-1] - hh_ll_bins[0]
        y_range = ev_bins[-1] - ev_bins[0]
        aspect_ratio = y_range / x_range

        im = ax.imshow(
            heatmap, origin='lower', cmap='plasma', vmin=0, vmax=1,
            extent=[hh_ll_bins[0], hh_ll_bins[-1], ev_bins[0], ev_bins[-1]],
            aspect=1/aspect_ratio, interpolation='nearest'
        )

        if kappa < 0:
            title_color, risk_label = 'blue', 'Risk-Averse'
        elif kappa > 0:
            title_color, risk_label = 'red', 'Risk-Seeking'
        else:
            title_color, risk_label = 'black', 'Neutral'

        ax.set_title(f'k = {kappa:+.1f}\n{risk_label}', fontsize=11, color=title_color, weight='bold')

        if idx == 0:
            ax.set_ylabel("DeltaEV", fontsize=9)
        else:
            ax.set_yticks([])
        ax.set_xlabel("DeltaHH-LL", fontsize=9)

        ax.axhline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)

        p_right = np.mean(choices == 2)
        ax.text(0.02, 0.98, f'R: {p_right:.2f}',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    cbar = fig.colorbar(im, ax=axes[-1], orientation='vertical',
                       fraction=0.046, pad=0.04)
    cbar.set_label('P(Right Choice)', fontsize=9)

    plt.suptitle('Risk Sensitivity: Effect of k on Choice Behavior',
                 fontsize=14, weight='bold')
    plt.tight_layout()

    savefile = os.path.join(figspath, 'gambling_kappa_comparison.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"\nSaved kappa comparison to {savefile}")
    plt.close()


def plot_kappa_summary(kappa_values, trialsfiles, figspath):
    """
    Create summary plots showing aggregate statistics across kappa values.

    Shows:
    1. Mean reward vs kappa
    2. Choice rate vs kappa
    3. Risk preference metric vs kappa
    """
    kappas = []
    mean_rewards = []
    choice_rates = []
    risk_preferences = []

    for kappa in kappa_values:
        if kappa not in trialsfiles:
            continue

        try:
            td = load_trial_data(trialsfiles[kappa])
        except ValueError:
            continue

        trials = td['trials']
        action_indices = convert_actions(td['A'])
        R_np = to_numpy(td['R'])
        M_np = to_numpy(td['M'])

        kappas.append(kappa)

        # Mean reward
        total_reward = np.sum(R_np * M_np)
        n_trials = R_np.shape[1]
        mean_rewards.append(total_reward / n_trials)

        # Choice rate and risk preference
        n_choices = 0
        n_risky_choices = 0

        for i, trial in enumerate(trials):
            trial_actions = action_indices[:, i]
            choice_idx = np.where((trial_actions == 1) | (trial_actions == 2))[0]

            if len(choice_idx) > 0:
                n_choices += 1
                choice = trial_actions[choice_idx[0]]

                # Determine if risky (higher reward size)
                size_l = trial['size_l']
                size_r = trial['size_r']

                if choice == 1 and size_l > size_r:
                    n_risky_choices += 1
                elif choice == 2 and size_r > size_l:
                    n_risky_choices += 1

        choice_rates.append(n_choices / n_trials)
        risk_preferences.append(n_risky_choices / max(n_choices, 1))

    # Create summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(kappas, mean_rewards, 'o-', linewidth=2, markersize=8, color='steelblue')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Kappa (κ)', fontsize=12)
    axes[0].set_ylabel('Mean Reward per Trial', fontsize=12)
    axes[0].set_title('Performance vs Risk Sensitivity', fontsize=13, weight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(kappas, choice_rates, 'o-', linewidth=2, markersize=8, color='forestgreen')
    axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Kappa (κ)', fontsize=12)
    axes[1].set_ylabel('Proportion Making Choice', fontsize=12)
    axes[1].set_title('Choice Rate vs Risk Sensitivity', fontsize=13, weight='bold')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(kappas, risk_preferences, 'o-', linewidth=2, markersize=8, color='crimson')
    axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    axes[2].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Kappa (κ)', fontsize=12)
    axes[2].set_ylabel('P(Choose Risky | Choose)', fontsize=12)
    axes[2].set_title('Risk Preference vs κ', fontsize=13, weight='bold')
    axes[2].set_ylim([0, 1.05])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    for ax in axes:
        ax.axvspan(min(kappas), 0, alpha=0.1, color='blue', label='Risk-Averse')
        ax.axvspan(0, max(kappas), alpha=0.1, color='red', label='Risk-Seeking')

    plt.tight_layout()

    savefile = os.path.join(figspath, 'gambling_kappa_summary.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"Saved kappa summary to {savefile}")
    plt.close()


def plot_predicted_values(trialsfile, figspath, kappa=None):
    """
    Plot predicted values for all 25 gambling options as a 5x5 heatmap.

    Parameters
    ----------
    trialsfile : str
        Path to trial data file with neural activity
    figspath : str
        Directory to save figure
    kappa : float, optional
        Kappa value for title (if applicable)
    """
    try:
        td = load_trial_data(trialsfile)
        use_trial_data = (td['format'] in ('activity', 'rpe'))
    except Exception:
        use_trial_data = False

    if use_trial_data:
        action_indices = convert_actions(td['A'])
        Z_b_np = to_numpy(td['Z_b'])

        value_grid = compute_value_grid(td['trials'], action_indices, Z_b_np)

        print(f"\nPredicted value statistics (from trial data):")
        valid_values = value_grid[~np.isnan(value_grid)]
        if len(valid_values) > 0:
            print(f"  Min: {np.nanmin(value_grid):.3f}")
            print(f"  Max: {np.nanmax(value_grid):.3f}")
            print(f"  Mean: {np.nanmean(value_grid):.3f}")
            print(f"  Coverage: {np.sum(~np.isnan(value_grid))}/25 options")

        # Fill missing values with theoretical EV
        theoretical_ev = compute_theoretical_evs()
        value_grid = np.where(np.isnan(value_grid), theoretical_ev, value_grid)
    else:
        print("\nUsing theoretical expected values (prob × reward)...")
        value_grid = compute_theoretical_evs()

    print(f"\nFinal value statistics:")
    print(f"  Min: {value_grid.min():.3f}")
    print(f"  Max: {value_grid.max():.3f}")
    print(f"  Mean: {value_grid.mean():.3f}")
    print(f"  Expected EV range: 0.40 to 1.00")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(value_grid, cmap='viridis', aspect='auto', origin='lower',
                   vmin=0.4, vmax=1.0)

    ax.set_ylabel('EV', fontsize=14, weight='bold')
    ax.set_xlabel('HH-LL(%)', fontsize=14, weight='bold')

    if kappa is not None:
        ax.set_title(f'κ = {kappa:.1f}', fontsize=16, weight='bold')
    else:
        ax.set_title('κ = 0', fontsize=16, weight='bold')

    prob_labels = ['10', '30', '50', '70', '90']
    ax.set_yticks(range(5))
    ax.set_yticklabels(prob_labels, fontsize=11)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['', '', '', '', ''], fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Predicted Value', fontsize=11, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    if kappa is not None:
        suffix = f"_kappa{kappa:.1f}".replace('.', 'p').replace('-', 'neg')
        savefile = os.path.join(figspath, f'predicted_values{suffix}.png')
    else:
        savefile = os.path.join(figspath, 'predicted_values.png')

    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"Saved predicted values to {savefile}")
    plt.close()

    return value_grid


def _trial_decision_indices(trial, n_timepoints):
    """Return saved timestep indices for a trial's decision epoch."""
    epochs = trial.get('epochs', {})
    decision = epochs.get('decision', range(n_timepoints))
    indices = np.asarray(list(decision), dtype=int)
    return indices[(indices >= 0) & (indices < n_timepoints)]


def _choice_value_records(trialsfile):
    """
    Return per-trial chosen-option critic values.

    Each record is (trial_index, probability, objective_ev, ev_column, critic_value).
    The critic value is Z_b at the last valid decision-epoch timestep.
    """
    td = load_trial_data(trialsfile)
    if td['format'] == 'behavior':
        return np.empty((0, 5))

    action_indices = convert_actions(td['A'])
    z_b = to_numpy(td['Z_b'])
    if z_b.ndim == 3:
        z_b = np.squeeze(z_b, axis=-1) if z_b.shape[-1] == 1 else np.mean(z_b, axis=-1)

    records = []
    for trial_idx, trial in enumerate(td['trials']):
        decision_idx = _trial_decision_indices(trial, z_b.shape[0])
        if len(decision_idx) == 0:
            continue

        trial_actions = action_indices[decision_idx, trial_idx]
        choice_positions = np.where((trial_actions == 1) | (trial_actions == 2))[0]
        choice_times = decision_idx[choice_positions]
        if len(choice_times) == 0:
            continue

        choice_t = choice_times[0]
        choice = action_indices[choice_t, trial_idx]
        target = trial['target_l'] if choice == 1 else trial['target_r']
        prob = trial['prob_l'] if choice == 1 else trial['prob_r']
        size = trial['size_l'] if choice == 1 else trial['size_r']
        objective_ev = prob * size
        ev_col = target % 5

        decision_values = z_b[decision_idx, trial_idx]
        if 'M' in td:
            mask = to_numpy(td['M'])[decision_idx, trial_idx].astype(bool)
            decision_values = decision_values[mask]
        if len(decision_values) == 0:
            continue

        records.append((trial_idx, prob, objective_ev, ev_col, decision_values[-1]))

    return np.asarray(records, dtype=float)


def _selected_kappas_for_tuning(kappa_values):
    """Use the full available kappa sweep."""
    return sorted(kappa_values)


def plot_kappa_value_tuning(kappa_values, trialsfiles, figspath, window=75):
    """
    Plot critic value tuning across trials and the V/P relationship.

    The first panel shows the final decision critic value for every saved
    trial. The second panel averages critic value by reward probability within
    each objective-EV column, then averages those five probability curves.
    """
    selected = _selected_kappas_for_tuning(kappa_values)
    fallback_colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, max(len(selected), 1)))
    colors = {kappa: fallback_colors[i] for i, kappa in enumerate(selected)}

    records_by_kappa = {}
    for kappa in selected:
        if kappa not in trialsfiles:
            continue
        records = _choice_value_records(trialsfiles[kappa])
        if len(records) == 0:
            print(f"  No choice-value records for κ={kappa:+.1f}; skipping")
            continue
        records_by_kappa[kappa] = records

    if not records_by_kappa:
        print("No kappa value-tuning data available.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for color_i, kappa in enumerate(selected):
        if kappa not in records_by_kappa:
            continue
        records = records_by_kappa[kappa]
        order = np.argsort(records[:, 0])
        x = records[order, 0][:20]
        y = records[order, 4][:20]
        color = colors.get(kappa, fallback_colors[color_i])
        ax.plot(x, y, lw=0.9, color=color, alpha=0.85, label=f'κ={kappa:+.1f}')

    ax.set_xlabel('Trials', fontsize=18)
    ax.set_ylabel('V', fontsize=18, fontstyle='italic')
    ax.set_title('Final Decision Critic Value Across Trials', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=13)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    plt.tight_layout()

    savefile = os.path.join(figspath, 'kappa_value_across_trials.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"Saved kappa value tuning over trials to {savefile}")
    plt.close()

    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for color_i, kappa in enumerate(selected):
        if kappa not in records_by_kappa:
            continue
        records = records_by_kappa[kappa]
        value_by_prob_ev = np.full((5, 5), np.nan)
        for prob_i, prob in enumerate(probs):
            for ev_col in range(5):
                mask = (np.isclose(records[:, 1], prob)) & (records[:, 3] == ev_col)
                if np.any(mask):
                    value_by_prob_ev[prob_i, ev_col] = np.nanmean(records[mask, 4])

        curve = np.nanmean(value_by_prob_ev, axis=1)
        color = colors.get(kappa, fallback_colors[color_i])
        ax.plot(probs, curve, lw=2.2, color=color, label=f'κ={kappa:+.1f}')
        if np.isfinite(curve).any():
            mid_idx = np.nanargmin(np.abs(probs - 0.5))
            ax.axhline(curve[mid_idx], color=color, lw=1.0,
                       ls=(0, (4, 7)), alpha=0.55)

    ax.axvline(0.5, color='0.7', lw=1.0, ls=(0, (5, 6)))
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel('P(reward)', fontsize=20)
    ax.set_ylabel('V', fontsize=20, fontstyle='italic')
    ax.set_title('Critic V/P Relationship', fontsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()

    savefile = os.path.join(figspath, 'kappa_value_probability_relationship.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"Saved kappa V/P relationship to {savefile}")
    plt.close()


def plot_regression_analysis(trialsfile, figspath):
    """
    Perform multiple regression analysis and plot βHH-LL vs βEV scatter plots.
    """
    td = load_trial_data(trialsfile)

    if td['format'] == 'behavior':
        print("Error: This analysis requires activity data. Use 'trials-a' action.")
        return

    r_policy_np = to_numpy(td['r_policy'])
    r_value_np = to_numpy(td['r_value'])

    delta_hh_lls, delta_evs, _ = compute_deltas(td['trials'])

    beta_hh_ll_policy, beta_ev_policy = regress_neurons(r_policy_np, delta_hh_lls, delta_evs)
    beta_hh_ll_value, beta_ev_value = regress_neurons(r_value_np, delta_hh_lls, delta_evs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, bh, be, title in [
        (axes[0], beta_hh_ll_policy, beta_ev_policy, 'Policy Network Regression Coefficients'),
        (axes[1], beta_hh_ll_value, beta_ev_value, 'Value Network Regression Coefficients'),
    ]:
        scatter = ax.scatter(bh, be, c=np.arange(len(bh)), cmap='viridis', s=30, alpha=0.6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('β_HH-LL', fontsize=12)
        ax.set_ylabel('β_EV', fontsize=12)
        ax.set_title(title, fontsize=13, weight='bold')
        ax.grid(True, alpha=0.2)
        plt.colorbar(scatter, ax=ax, label='Neuron Index')

    plt.tight_layout()

    savefile = os.path.join(figspath, 'regression_analysis.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"Saved regression analysis to {savefile}")
    plt.close()

    return {
        'policy': {'beta_hh_ll': beta_hh_ll_policy, 'beta_ev': beta_ev_policy},
        'value': {'beta_hh_ll': beta_hh_ll_value, 'beta_ev': beta_ev_value}
    }


def compute_policy_value_grids(trials, Policy_Values, Policy_D1_Pull, Policy_D2_Pull, M):
    """
    Maps the Left (index 1) and Right (index 2) logits back to the 
    specific 25 gambling options presented in the trials.
    Extracts the values exactly at the timestep the decision was made.
    
    AXES SWAPPED: 
    Y-axis (Rows) = Expected Value (EV)
    X-axis (Cols) = Probability
    """
    grid_V = np.zeros((5, 5))
    grid_D1 = np.zeros((5, 5))
    grid_D2 = np.zeros((5, 5))
    counts = np.zeros((5, 5))

    Policy_Values_np = to_numpy(Policy_Values)
    Policy_D1_Pull_np = to_numpy(Policy_D1_Pull)
    Policy_D2_Pull_np = to_numpy(Policy_D2_Pull)
    M_np = to_numpy(M)

    for i, trial in enumerate(trials):
        target_l = trial['target_l']
        target_r = trial['target_r']

        t_choice = int(np.sum(M_np[:, i])) - 1
        if t_choice < 0:
            continue

        # --- SWAPPED AXES ASSIGNMENT ---
        # target // 5 gives Probability (0 to 4) -> Column (X)
        # target % 5 gives EV Magnitude (0 to 4) -> Row (Y)
        row_l, col_l = target_l % 5, target_l // 5
        row_r, col_r = target_r % 5, target_r // 5

        grid_V[row_l, col_l] += Policy_Values_np[t_choice, i, 1]
        grid_D1[row_l, col_l] += Policy_D1_Pull_np[t_choice, i, 1]
        grid_D2[row_l, col_l] += Policy_D2_Pull_np[t_choice, i, 1]
        counts[row_l, col_l] += 1

        grid_V[row_r, col_r] += Policy_Values_np[t_choice, i, 2]
        grid_D1[row_r, col_r] += Policy_D1_Pull_np[t_choice, i, 2]
        grid_D2[row_r, col_r] += Policy_D2_Pull_np[t_choice, i, 2]
        counts[row_r, col_r] += 1

    with np.errstate(invalid='ignore'):
        grid_V = np.divide(grid_V, counts, out=np.full_like(grid_V, np.nan), where=counts!=0)
        grid_D1 = np.divide(grid_D1, counts, out=np.full_like(grid_D1, np.nan), where=counts!=0)
        grid_D2 = np.divide(grid_D2, counts, out=np.full_like(grid_D2, np.nan), where=counts!=0)

    return grid_V, grid_D1, grid_D2


def plot_policy_subjective_values(trialsfile, figspath, context_val=None, dopamine_split=False):
    """
    Plot first-half pull, second-half pull, and total policy logits for the
    25 gambling options.
    """
    td = load_trial_data(trialsfile)
    
    if 'Policy_Values' not in td:
        return

    grid_V, grid_D1, grid_D2 = compute_policy_value_grids(
        td['trials'], td['Policy_Values'], td['Policy_D1_Pull'], td['Policy_D2_Pull'], td['M']
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = policy_split_labels(dopamine_split)
    titles = [labels['half1_title'], labels['half2_title'], labels['total_title']]
    grids = [grid_D1, grid_D2, grid_V]
    
    max_val = np.nanmax(np.abs([grid_D1, grid_D2, grid_V]))
    
    for i, ax in enumerate(axes):
        im = ax.imshow(grids[i], cmap='RdBu', origin='lower', vmin=-max_val, vmax=max_val, aspect='auto')
        
        ax.set_title(titles[i], fontsize=14, weight='bold')
        
        # --- SWAPPED LABELS ---
        ax.set_ylabel('Expected Value (EV)', fontsize=12)
        ax.set_xlabel('Reward Probability', fontsize=12)
        
        ax.set_yticks(range(5))
        ax.set_yticklabels(['0.4', '', '0.7', '', '1.0'])
        ax.set_xticks(range(5))
        ax.set_xticklabels(['10%', '30%', '50%', '70%', '90%'])
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Logits (Preference)', rotation=270, labelpad=15)

    title_suffix = f" (Context = {context_val:+.2f})" if context_val is not None else ""
    plt.suptitle(f"Policy Network Internal Value Representation{title_suffix}", fontsize=18, y=1.05)
    plt.tight_layout()

    ctx_str = f"_ctx{context_val:+.2f}".replace('.', 'p').replace('-', 'neg').replace('+', 'pos') if context_val is not None else ""
    savefile = os.path.join(figspath, f'policy_subjective_values{ctx_str}.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.close()

def plot_mega_policy_subjective_values(contexts, trialsfiles, figspath, dopamine_split=False):
    """
    Creates a 3x9 mega-plot showing the internal value representation across ALL contexts.
    Rows: first-half pull, second-half pull, total policy logits.
    Columns: Dopamine Contexts (from Risk Averse to Risk Seeking)
    Uses a globally locked color scale so intensities are directly comparable.
    """
    print(f"\nGenerating 3x{len(contexts)} Policy Subjective Values Mega-Plot...")
    
    all_grids = {}
    global_max = 0

    for ctx in contexts:
        if ctx not in trialsfiles:
            continue
        td = load_trial_data(trialsfiles[ctx])
        if 'Policy_Values' not in td:
            continue

        grid_V, grid_D1, grid_D2 = compute_policy_value_grids(
            td['trials'], td['Policy_Values'], td['Policy_D1_Pull'], td['Policy_D2_Pull'], td['M']
        )
        all_grids[ctx] = (grid_D1, grid_D2, grid_V)

        local_max = np.nanmax(np.abs([grid_D1, grid_D2, grid_V]))
        if local_max > global_max:
            global_max = local_max

    if not all_grids:
        print("Error: No valid policy value data found for mega-plot.")
        return

    # Increased hspace to 0.35 so the X-labels don't hit the titles below them
    fig = plt.figure(figsize=(3 * len(contexts) + 2, 14)) 
    gs = fig.add_gridspec(3, len(contexts), hspace=0.35, wspace=0.1)
    
    labels = policy_split_labels(dopamine_split)
    row_labels = [labels['half1_title'], labels['half2_title'], labels['total_title']]
    sorted_contexts = sorted(all_grids.keys())

    for row in range(3):
        for col, ctx in enumerate(sorted_contexts):
            grids = all_grids[ctx]
            grid_to_plot = grids[row] 
            
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(grid_to_plot, cmap='RdBu', origin='lower',
                           vmin=-global_max, vmax=global_max, aspect='auto')

            if row == 0:
                ax.set_title(f'Ctx = {ctx:+.2f}', fontsize=16, weight='bold', pad=15)

            if col == 0:
                ax.set_ylabel(f'{row_labels[row]}\n\nExpected Value (EV)', fontsize=14, weight='bold')
                ax.set_yticks(range(5))
                ax.set_yticklabels(['0.4', '', '0.7', '', '1.0'], fontsize=12)
            else:
                ax.set_yticks([])

            # --- X-axis Labels (NOW ON ALL ROWS) ---
            ax.set_xticks(range(5))
            ax.set_xticklabels(['10%', '30%', '50%', '70%', '90%'], fontsize=12, rotation=45)
            
            # Put the actual "Reward Probability" text on every column to be safe
            if col == 0:
                ax.set_xlabel('Reward Probability', fontsize=14, weight='bold')

    # Giant colorbar on the far right
    cbar_ax = fig.add_axes([0.91, 0.15, 0.005, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Logits (Preference)', rotation=270, labelpad=25, fontsize=16, weight='bold')
    cbar.ax.tick_params(labelsize=12)

    plt.suptitle("Policy Network Internal Value Representation Across Dopamine Contexts",
                 fontsize=24, weight='bold', y=0.96)

    plt.subplots_adjust(left=0.06, right=0.89, top=0.88, bottom=0.08)

    savefile = os.path.join(figspath, 'mega_policy_subjective_values_3x9.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"Saved 3x9 Mega-Plot to {savefile}")
    plt.close()

def do(action, args, config):
    """
    Manage analysis tasks.
    """
    print("ACTION*:   " + str(action))
    print("ARGS*:     " + str(args))

    if 'trials' in action:
        # Generate trial data
        try:
            trials_per_condition = int(args[0])
        except:
            trials_per_condition = 10  # 2 trials per condition = 625*2 = 1250 trials

        model = config['model']
        pg = model.get_pg(config['savefile'], config['seed'], config['dt'])

        # CRITICAL FIX: Reset RNG to a different seed for testing to prevent sequence memorization
        # The loaded model has the training RNG state, so trials would be identical to training
        pg.rng = np.random.RandomState(seed=999)  # Use RandomState, not Generator
        print("⚠️  Reset RNG to test seed (999) - different from training to prevent sequence memorization")

        # Generate all 625 conditions (25 left targets × 25 right targets)
        n_conditions = 25 * 25
        n_trials = trials_per_condition * n_conditions

        print(f"{n_trials} trials ({trials_per_condition} per condition)")

        task = model.Task()
        trials = []

        # Create systematic list of all conditions (like JAX code)
        # Each condition appears trials_per_condition times
        conditions = []
        for _ in range(trials_per_condition):
            for target_l in range(25):
                for target_r in range(25):
                    conditions.append((target_l, target_r))

        # Shuffle conditions (using the already-reset test RNG)
        pg.rng.shuffle(conditions)

        for target_l, target_r in conditions:
            context = {'target_l': target_l, 'target_r': target_r}
            trials.append(task.get_condition(pg.rng, pg.dt, context))

        runtools.run(action, trials, pg, config['trialspath'], dt_save=config['dt-save'])

    elif action == 'behavior':
        # Plot behavioral heatmap
        trialsfile = runtools.behaviorfile(config['trialspath'])
        if not os.path.exists(trialsfile):
            trialsfile = runtools.activityfile(config['trialspath'])
            if not os.path.exists(trialsfile):
                raise SystemExit(
                    "Missing trial data. Run trials-b or trials-a before plotting behavior."
                )
        plot_heatmap(trialsfile, config['figspath'])

    elif action == 'risk-ev-choice':
        if len(args) > 0:
            trialsfile = args[0]
        else:
            trialsfile = runtools.activityfile(config['trialspath'])
            if not os.path.exists(trialsfile):
                trialsfile = runtools.behaviorfile(config['trialspath'])

        if not os.path.exists(trialsfile):
            raise SystemExit(f"Missing trial data file: {trialsfile}")

        plot_heatmap(trialsfile, config['figspath'])
        
    elif action in ('opto-sweep', 'opto-sweep-dense'):
            # Optogenetic VTA stimulation sweep
            # Test different dopamine offset levels during inference

            if action == 'opto-sweep-dense':
                opto_offsets = np.linspace(-0.3, 0.3, 13)  # -0.3 to +0.3 in 0.05 steps
            else:
                opto_offsets = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
                # opto_offsets =  [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            print("\n" + "="*80)
            print(f"STARTING OPTOSTIMULATION SWEEP: {opto_offsets}")
            print("="*80)

            # Load the base model
            model = config['model']
            pg = model.get_pg(config['savefile'], config['seed'], config['dt'])

            # Enable RPE modulation
            if not getattr(pg, 'use_rpe_modulation', pg.config.get('use_rpe_modulation', False)):
                print("\n⚠️  WARNING: RPE modulation is OFF. Enabling it now...")
                pg.use_rpe_modulation = True
            else:
                pg.use_rpe_modulation = True

            # Saved models may carry older clamp values. For acute push-pull
            # optostimulation, use the direct-context-like [0.1, 1.9] gain range.
            pg.rpe_modulation_gain = config.get(
                'rpe_modulation_gain',
                getattr(pg, 'rpe_modulation_gain', 3.0)
            )
            pg.rpe_modulation_clamp = config.get('rpe_modulation_clamp', 0.9)
            print(
                f"RPE modulation: gain={pg.rpe_modulation_gain}, "
                f"clamp=±{pg.rpe_modulation_clamp}"
            )

            trialsfiles = {}

            # Loop through opto levels, run inference, and save data
            for opto_offset in opto_offsets:
                print(f"\nProcessing Opto Offset = {opto_offset:+.3f}...")

                # Set optostimulation parameters
                pg.opto_stim_offset = opto_offset
                pg.opto_stim_gain = 1.0
                pg.opto_stim_phase = 'all'

                pg.rng = np.random.RandomState(seed=999)  # Fixed seed for comparison

                # Generate psychometric trial set (same as context-sweep)
                task = model.Task()
                psychometric_specs = generate_psychometric_trial_set(trials_per_comparison=2)
                pg.rng.shuffle(psychometric_specs)

                trials = []
                for trial_spec in psychometric_specs:
                    trials.append(task.get_condition(pg.rng, pg.dt, trial_spec))

                # RUN INFERENCE WITH OPTOSTIMULATION
                results = pg.run_trials(trials, return_states=True)

                # Pack data
                packed = [
                    trials, results['U'], results['Z'], results['Z_b'],
                    results['A'], results['R'], results['M'], results['perf'],
                    results['r_policy'], results['r_value']
                ]
                if 'RPE_objective' in results:
                    packed.extend([results['RPE_objective'], results['RPE_subjective']])

                packed.extend([
                    results['Policy_Values'],
                    results['Policy_D1_Pull'],
                    results['Policy_D2_Pull'],
                    results.get('r_policy_mod', results['r_policy'])
                ])

                # Save file for this opto level
                opto_str = f"{opto_offset:+.3f}".replace('.', 'p').replace('+', 'pos').replace('-', 'neg')

                os.makedirs(config['trialspath'], exist_ok=True)
                filepath = os.path.join(config['trialspath'], f'trials_activity_opto{opto_str}.pkl')

                utils.save(filepath, packed)
                trialsfiles[opto_offset] = filepath

                print(f"  Saved: {filepath}")
                if 'RPE_continuous' in results:
                    rpe_mean = results['RPE_continuous'].mean().item()
                    print(f"  Mean RPE: {rpe_mean:+.4f} (shift from natural: ~{opto_offset:+.3f})")

            # Generate megaplot (reuse context plotting functions with opto data)
            print("\n" + "="*80)
            print("GENERATING OPTOSTIMULATION MEGAPLOT")
            print("="*80)

            # Use context plotting functions (they work the same way)
            plot_opto_mega_comparison(opto_offsets, trialsfiles, config['savefile'], config['figspath'])
            plot_opto_choice_probability_curves(opto_offsets, trialsfiles, config['figspath'])

            print("\n✓ Optostimulation sweep complete!")
            print(f"Results saved in: {config['trialspath']}")
            print(f"Plots saved in: {config['figspath']}")

    elif action in ('context-sweep', 'context-sweep-dense', 'context-sweep-curves-dense', 'context-sweep-0p2'):
            # 1. Define the contexts we want to sweep
            if action == 'context-sweep-0p2':
                contexts = context_values_step(0.2)
            elif action in ('context-sweep-dense', 'context-sweep-curves-dense'):
                contexts = dense_context_values()
            else:
                contexts = [0]
                # -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0

            start_context = float(args[0]) if len(args) > 0 else None
            if start_context is not None:
                contexts = [ctx for ctx in contexts if ctx >= start_context - 1e-9]
	            
            print("\n" + "="*80)
            print(f"STARTING CONTEXT SWEEP: {contexts}")
            print("="*80)

            # 2. Load the base model ONCE
            model = config['model']
            pg = model.get_pg(config['savefile'], config['seed'], config['dt'])
            
            trialsfiles = {}
            
            # 3. Loop through contexts, run inference, and save data
            for ctx in contexts:
                ctx_str = format_context_str(ctx)
                os.makedirs(config['trialspath'], exist_ok=True)
                filepath = os.path.join(config['trialspath'], f'trials_activity_ctx{ctx_str}.pkl')
                if os.path.exists(filepath):
                    print(f"\nSkipping Context = {ctx:+.2f}; using existing {filepath}")
                    trialsfiles[ctx] = filepath
                    continue

                print(f"\nProcessing Context = {ctx:+.2f}...")
	                
                pg.rng = np.random.RandomState(seed=999) # Prevent sequence memorization
                trials_per_condition = 2
                # NEW CODE - STRUCTURED PSYCHOMETRIC TRIALS
                task = model.Task()

                # Generate psychometric trial set (matched-EV comparisons)
                psychometric_specs = generate_psychometric_trial_set(trials_per_comparison=2)
                pg.rng.shuffle(psychometric_specs)  # Randomize order

                trials = []
                for trial_spec in psychometric_specs:
                    trials.append(task.get_condition(pg.rng, pg.dt, trial_spec))
                
                # RUN INFERENCE WITH EXPLICIT CONTEXT
                results = pg.run_trials(trials, return_states=True, context_input=ctx)
                
                # Pack data to match format for load_trial_data()
                packed = [
                    trials, results['U'], results['Z'], results['Z_b'],
                    results['A'], results['R'], results['M'], results['perf'],
                    results['r_policy'], results['r_value']
                ]
                if 'RPE_objective' in results:
                    packed.extend([results['RPE_objective'], results['RPE_subjective']])
                
                # --- NEW: Append the Policy Value Arrays ---
                packed.extend([
                    results['Policy_Values'], 
                    results['Policy_D1_Pull'], 
                    results['Policy_D2_Pull'],
                    results.get('r_policy_mod', results['r_policy'])
                ])
                
                utils.save(filepath, packed)
                trialsfiles[ctx] = filepath

            if action == 'context-sweep-curves-dense':
                plot_context_choice_probability_curves(contexts, trialsfiles, config['figspath'])
                plot_context_choice_probability_mega(contexts, trialsfiles, config['figspath'])
                return
                
            # 4. Generate the Mega-Plot
            plot_context_mega_comparison(contexts, trialsfiles, config['savefile'], config['figspath'])

            # 5. Generate the Policy Subjective Value Plots
            print("\nGenerating Policy Subjective Value Plots...")
            for ctx in contexts:
                filepath = trialsfiles[ctx]
                plot_policy_subjective_values(
                    filepath,
                    config['figspath'],
                    context_val=ctx,
                    dopamine_split=model_uses_dopamine_split(config['savefile'])
                )

            plot_mega_policy_subjective_values(
                contexts,
                trialsfiles,
                config['figspath'],
                dopamine_split=model_uses_dopamine_split(config['savefile'])
            )
            plot_context_choice_probability_curves(contexts, trialsfiles, config['figspath'])
            plot_context_choice_probability_mega(contexts, trialsfiles, config['figspath'])

    elif action in ('context-curves', 'context-curves-dense'):
            if action == 'context-curves-dense':
                contexts = dense_context_values()
            else:
                contexts = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
            trialsfiles = {}

            for ctx in contexts:
                ctx_str = format_context_str(ctx)
                filepath = os.path.join(config['trialspath'], f'trials_activity_ctx{ctx_str}.pkl')
                if not os.path.exists(filepath):
                    raise SystemExit(
                        f"Missing context trial file: {filepath}\n"
                        "Run context-sweep once first to generate the fixed-context trial files."
                    )
                trialsfiles[ctx] = filepath

            plot_context_choice_probability_curves(contexts, trialsfiles, config['figspath'])
            plot_context_choice_probability_mega(contexts, trialsfiles, config['figspath'])

    elif action == 'mega-comparison':
        # Define all kappa values
        kappas = [
            -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]

        # Get base name from config
        base_name = config.get('name', 'gambling')

        # Determine repository root (go up from scripts/plotting/)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        trials_base = os.path.join(repo_root, 'data', 'trials')
        weights_base = os.path.join(repo_root, 'data', 'weights')

        print("\n" + "="*80)
        print("MEGA-COMPARISON: Scanning for kappa sweep models")
        print("="*80)
        print(f"Trials base: {trials_base}")
        print(f"Weights base: {weights_base}")
        print(f"\nLooking for models:")

        trialsfiles = {}
        modelfiles = {}
        for kappa in kappas:
            if kappa == 0.0:
                # Special case: κ=0 is just "gambling" without suffix
                dirname = 'gambling'
                modelname = 'gambling'
            else:
                # Convert kappa to suffix: -0.5 → "neg0p5", 0.5 → "pos0p5"
                if kappa < 0:
                    suffix = f"neg{abs(kappa)}".replace('.', 'p')
                else:
                    suffix = f"pos{kappa}".replace('.', 'p')
                dirname = f'gambling{suffix}'
                modelname = f'gambling{suffix}'

            trialsfile = os.path.join(trials_base, dirname, 'trials_activity.pkl')
            modelfile = os.path.join(weights_base, modelname, f'{modelname}.pkl')

            if os.path.exists(trialsfile) and os.path.exists(modelfile):
                trialsfiles[kappa] = trialsfile
                modelfiles[kappa] = modelfile
                print(f"  ✓ κ={kappa:+.1f}: {dirname}")
            else:
                print(f"  ✗ κ={kappa:+.1f}: NOT FOUND ({dirname})")
                # Try trials_behavior.pkl as fallback (won't have neural data though)
                if os.path.exists(trialsfile):
                    print(f"     (found trials but missing model file)")
                if os.path.exists(modelfile):
                    print(f"     (found model but missing trials_activity.pkl)")

        if len(trialsfiles) < 2:
            print(f"\n❌ Only found {len(trialsfiles)} trial files, need at least 2")
            print(f"\n💡 You need to run 'trials-a' action for each kappa model first!")
            print(f"   Example: python3 scripts/training/train.py tasks/gambling.py --suffix neg0p8 run scripts/plotting/gambling.py trials-a 2")
            return

        # Extract kappa values for which we found files (in order)
        kappa_values = [k for k in kappas if k in trialsfiles]

        print(f"\n🎨 Generating mega-comparison plot with {len(kappa_values)} models...")
        plot_mega_comparison(kappa_values, trialsfiles, modelfiles, config['figspath'])

    elif action == 'finetuned-kappa-mega':
        kappas = [
            -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]

        def tag_for_kappa(kappa):
            return f"{kappa:+.1f}".replace('+', '').replace('-', 'neg').replace('.', 'p')

        data_root = os.path.dirname(os.path.dirname(os.path.dirname(config['trialspath'])))

        print("\n" + "="*80)
        print("FINETUNED KAPPA MEGA-COMPARISON")
        print("="*80)
        print(f"Data root: {data_root}")

        trialsfiles = {}
        modelfiles = {}
        for kappa in kappas:
            tag = tag_for_kappa(kappa)
            group = f"finetuned_kappa_{kappa:.1f}"
            name = f"gambling_ft_kappa_{tag}"
            trialsfile = os.path.join(data_root, group, 'trials', name, 'trials_activity.pkl')
            modelfile = os.path.join(data_root, group, 'weights', name, f'{name}.pkl')

            if os.path.exists(trialsfile) and os.path.exists(modelfile):
                trialsfiles[kappa] = trialsfile
                modelfiles[kappa] = modelfile
                print(f"  ✓ κ={kappa:+.1f}: {name}")
            else:
                print(f"  ✗ κ={kappa:+.1f}: missing")
                if not os.path.exists(trialsfile):
                    print(f"     missing trials: {trialsfile}")
                if not os.path.exists(modelfile):
                    print(f"     missing model:  {modelfile}")

        kappa_values = [k for k in kappas if k in trialsfiles]
        if len(kappa_values) < 2:
            raise SystemExit(f"Need at least 2 finetuned kappa trial files, found {len(kappa_values)}.")

        plot_mega_comparison(
            kappa_values,
            trialsfiles,
            modelfiles,
            config['figspath'],
            plot_title='Finetuned Kappa Comparison'
        )
        plot_kappa_comparison(kappa_values, trialsfiles, config['figspath'])
        plot_kappa_summary(kappa_values, trialsfiles, config['figspath'])
        plot_kappa_value_tuning(kappa_values, trialsfiles, config['figspath'])

    elif action == 'hardwired-kappa-mega':
        kappas = [
            -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]

        def tag_for_kappa(kappa):
            return f"{kappa:+.1f}".replace('+', '').replace('-', 'neg').replace('.', 'p')

        data_root = os.path.dirname(os.path.dirname(os.path.dirname(config['trialspath'])))

        print("\n" + "="*80)
        print("HARDWIRED KAPPA MEGA-COMPARISON")
        print("="*80)
        print(f"Data root: {data_root}")

        trialsfiles = {}
        modelfiles = {}
        for kappa in kappas:
            tag = tag_for_kappa(kappa)
            group = f"hardwired_kappa_{kappa:.1f}"
            name = f"gambling_hardwired_kappa_{tag}"
            trialsfile = os.path.join(data_root, group, 'trials', name, 'trials_activity.pkl')
            modelfile = os.path.join(data_root, group, 'weights', name, f'{name}.pkl')

            if os.path.exists(trialsfile) and os.path.exists(modelfile):
                trialsfiles[kappa] = trialsfile
                modelfiles[kappa] = modelfile
                print(f"  ✓ κ={kappa:+.1f}: {name}")
            else:
                print(f"  ✗ κ={kappa:+.1f}: missing")
                if not os.path.exists(trialsfile):
                    print(f"     missing trials: {trialsfile}")
                if not os.path.exists(modelfile):
                    print(f"     missing model:  {modelfile}")

        kappa_values = [k for k in kappas if k in trialsfiles]
        if len(kappa_values) < 2:
            raise SystemExit(f"Need at least 2 hardwired kappa trial files, found {len(kappa_values)}.")

        plot_mega_comparison(
            kappa_values,
            trialsfiles,
            modelfiles,
            config['figspath'],
            plot_title='Hardwired Kappa Comparison'
        )
        plot_kappa_comparison(kappa_values, trialsfiles, config['figspath'])
        plot_kappa_summary(kappa_values, trialsfiles, config['figspath'])
        plot_kappa_value_tuning(kappa_values, trialsfiles, config['figspath'])

    elif action == 'finetuned-kappa-value-tuning':
        kappas = [
            -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]

        def tag_for_kappa(kappa):
            return f"{kappa:+.1f}".replace('+', '').replace('-', 'neg').replace('.', 'p')

        data_root = os.path.dirname(os.path.dirname(os.path.dirname(config['trialspath'])))
        trialsfiles = {}
        for kappa in kappas:
            tag = tag_for_kappa(kappa)
            group = f"finetuned_kappa_{kappa:.1f}"
            name = f"gambling_ft_kappa_{tag}"
            trialsfile = os.path.join(data_root, group, 'trials', name, 'trials_activity.pkl')
            if os.path.exists(trialsfile):
                trialsfiles[kappa] = trialsfile
                print(f"  ✓ κ={kappa:+.1f}: {name}")
            else:
                print(f"  ✗ κ={kappa:+.1f}: missing trials")

        kappa_values = [k for k in kappas if k in trialsfiles]
        if len(kappa_values) < 2:
            raise SystemExit(f"Need at least 2 finetuned kappa trial files, found {len(kappa_values)}.")

        plot_kappa_value_tuning(kappa_values, trialsfiles, config['figspath'])

    elif action == 'distribution-comparison':
        # Compare 4 models with different per-neuron kappa distributions:
        # gaussian, baseline, uniform, gaussian_neg0.2

        print("\n" + "="*80)
        print("DISTRIBUTION COMPARISON: 4-column plot")
        print("="*80)
        print("Expected models:")
        print("  - gaussian: Trained with Gaussian κ ~ N(0, 0.3²)")
        print("  - baseline: Trained with κ = 0 (all neurons)")
        print("  - uniform:  Trained with Uniform κ ~ U[-0.3, 0.3]")
        print("  - gaussian_neg0.2: Trained with Gaussian κ ~ N(-0.2, 0.3²)")
        print("="*80 + "\n")

        # Trial and model files are inside the package under examples/work/
        examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        trials_base = os.path.join(examples_dir, 'work', 'trials')
        data_base = os.path.join(examples_dir, 'work', 'data')

        # Expected model names (you can customize these)
        model_configs = {
            'gaussian': 'gambling_dist_gaussian',
            'baseline': 'gambling',
            'uniform': 'gambling_dist_uniform',
            'gaussian_neg0.2': 'gambling_dist_gaussian_neg0.2'
        }

        trialsfiles = {}
        modelfiles = {}

        for dist_name, modelname in model_configs.items():
            dirname = modelname
            trialsfile = os.path.join(trials_base, dirname, 'trials_activity.pkl')
            modelfile = os.path.join(weights_base, modelname, f'{modelname}.pkl')

            if os.path.exists(trialsfile) and os.path.exists(modelfile):
                trialsfiles[dist_name] = trialsfile
                modelfiles[dist_name] = modelfile
                print(f"  ✓ {dist_name}: {dirname}")
            else:
                print(f"  ✗ {dist_name}: NOT FOUND ({dirname})")
                if os.path.exists(trialsfile):
                    print(f"     (found trials but missing model file)")
                if os.path.exists(modelfile):
                    print(f"     (found model but missing trials_activity.pkl)")

        if 'baseline' not in trialsfiles:
            print("\n❌ Baseline model is required!")
            print("   Run: python examples/do.py examples/models/gambling.py train")
            print("   Then: python examples/do.py examples/models/gambling.py run examples/analysis/gambling.py trials-a 2")
            return

        if len(trialsfiles) < 2:
            print(f"\n❌ Only found {len(trialsfiles)} trial files, need at least 2 (including baseline)")
            print(f"\n💡 Train models with per-neuron kappa distributions, then run 'trials-a' for each:")
            print(f"   Example: python examples/do.py examples/models/gambling.py run examples/analysis/gambling.py trials-a 2 --suffix _dist_gaussian")
            return

        print(f"\n🎨 Generating distribution comparison plot with {len(trialsfiles)} models...")
        plot_distribution_comparison(trialsfiles, modelfiles, config['figspath'])

    elif action == 'temporal-activity':
            # Plot temporal activity sorted by ΔEV (Figure 3c,d style)
            trialsfile = runtools.activityfile(config['trialspath'])

            # Get arguments
            network = args[0] if len(args) > 0 else 'value'
            n_examples = int(args[1]) if len(args) > 1 else 3
            kappa = float(args[2]) if len(args) > 2 else config.get('kappa', None)

            print(f"\nPlotting temporal activity for {network} network (κ={kappa})...")
            results = plot_temporal_activity_sorted(
                trialsfile, config['figspath'], 
                network=network, n_examples=n_examples, kappa=kappa
            )

            if results is not None:
                print(f"\nTemporal activity analysis complete!")
                print(f"  Example neurons: {[idx for _, idx in results['example_neurons']]}")
                print(f"  ΔEV range: [{results['delta_evs_sorted'].min():.2f}, {results['delta_evs_sorted'].max():.2f}]")
    elif action == 'regression-scatter':
        # Plot regression scatter with sorted coloring
        trialsfile = runtools.activityfile(config['trialspath'])

        # Get optional arguments
        network = args[0] if len(args) > 0 else 'policy'
        kappa = float(args[1]) if len(args) > 1 else config.get('kappa', None)

        print(f"\nGenerating regression scatter plot for {network} network...")
        results = plot_regression_scatter(trialsfile, config['figspath'], 
                                         network=network, kappa=kappa)

        if results is not None:
            print("\nRegression scatter complete!")

    elif action == 'rpe-signals':
        # Plot Reward Prediction Error (RPE) signals - both objective and subjective
        trialsfile = runtools.activityfile(config['trialspath'])

        # Get kappa from args or config
        if len(args) > 0:
            kappa = float(args[0])
        else:
            kappa = config.get('kappa', None)

        # Number of example trials
        if len(args) > 1:
            n_examples = int(args[1])
        else:
            n_examples = 2

        print(f"\nGenerating RPE signal plots (κ={kappa})...")
        results = plot_rpe_signals(trialsfile, config['figspath'], kappa=kappa, n_examples=n_examples)

        if results is not None:
            print("\nRPE signal analysis complete!")
            print(f"  Objective RPE mean: {results['rpe_objective'].mean():.4f}")
            print(f"  Subjective RPE mean: {results['rpe_subjective'].mean():.4f}")

    elif action == 'neural-analysis':
        # Complete neural activity analysis (requires activity data from 'trials-a')
        trialsfile = runtools.activityfile(config['trialspath'])

        print("\n" + "="*60)
        print("Generating neural activity visualizations...")
        print("="*60)

        # 1. Predicted values heatmap
        print("\n1. Predicted values (5x5 grid)...")
        plot_predicted_values(trialsfile, config['figspath'])

        # 2. Temporal activity plots for policy network
        print("\n2. Policy network temporal activity...")
        plot_temporal_activity_sorted(trialsfile, config['figspath'], network='policy', n_examples=3)

        # 3. Temporal activity plots for value network
        print("\n3. Value network temporal activity...")
        plot_temporal_activity_sorted(trialsfile, config['figspath'], network='value', n_examples=3)

        # 4. Regression analysis
        print("\n4. Regression analysis (βHH-LL vs βEV)...")
        plot_regression_analysis(trialsfile, config['figspath'])

        print("\n" + "="*60)
        print("Neural analysis complete!")
        print("="*60)

    elif action == 'kappa-comparison':
        # Plot comparison across multiple kappa values
        # Expected format: kappa-comparison k1,k2,k3,... path1,path2,path3,...
        try:
            kappa_str = args[0]
            paths_str = args[1]

            kappa_values = [float(k) for k in kappa_str.split(',')]
            paths = paths_str.split(',')

            if len(kappa_values) != len(paths):
                print("Error: Number of kappa values must match number of paths")
                return

            # Build dictionary mapping kappa to trial files
            trialsfiles = {}
            for kappa, path in zip(kappa_values, paths):
                trialsfiles[kappa] = runtools.behaviorfile(path)

            plot_kappa_comparison(kappa_values, trialsfiles, config['figspath'])
            plot_kappa_summary(kappa_values, trialsfiles, config['figspath'])

        except Exception as e:
            print(f"Error in kappa-comparison: {e}")
            print("Usage: kappa-comparison 'k1,k2,k3' 'path1,path2,path3'")
            print("Example: kappa-comparison '-0.8,-0.4,0,0.4,0.8' 'trials_k-0.8,trials_k-0.4,trials_k0,trials_k0.4,trials_k0.8'")

    elif action == 'kappa-sweep':
        # Automatically generate trials and plots for a sweep of kappa values
        try:
            kappa_min = float(args[0]) if len(args) > 0 else -0.8
            kappa_max = float(args[1]) if len(args) > 1 else 0.8
            n_kappas = int(args[2]) if len(args) > 2 else 9

            kappa_values = np.linspace(kappa_min, kappa_max, n_kappas)
            print(f"\nKappa sweep: {kappa_values}")

            trialsfiles = {}
            modelfiles = {}

            # Load base config
            model = config['model']
            base_savefile = config['savefile']

            for kappa in kappa_values:
                print(f"\n{'='*60}")
                print(f"Processing κ = {kappa:+.2f}")
                print('='*60)

                # Create kappa-specific savefile path
                kappa_str = f"{kappa:+.2f}".replace('.', 'p').replace('-', 'neg').replace('+', 'pos')
                kappa_savefile = base_savefile.replace('.pkl', f'_kappa{kappa_str}.pkl')

                # Load model with this kappa value
                pg = model.get_pg(kappa_savefile, config['seed'], config['dt'], kappa=kappa)

                # Generate trials
                trials_per_condition = 2
                n_trials = trials_per_condition * 625

                task = model.Task()
                trials = []

                # Systematic exploration
                conditions = []
                for _ in range(trials_per_condition):
                    for target_l in range(25):
                        for target_r in range(25):
                            conditions.append((target_l, target_r))

                rng = np.random.default_rng()
                rng.shuffle(conditions)

                for target_l, target_r in conditions:
                    context = {'target_l': target_l, 'target_r': target_r}
                    trials.append(task.get_condition(pg.rng, pg.dt, context))

                # Run trials
                kappa_trialspath = config['trialspath'].replace('trials', f'trials_kappa{kappa_str}')
                os.makedirs(os.path.dirname(kappa_trialspath), exist_ok=True)

                results = pg.run_trials(trials, return_states=True)
                packed = [
                    trials, results['U'], results['Z'], results['Z_b'],
                    results['A'], results['R'], results['M'], results['perf'],
                    results['r_policy'], results['r_value']
                ]
                if 'RPE_objective' in results:
                    packed.extend([results['RPE_objective'], results['RPE_subjective']])
                packed.extend([
                    results['Policy_Values'],
                    results['Policy_D1_Pull'],
                    results['Policy_D2_Pull'],
                    results.get('r_policy_mod', results['r_policy'])
                ])

                os.makedirs(kappa_trialspath, exist_ok=True)
                activity_file = os.path.join(kappa_trialspath, 'trials_activity.pkl')
                utils.save(activity_file, packed)

                # Store path
                trialsfiles[kappa] = activity_file
                modelfiles[kappa] = kappa_savefile

            # Create comparison plots
            print(f"\n{'='*60}")
            print("Generating comparison plots...")
            print('='*60)

            plot_mega_comparison(list(kappa_values), trialsfiles, modelfiles, config['figspath'])
            plot_kappa_comparison(list(kappa_values), trialsfiles, config['figspath'])
            plot_kappa_summary(list(kappa_values), trialsfiles, config['figspath'])

            print("\nKappa sweep complete!")

        except Exception as e:
            print(f"Error in kappa-sweep: {e}")
            import traceback
            traceback.print_exc()
            print("\nUsage: kappa-sweep [kappa_min] [kappa_max] [n_kappas]")
            print("Example: kappa-sweep -0.8 0.8 9")

    elif action == 'kappa-single':
        # Generate a kappa-style mega plot for the current checkpoint only.
        try:
            kappa = float(args[0]) if len(args) > 0 else float(config.get('kappa', 0.0) or 0.0)
            trials_per_condition = int(args[1]) if len(args) > 1 else 2

            print(f"\nKappa single-condition plot: κ={kappa:+.2f}")

            model = config['model']
            pg = model.get_pg(config['savefile'], config['seed'], config['dt'], kappa=kappa)
            pg.rng = np.random.RandomState(seed=999)

            task = model.Task()
            conditions = []
            for _ in range(trials_per_condition):
                for target_l in range(25):
                    for target_r in range(25):
                        conditions.append((target_l, target_r))
            pg.rng.shuffle(conditions)

            trials = [
                task.get_condition(pg.rng, pg.dt, {'target_l': target_l, 'target_r': target_r})
                for target_l, target_r in conditions
            ]

            results = pg.run_trials(trials, return_states=True)
            packed = [
                trials, results['U'], results['Z'], results['Z_b'],
                results['A'], results['R'], results['M'], results['perf'],
                results['r_policy'], results['r_value']
            ]
            if 'RPE_objective' in results:
                packed.extend([results['RPE_objective'], results['RPE_subjective']])
            packed.extend([
                results['Policy_Values'],
                results['Policy_D1_Pull'],
                results['Policy_D2_Pull'],
                results.get('r_policy_mod', results['r_policy'])
            ])

            os.makedirs(config['trialspath'], exist_ok=True)
            activity_file = os.path.join(config['trialspath'], 'trials_activity_kappa_single.pkl')
            utils.save(activity_file, packed)

            plot_mega_comparison(
                [kappa],
                {kappa: activity_file},
                {kappa: config['savefile']},
                config['figspath'],
                plot_title='Single Kappa Condition'
            )

            print("\nKappa single-condition plot complete!")

        except Exception as e:
            print(f"Error in kappa-single: {e}")
            import traceback
            traceback.print_exc()
            print("\nUsage: kappa-single [kappa] [trials_per_condition]")
            print("Example: kappa-single 0.0 2")

    elif action == 'context-sweep-gaussian':
        # Run context sweep with Gaussian sampling around each context mean
        try:
            # Parse arguments
            c_min = float(args[0]) if len(args) > 0 else -1.0
            c_max = float(args[1]) if len(args) > 1 else 1.0
            c_step = float(args[2]) if len(args) > 2 else 0.1
            context_std = float(args[3]) if len(args) > 3 else 0.1
            trials_per_condition = int(args[4]) if len(args) > 4 else 2

            print(f"\n{'='*70}")
            print("CONTEXT SWEEP WITH GAUSSIAN SAMPLING")
            print(f"{'='*70}")
            print(f"  Context range: [{c_min:.2f}, {c_max:.2f}], step={c_step:.2f}")
            print(f"  Gaussian std: {context_std:.2f}")
            print(f"  Trials per condition: {trials_per_condition}")

            # Generate context means
            contexts = np.arange(c_min, c_max + c_step/2, c_step)
            print(f"  Total contexts: {len(contexts)}")
            print(f"  Contexts: {contexts}")

            # Load the model
            model = config['model']
            pg = model.get_pg(config['savefile'], config['seed'], config['dt'])

            trialsfiles = {}

            # Loop through contexts, run inference with Gaussian sampling
            for ctx_mean in contexts:
                print(f"\nProcessing Context mean = {ctx_mean:+.2f}...")

                pg.rng = np.random.RandomState(seed=999)
                task = model.Task()

                # Generate psychometric trial set
                psychometric_specs = generate_psychometric_trial_set(trials_per_comparison=trials_per_condition)
                pg.rng.shuffle(psychometric_specs)

                trials = []
                for trial_spec in psychometric_specs:
                    trials.append(task.get_condition(pg.rng, pg.dt, trial_spec))

                # RUN INFERENCE WITH GAUSSIAN-SAMPLED CONTEXT
                context_spec = {
                    'distribution': 'gaussian',
                    'mean': float(ctx_mean),
                    'std': context_std,
                    'low': -1.0,
                    'high': 1.0
                }
                results = pg.run_trials(trials, return_states=True, context_input=context_spec)

                # Pack data to match format for load_trial_data()
                packed = [
                    trials, results['U'], results['Z'], results['Z_b'],
                    results['A'], results['R'], results['M'], results['perf'],
                    results['r_policy'], results['r_value']
                ]
                if 'RPE_objective' in results:
                    packed.extend([results['RPE_objective'], results['RPE_subjective']])

                # Append the Policy Value Arrays
                packed.extend([
                    results['Policy_Values'],
                    results['Policy_D1_Pull'],
                    results['Policy_D2_Pull'],
                    results.get('r_policy_mod', results['r_policy'])
                ])

                # Save file for this context mean
                ctx_str = format_context_str(ctx_mean)
                os.makedirs(config['trialspath'], exist_ok=True)
                filepath = os.path.join(config['trialspath'], f'trials_activity_ctx{ctx_str}_gaussian.pkl')

                utils.save(filepath, packed)
                trialsfiles[ctx_mean] = filepath

                # Print actual sampled context statistics
                actual_contexts = results['contexts'].cpu().numpy()
                print(f"  Actual context: mean={actual_contexts.mean():.3f}, std={actual_contexts.std():.3f}")

            # Generate the plots using existing functions
            print(f"\n{'='*70}")
            print("GENERATING PLOTS")
            print(f"{'='*70}")

            plot_context_mega_comparison(list(contexts), trialsfiles, config['savefile'], config['figspath'])
            plot_mega_policy_subjective_values(
                list(contexts),
                trialsfiles,
                config['figspath'],
                dopamine_split=model_uses_dopamine_split(config['savefile'])
            )
            plot_context_choice_probability_curves(list(contexts), trialsfiles, config['figspath'])
            plot_context_choice_probability_mega(list(contexts), trialsfiles, config['figspath'])

            print(f"\n{'='*70}")
            print("Context sweep with Gaussian sampling complete!")
            print(f"{'='*70}")

        except Exception as e:
            print(f"Error in context-sweep-gaussian: {e}")
            import traceback
            traceback.print_exc()
            print("\nUsage: context-sweep-gaussian [c_min] [c_max] [c_step] [context_std] [trials_per_condition]")
            print("Example: context-sweep-gaussian -1.0 1.0 0.1 0.1 2")

    else:
        print(f"Unrecognized action: {action}")
        print("Available actions:")
        print("  trials-b                - Generate trial data (behavior only)")
        print("  trials-a                - Generate trial data with neural activity")
        print("  behavior                - Plot behavioral heatmap")
        print("  risk-ev-choice          - Plot behavioral proportion by ΔHH-LL and ΔEV")
        print("  neural-analysis         - Complete neural activity analysis (requires trials-a)")
        print("  kappa-comparison        - Compare multiple kappa values")
        print("  kappa-sweep             - Automated sweep across kappa values")
        print("  kappa-single            - Current-checkpoint kappa-style plot for one condition")
        print("  finetuned-kappa-value-tuning - Plot critic V over trials and V/P curves")
        print("  context-sweep           - Context sweep with FIXED context values")
        print("  context-sweep-gaussian  - Context sweep with GAUSSIAN sampling around each mean")


def plot_regression_scatter(trialsfile, figspath, network='policy', kappa=None):
    """
    Scatter plot of regression coefficients for HH-LL and EV for all neurons.
    Neurons are colored by their neuron index (sorted by β_EV value).
    """
    td = load_trial_data(trialsfile)

    if td['format'] == 'behavior':
        print("Error: This analysis requires activity data. Use 'trials-a' action.")
        return None

    if network == 'policy':
        neural_data = to_numpy(td['r_policy'])
        title_prefix = 'Policy Network'
    else:
        neural_data = to_numpy(td['r_value'])
        title_prefix = 'Value Network'

    delta_hh_lls, delta_evs, _ = compute_deltas(td['trials'])

    T, n_trials, n_neurons = neural_data.shape
    beta_hh_ll, beta_ev, r_squared = [], [], []

    for n in range(n_neurons):
        activity = np.mean(neural_data[25:50, :, n], axis=0)
        X = np.column_stack([delta_hh_lls, delta_evs, np.ones(len(activity))])
        coeffs, _, _, _ = np.linalg.lstsq(X, activity, rcond=None)
        beta_hh_ll.append(coeffs[0])
        beta_ev.append(coeffs[1])

        y_pred = X @ coeffs
        ss_res = np.sum((activity - y_pred) ** 2)
        ss_tot = np.sum((activity - np.mean(activity)) ** 2)
        r_squared.append(1 - (ss_res / ss_tot) if ss_tot > 0 else 0)

    beta_hh_ll = np.array(beta_hh_ll)
    beta_ev = np.array(beta_ev)
    r_squared = np.array(r_squared)

    sort_idx = np.argsort(beta_ev)
    neuron_colors = np.zeros(n_neurons)
    neuron_colors[sort_idx] = np.arange(n_neurons)

    colors_map = LinearSegmentedColormap.from_list('teal_brown',
                                                     ['#008080', '#90EE90', '#FFD700', '#8B4513'])

    fig, ax = plt.subplots(figsize=(7, 6))

    scatter = ax.scatter(
        beta_hh_ll, beta_ev, c=neuron_colors, cmap=colors_map,
        s=80, alpha=0.8, edgecolors='none', vmin=0, vmax=n_neurons-1
    )

    ax.axhline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlim([-0.5, 0.5]); ax.set_ylim([-0.5, 0.5])
    ax.set_xlabel(r'$\beta_{HH-LL}$', fontsize=20, color='magenta')
    ax.set_ylabel(r'$\beta_{EV}$', fontsize=20, color='cyan')
    ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([-0.5, 0, 0.5])
    ax.tick_params(labelsize=14)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, n_neurons-1])
    cbar.set_ticklabels(['1', str(n_neurons)])
    cbar.ax.tick_params(labelsize=14)

    ax.set_aspect('equal')
    plt.tight_layout()

    if kappa is not None:
        savefile = os.path.join(figspath, f'{network}_regression_scatter_kappa{format_kappa_str(kappa)}.png')
    else:
        savefile = os.path.join(figspath, f'{network}_regression_scatter.png')

    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"\nSaved {network} regression scatter to {savefile}")
    print(f"\n{title_prefix} Regression Analysis:")
    print(f"  β_HH-LL range: [{beta_hh_ll.min():.3f}, {beta_hh_ll.max():.3f}]")
    print(f"  β_EV range: [{beta_ev.min():.3f}, {beta_ev.max():.3f}]")
    print(f"  Mean R²: {np.mean(r_squared):.3f}")
    plt.close()

    return {
        'beta_hh_ll': beta_hh_ll, 'beta_ev': beta_ev,
        'r_squared': r_squared, 'sort_idx': sort_idx, 'neuron_colors': neuron_colors
    }
    
def plot_temporal_activity_sorted(trialsfile, figspath, network='value', n_examples=3, kappa=None):
    """
    Plot temporal neural activity sorted by ΔEV (Figure 3c,d style from poster).
    """
    print("\n" + "="*80)
    print(f"TEMPORAL ACTIVITY ANALYSIS - {network.upper()} NETWORK")
    print("="*80)

    td = load_trial_data(trialsfile)

    if td['format'] == 'behavior':
        print("Error: This analysis requires activity data. Use 'trials-a' action.")
        return None

    if network == 'policy':
        neural_data = to_numpy(td['r_policy'])
        title_prefix = 'Policy Network'
        network_color = '#8B008B'
    else:
        neural_data = to_numpy(td['r_value'])
        title_prefix = 'Value Network'
        network_color = '#0000CD'

    # ===== RESCALE FIRING RATES TO [0, 1] =====
    # Assuming neural_data comes from tanh(x), which is in [-1, 1]
    # If it's already from tanh, rescale. Otherwise, apply tanh first.
    # Check the range to determine
    data_min = neural_data.min()
    data_max = neural_data.max()

    if data_min < 0 and data_max <= 1.0:
        # Data is likely already tanh output in [-1, 1]
        print(f"  Detected tanh range: [{data_min:.3f}, {data_max:.3f}]")
        neural_data = (neural_data + 1) / 2  # Rescale to [0, 1]
        print(f"  Rescaled to: [{neural_data.min():.3f}, {neural_data.max():.3f}]")
    elif data_min >= 0:
        # Data might already be in [0, 1] or needs no transformation
        print(f"  Data range: [{data_min:.3f}, {data_max:.3f}]")
        if data_max > 1.5:
            # Might be raw states, apply tanh + rescale
            neural_data = np.tanh(neural_data)
            neural_data = (neural_data + 1) / 2
            print(f"  Applied tanh and rescaled to: [{neural_data.min():.3f}, {neural_data.max():.3f}]")

    T, n_trials, n_neurons = neural_data.shape
    time_ms = np.arange(T) * 10  # Convert to milliseconds

    print(f"\n{title_prefix}: {n_neurons} neurons, {n_trials} trials, {T} timesteps")

    # Calculate ΔEV for each trial
    _, delta_evs, _ = compute_deltas(td['trials'])

    # Sort trials by ΔEV
    sort_idx = np.argsort(delta_evs)
    delta_evs_sorted = delta_evs[sort_idx]

    print(f"ΔEV range: [{delta_evs.min():.2f}, {delta_evs.max():.2f}]")

    # Find example neurons with diverse ΔEV tuning
    # Compute mean activity during stimulus period for each neuron
    stimulus_activity = np.mean(neural_data[50:76, :, :], axis=0)  # (trials, neurons)

    # Find neurons that correlate with ΔEV
    correlations = []
    for n in range(n_neurons):
        corr = np.corrcoef(stimulus_activity[:, n], delta_evs)[0, 1]
        correlations.append(corr)
    correlations = np.array(correlations)

    # Select diverse neurons
    # High positive correlation (prefers high EV on right)
    high_pos_idx = np.argmax(correlations)
    # High negative correlation (prefers low EV on right / high EV on left)
    high_neg_idx = np.argmin(correlations)
    # Near zero correlation (not tuned to EV)
    mid_idx = np.argsort(np.abs(correlations))[0]

    example_neurons = [
        (f'Example #1', high_pos_idx),
        (f'Example #2', mid_idx),
        (f'Example #3', high_neg_idx),
    ]

    print(f"\nSelected example neurons:")
    for label, idx in example_neurons:
        print(f"  {label}: neuron #{idx}")

    # Create figure
    fig, axes = plt.subplots(n_examples, 1, figsize=(8, 3*n_examples))
    if n_examples == 1:
        axes = [axes]

    # Color map for lines (sorted by ΔEV)
    colors_map = plt.cm.viridis(np.linspace(0, 1, n_trials))

    # Plot each example neuron
    for ax, (label, neuron_idx) in zip(axes, example_neurons):
        # Get activity for this neuron, sorted by ΔEV
        activity_sorted = neural_data[:, sort_idx, neuron_idx]  # (T, n_trials)

        # Bin trials into groups for cleaner visualization
        n_bins = 10
        bin_size = n_trials // n_bins

        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            end_idx = (bin_idx + 1) * bin_size if bin_idx < n_bins - 1 else n_trials

            # Average activity across trials in this bin
            activity_bin = np.mean(activity_sorted[:, start_idx:end_idx], axis=1)

            # Color based on middle of bin
            color_idx = (start_idx + end_idx) // 2

            ax.plot(time_ms, activity_bin, 
                   color=colors_map[color_idx],
                   linewidth=2, alpha=0.8)

        # Mark epoch boundaries
        ax.axvline(250, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(500, color='black', linestyle='--', alpha=0.3, linewidth=1)

        # Shade decision period
        ax.axvspan(500, 760, alpha=0.15, color='gray')

        # Labels and formatting
        ax.set_ylabel('Activity', fontsize=12)
        ax.set_title(label, fontsize=11, weight='bold')
        ax.set_xlim([0, 770])
        # ===== CHANGED: Set y-axis to [0, 1] range =====
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.2)

        # Only show x-label on bottom plot
        if ax == axes[-1]:
            ax.set_xlabel('Time from trial start (ms)', fontsize=12)

        # Add epoch labels at top
        if ax == axes[0]:
            ax.text(125, 0.93, 'Fixation', 
                   ha='center', fontsize=9, color='gray')
            ax.text(375, 0.93, 'Stimulus', 
                   ha='center', fontsize=9, color='gray')
            ax.text(630, 0.93, 'Decision', 
                   ha='center', fontsize=9, color='gray')

    # Add colorbar to show ΔEV gradient
    sm = plt.cm.ScalarMappable(
        cmap='viridis',
        norm=plt.Normalize(vmin=delta_evs_sorted.min(), vmax=delta_evs_sorted.max())
    )
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('ΔEV\nlog(R/L)\n', fontsize=10, rotation=0, labelpad=20)

    # Overall title
    if kappa is not None:
        fig.suptitle(f'{title_prefix} - Temporal Activity (κ={kappa:.1f})', 
                    fontsize=14, weight='bold', y=0.995, color=network_color)
    else:
        fig.suptitle(f'{title_prefix} - Temporal Activity', 
                    fontsize=14, weight='bold', y=0.995, color=network_color)

    plt.tight_layout(rect=[0, 0, 0.9, 0.99])

    # Save
    if kappa is not None:
        savefile = os.path.join(figspath, f'{network}_temporal_activity_kappa{format_kappa_str(kappa)}.png')
    else:
        savefile = os.path.join(figspath, f'{network}_temporal_activity.png')

    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {savefile}")
    plt.close()

    print("="*80 + "\n")

    return {
        'example_neurons': example_neurons,
        'correlations': correlations,
        'delta_evs_sorted': delta_evs_sorted
    }

def plot_mega_comparison(kappa_values, trialsfiles, modelfiles, figspath, plot_title=None, horizontal=False):
    """
    Create mega-plot comparing kappa values.

    Parameters
    ----------
    horizontal : bool
        If True, create horizontal layout (11 columns × n_kappas rows) for single condition plots.
        If False, create vertical layout (11 rows × n_kappas columns) for multi-condition plots.
    """
    kappa_values = list(kappa_values)
    n_kappas = len(kappa_values)

    # For single condition, use horizontal layout by default
    if n_kappas == 1 and not horizontal:
        horizontal = True

    if horizontal:
        print(f"\nCREATING MEGA-PLOT (HORIZONTAL): {n_kappas} ROWS × 11 COLUMNS\n")
    else:
        print(f"\nCREATING MEGA-PLOT (VERTICAL): 11 ROWS × {n_kappas} COLUMNS\n")

    labels = policy_split_labels(
        any(model_uses_dopamine_split(path) for path in modelfiles.values())
    )

    all_data, baseline_data = _load_comparison_data(kappa_values, trialsfiles, modelfiles)

    if baseline_data is None:
        first_available = next(iter(all_data.values()), None)
        if first_available is None:
            print("Error: No kappa data found!")
            return
        baseline_data = first_available
        print("Warning: No κ=0 baseline found; using first available condition as weight baseline.")

    policy_lim, value_lim = _compute_weight_limits(all_data, baseline_data, kappa_values)

    grid_max_vals = []
    for kappa in kappa_values:
        if kappa in all_data and all_data[kappa].get('grid_V') is not None:
            grid_max_vals.extend([
                np.nanmax(np.abs(all_data[kappa]['grid_D1'])),
                np.nanmax(np.abs(all_data[kappa]['grid_D2'])),
                np.nanmax(np.abs(all_data[kappa]['grid_V']))
            ])
    global_grid_max = max(grid_max_vals) if grid_max_vals else 1.0

    if horizontal:
        # Horizontal layout: n_kappas rows × 11 columns
        fig = plt.figure(figsize=(33, 3 * n_kappas))
        gs_base = fig.add_gridspec(n_kappas, 11, hspace=0.25, wspace=0.35)

        # Create a transposed gridspec wrapper
        class TransposedGridSpec:
            def __init__(self, gs):
                self._gs = gs
            def __getitem__(self, key):
                # Swap row and col: gs[row, col] becomes gs_base[col, row]
                if isinstance(key, tuple) and len(key) == 2:
                    row, col = key
                    return self._gs[col, row]
                return self._gs[key]

        gs = TransposedGridSpec(gs_base)
    else:
        # Vertical layout: 11 rows × n_kappas columns
        fig = plt.figure(figsize=(3 * n_kappas, 33))
        gs = fig.add_gridspec(11, n_kappas, hspace=0.35, wspace=0.25)

    def get_title(kappa):
        return f'κ={kappa:+.1f}'

    # Now use the same plotting code for both layouts (gridspec handles transposition)
    # Row 0: Behavioral heatmaps
    _plot_row_behavior(fig, gs, 0, kappa_values, all_data, get_title)
    # Row 1: Predicted values
    _plot_row_values(fig, gs, 1, kappa_values, all_data)
    # Row 2: Policy regression
    _plot_row_regression(fig, gs, 2, kappa_values, all_data, 'policy', 'Policy\nβEV')
    # Row 3: Value regression
    _plot_row_regression(fig, gs, 3, kappa_values, all_data, 'value', 'Value\nβEV')
    # Row 4: Policy output weights
    _plot_row_weights(fig, gs, 4, kappa_values, all_data, baseline_data,
                      'policy', 'red', policy_lim, 'Policy\nOutput Weight',
                      'Policy Output Weight (κ=0)')
    # Row 5: Value output weights
    _plot_row_weights(fig, gs, 5, kappa_values, all_data, baseline_data,
                      'value', 'blue', value_lim, 'Value\nOutput Weight',
                      'Value Output Weight (κ=0)')
    # Row 6: Policy β vs weights
    _plot_row_beta_vs_weights(fig, gs, 6, kappa_values, all_data, 'policy', 'red', policy_lim)
    # Row 7: Value β vs weights
    _plot_row_beta_vs_weights(fig, gs, 7, kappa_values, all_data, 'value', 'blue', value_lim)
    # Rows 8-10: Policy half-split contributions and total logits.
    _plot_row_policy_grids(fig, gs, 8, kappa_values, all_data, 'grid_D1', labels['half1_row'], global_grid_max)
    _plot_row_policy_grids(fig, gs, 9, kappa_values, all_data, 'grid_D2', labels['half2_row'], global_grid_max)
    _plot_row_policy_grids(fig, gs, 10, kappa_values, all_data, 'grid_V', labels['total_row'], global_grid_max)

    plt.tight_layout(rect=[0, 0, 1, 1])

    if plot_title == 'Neuromodulation Comparison':
        savefile = os.path.join(figspath, 'mega_comparison_neuromodulation.png')
    elif plot_title == 'Single Kappa Condition':
        savefile = os.path.join(figspath, 'kappa_single_condition.png')
    else:
        savefile = os.path.join(figspath, 'mega_comparison_all_kappas.png')

    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"\nSaved mega-plot to {savefile}\n")
    plt.close()

def plot_distribution_comparison(trialsfiles, modelfiles, figspath):
    """
    Create 8×4 comparison plot with different per-neuron kappa distributions.
    """
    print("\n" + "="*80)
    print("CREATING DISTRIBUTION COMPARISON PLOT: 8 ROWS × 4 COLUMNS")
    print("="*80)

    dist_names = ['gaussian', 'baseline', 'uniform', 'gaussian_neg0.2']
    dist_labels = {
        'gaussian': 'Gaussian\nN(0, 0.3²)',
        'baseline': 'Baseline\nκ=0',
        'uniform': 'Uniform\nU[-0.3, 0.3]',
        'gaussian_neg0.2': 'Gaussian\nN(-0.2, 0.3²)'
    }

    all_data, baseline_data = _load_comparison_data(dist_names, trialsfiles, modelfiles)

    if baseline_data is None:
        print("Error: No baseline data found!")
        return

    if len(all_data) < 3:
        print(f"Warning: Only found {len(all_data)} datasets, expected 3")

    policy_lim, value_lim = _compute_weight_limits(all_data, baseline_data, dist_names)

    fig = plt.figure(figsize=(16, 24))
    gs = fig.add_gridspec(8, 4, hspace=0.35, wspace=0.30)

    def get_title(key):
        return dist_labels.get(key, str(key))

    # Row 0: Behavioral heatmaps
    _plot_row_behavior(fig, gs, 0, dist_names, all_data, get_title)
    # Row 1: Predicted values
    _plot_row_values(fig, gs, 1, dist_names, all_data)
    # Row 2: Policy regression
    _plot_row_regression(fig, gs, 2, dist_names, all_data, 'policy', 'Policy\nβEV')
    # Row 3: Value regression
    _plot_row_regression(fig, gs, 3, dist_names, all_data, 'value', 'Value\nβEV')
    # Row 4: Policy output weights
    _plot_row_weights(fig, gs, 4, dist_names, all_data, baseline_data,
                      'policy', 'red', policy_lim, 'Policy\nOutput Weight',
                      'Policy Output Weight (κ=0)')
    # Row 5: Value output weights
    _plot_row_weights(fig, gs, 5, dist_names, all_data, baseline_data,
                      'value', 'blue', value_lim, 'Value\nOutput Weight',
                      'Value Output Weight (κ=0)')
    # Row 6: Policy β vs weights
    _plot_row_beta_vs_weights(fig, gs, 6, dist_names, all_data, 'policy', 'red', policy_lim)
    # Row 7: Value β vs weights
    _plot_row_beta_vs_weights(fig, gs, 7, dist_names, all_data, 'value', 'blue', value_lim)

    plt.tight_layout(rect=[0, 0, 0.98, 1])

    savefile = os.path.join(figspath, 'distribution_comparison_4col.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved distribution comparison plot to {savefile}\n")
    plt.close()

    print("\n" + "="*80)
    print("CREATING DISTRIBUTION COMPARISON PLOT: 8 ROWS × 4 COLUMNS")
    print("="*80)

def plot_rpe_signals(trialsfile, figspath, kappa=None, n_examples=10):
    """
    Plot Reward Prediction Error (RPE) signals - both objective and subjective.
    """
    print("\n" + "="*80)
    print(f"PLOTTING RPE SIGNALS")
    if kappa is not None:
        print(f"Kappa: {kappa:.2f}")
    print("="*80)

    td = load_trial_data(trialsfile)

    if td['format'] != 'rpe':
        print(f"Error: RPE signals not found in data (format={td['format']}, need 'rpe')")
        print("Make sure you're using a trials file generated with the updated run_trials code.")
        return

    rpe_obj = to_numpy(td['RPE_objective'])
    rpe_subj = to_numpy(td['RPE_subjective'])
    R_np = to_numpy(td['R'])
    action_indices = convert_actions(td['A'])
    trials = td['trials']

    M_np = to_numpy(td['M'])

    # Shape: [timesteps, trials]
    T, n_trials = rpe_obj.shape

    print(f"\nData shape: {rpe_obj.shape} (timesteps × trials)")
    print(f"Number of trials: {n_trials}")

    # Calculate statistics (only for valid timesteps)
    valid_mask = M_np > 0

    rpe_obj_valid = rpe_obj[valid_mask]
    rpe_subj_valid = rpe_subj[valid_mask]

    print(f"\nObjective RPE Statistics:")
    print(f"  Mean: {rpe_obj_valid.mean():.4f}")
    print(f"  Std:  {rpe_obj_valid.std():.4f}")
    print(f"  Min:  {rpe_obj_valid.min():.4f}")
    print(f"  Max:  {rpe_obj_valid.max():.4f}")

    print(f"\nSubjective RPE Statistics:")
    print(f"  Mean: {rpe_subj_valid.mean():.4f}")
    print(f"  Std:  {rpe_subj_valid.std():.4f}")
    print(f"  Min:  {rpe_subj_valid.min():.4f}")
    print(f"  Max:  {rpe_subj_valid.max():.4f}")

    # =========================
    # ESTIMATE KAPPA FROM DATA
    # =========================
    from scipy import stats

    # Separate positive and negative RPEs
    pos_mask = rpe_obj_valid > 0
    neg_mask = rpe_obj_valid < 0

    kappa_estimated = None
    eta_plus_estimated = None
    eta_minus_estimated = None
    r_squared_pos = None
    r_squared_neg = None

    print(f"\n{'='*80}")
    print(f"KAPPA ESTIMATION (Reverse Engineering)")
    print(f"{'='*80}")

    if pos_mask.sum() > 10:  # Need enough data points
        # Linear regression for positive RPEs: RPE_subj = η⁺ * RPE_obj
        slope_pos, _, r_value_pos, p_value_pos, _ = stats.linregress(
            rpe_obj_valid[pos_mask],
            rpe_subj_valid[pos_mask]
        )
        eta_plus_estimated = slope_pos
        r_squared_pos = r_value_pos ** 2
        kappa_from_gains = 1 - eta_plus_estimated

        print(f"\nPositive RPEs (Gains):")
        print(f"  η⁺ (estimated): {eta_plus_estimated:.4f}")
        print(f"  κ from gains:   {kappa_from_gains:.4f}")
        print(f"  R² fit:         {r_squared_pos:.4f}")
        print(f"  p-value:        {p_value_pos:.2e}")
        print(f"  Data points:    {pos_mask.sum()}")

    if neg_mask.sum() > 10:  # Need enough data points
        # Linear regression for negative RPEs: RPE_subj = η⁻ * RPE_obj
        slope_neg, _, r_value_neg, p_value_neg, _ = stats.linregress(
            rpe_obj_valid[neg_mask],
            rpe_subj_valid[neg_mask]
        )
        eta_minus_estimated = slope_neg
        r_squared_neg = r_value_neg ** 2
        kappa_from_losses = eta_minus_estimated - 1

        print(f"\nNegative RPEs (Losses):")
        print(f"  η⁻ (estimated): {eta_minus_estimated:.4f}")
        print(f"  κ from losses:  {kappa_from_losses:.4f}")
        print(f"  R² fit:         {r_squared_neg:.4f}")
        print(f"  p-value:        {p_value_neg:.2e}")
        print(f"  Data points:    {neg_mask.sum()}")

    # Combined estimate
    if eta_plus_estimated is not None and eta_minus_estimated is not None:
        kappa_from_gains = 1 - eta_plus_estimated
        kappa_from_losses = eta_minus_estimated - 1
        kappa_estimated = (kappa_from_gains + kappa_from_losses) / 2

        print(f"\n{'='*40}")
        print(f"FINAL ESTIMATE:")
        print(f"  κ (estimated):  {kappa_estimated:.4f}")
        if kappa is not None:
            print(f"  κ (actual):     {kappa:.4f}")
            print(f"  Difference:     {abs(kappa_estimated - kappa):.4f}")
            print(f"  Match quality:  {'✓ Good' if abs(kappa_estimated - kappa) < 0.1 else '⚠ Check model'}")
        print(f"{'='*40}")
    elif eta_plus_estimated is not None:
        kappa_estimated = 1 - eta_plus_estimated
        print(f"\n  κ (estimated from gains only): {kappa_estimated:.4f}")
    elif eta_minus_estimated is not None:
        kappa_estimated = eta_minus_estimated - 1
        print(f"\n  κ (estimated from losses only): {kappa_estimated:.4f}")

    # =========================
    # FIGURE 2: Example Trials Over Time
    # =========================
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples), sharex=True)

    if n_examples == 1:
        axes = [axes]

    # Select diverse trials with different reward outcomes
    trial_rewards = R_np.sum(axis=0)  # Total reward per trial

    # Find trials with different reward levels
    # Get unique reward values and select diverse examples
    unique_rewards = np.unique(trial_rewards)

    if len(unique_rewards) < n_examples:
        # Not enough diversity, use even spacing across all trials
        sorted_trials = np.argsort(trial_rewards)
        example_indices = np.linspace(0, n_trials-1, n_examples).astype(int)
        example_trials = sorted_trials[example_indices]
    else:
        # Select trials with diverse reward outcomes
        example_trials = []

        # Target reward levels across the range
        reward_min = trial_rewards.min()
        reward_max = trial_rewards.max()
        target_rewards = np.linspace(reward_min, reward_max, n_examples)

        for target in target_rewards:
            # Find trial closest to this target reward
            closest_idx = np.argmin(np.abs(trial_rewards - target))
            example_trials.append(closest_idx)

        example_trials = np.array(example_trials)

    for i, trial_idx in enumerate(example_trials):
        ax = axes[i]

        # Get trial data
        rpe_obj_trial = rpe_obj[:, trial_idx]
        rpe_subj_trial = rpe_subj[:, trial_idx]
        reward_trial = R_np[:, trial_idx]
        mask_trial = M_np[:, trial_idx]

        # Only plot valid timesteps
        valid_t = np.where(mask_trial > 0)[0]

        if len(valid_t) == 0:
            continue

        # Convert to milliseconds (assuming 10ms timesteps)
        time_ms = valid_t * 10

        # Plot RPEs
        ax.plot(time_ms, rpe_obj_trial[valid_t], 'b-', linewidth=2, label='Objective RPE', marker='o')
        ax.plot(time_ms, rpe_subj_trial[valid_t], 'purple', linewidth=2, label='Subjective RPE', 
                marker='s', linestyle='--')

        # Mark reward events
        reward_t = np.where((reward_trial != 0) & (mask_trial > 0))[0]
        for t in reward_t:
            t_ms = t * 10
            ax.axvline(t_ms, color='green' if reward_trial[t] > 0 else 'red', 
                      alpha=0.3, linestyle=':', linewidth=2)
            ax.text(t_ms, ax.get_ylim()[1]*0.9, f'R={reward_trial[t]:.1f}', 
                   rotation=90, va='top', fontsize=9)

        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.set_ylabel('RPE', fontsize=11)
        ax.set_title(f'Trial {trial_idx+1} (Total Reward: {trial_rewards[trial_idx]:.2f}, Duration: {len(valid_t)*10}ms)', 
                    fontsize=12, weight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add epoch markers
        ax.axvspan(0, 250, alpha=0.1, color='gray', label='Fixation')
        ax.axvspan(250, 500, alpha=0.1, color='yellow', label='Stimulus')

    axes[-1].set_xlabel('Time from trial start (ms)', fontsize=12)

    if kappa is not None:
        fig.suptitle(f'RPE Signals Over Time - Example Trials (κ={kappa:.2f})', 
                    fontsize=16, weight='bold', y=0.995)
    else:
        fig.suptitle('RPE Signals Over Time - Example Trials', 
                    fontsize=16, weight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    if kappa is not None:
        kappa_str = f"{kappa:+.2f}".replace('.', 'p').replace('-', 'neg').replace('+', 'pos')
        savefile = os.path.join(figspath, f'rpe_timecourse_kappa{kappa_str}.png')
    else:
        savefile = os.path.join(figspath, 'rpe_timecourse.png')

    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"Saved RPE timecourse plot to {savefile}")
    plt.close()

    print("\n" + "="*80)
    print("RPE plotting complete!")
    print("="*80 + "\n")

    return {
        'rpe_objective': rpe_obj,
        'rpe_subjective': rpe_subj,
        'kappa': kappa
    }

def plot_context_mega_comparison(contexts, trialsfiles, modelfile, figspath):
    """
    Create an 11-row mega-plot comparing behavior, neural activity, and 
    biological subjective values across context values.
    """
    print(f"\nCREATING MEGA-PLOT: 11 ROWS × {len(contexts)} CONTEXT VALUES\n")
    labels = policy_split_labels(model_uses_dopamine_split(modelfile))

    # =====================================================================
    # 🔍 DIAGNOSTIC BLOCK
    # =====================================================================
    print("\n" + "="*60)
    print("🔍 CONTEXT SENSITIVITY DIAGNOSTICS")
    print("="*60)

    try:
        from pyrl import utils
        model_data = utils.load(modelfile)
        policy_params = model_data.get('best_policy_params', {})
        Win_policy = policy_params.get('Win')
        Wout_policy = policy_params.get('Wout')

        if Win_policy is not None:
            Nin = Win_policy.shape[1]
            print(f"Policy Network Input Size (Nin): {Nin}")
            
            if Nin >= 8:
                context_weights = Win_policy[:, 7] 
                print(f"\nContext Input Weights (Win[:, 7]):")
                print(f"  Mean:    {np.mean(context_weights):.6f}")
                print(f"  Std Dev: {np.std(context_weights):.6f}")
                print(f"  Max Abs: {np.max(np.abs(context_weights)):.6f}")
                
                if np.std(context_weights) < 1e-4:
                    print("  ⚠️ WARNING: Context input weights are effectively ZERO.")
            else:
                print("  ⚠️ WARNING: Nin is less than 8. Context is NOT connected to the GRU!")

        if Wout_policy is not None:
            half_N = Wout_policy.shape[0] // 2
            w_d1 = Wout_policy[:half_N]
            w_d2 = Wout_policy[half_N:]
            print("\nPolicy Output Weight Balance:")
            print(f"  {labels['half1_short']} abs mean/std/norm: {np.mean(np.abs(w_d1)):.6f} / {np.std(w_d1):.6f} / {np.linalg.norm(w_d1):.6f}")
            print(f"  {labels['half2_short']} abs mean/std/norm: {np.mean(np.abs(w_d2)):.6f} / {np.std(w_d2):.6f} / {np.linalg.norm(w_d2):.6f}")
            print(f"  {labels['half2_short']}/{labels['half1_short']} norm ratio:     {np.linalg.norm(w_d2) / max(np.linalg.norm(w_d1), 1e-12):.6f}")

        for pname, label in [
            ('dopamine_sensitivity', 'Dopamine Sensitivity'),
            ('dopamine_bias', 'Dopamine Bias')
        ]:
            arr = policy_params.get(pname)
            if arr is not None:
                half_N = arr.shape[0] // 2
                d1 = arr[:half_N]
                d2 = arr[half_N:]
                print(f"\n{label} Balance:")
                print(f"  {labels['half1_short']} mean/std/absmean: {np.mean(d1):+.6f} / {np.std(d1):.6f} / {np.mean(np.abs(d1)):.6f}")
                print(f"  {labels['half2_short']} mean/std/absmean: {np.mean(d2):+.6f} / {np.std(d2):.6f} / {np.mean(np.abs(d2)):.6f}")
                print(f"  {labels['half2_short']}/{labels['half1_short']} absmean ratio: {np.mean(np.abs(d2)) / max(np.mean(np.abs(d1)), 1e-12):.6f}")
    except Exception as e:
        print(f"Could not load model weights for diagnostics: {e}")

    modelfiles = {ctx: modelfile for ctx in contexts}
    all_data, baseline_data = _load_comparison_data(contexts, trialsfiles, modelfiles)

    if -1.0 in all_data and 1.0 in all_data:
        ch_neg = all_data[-1.0]['choices']
        ch_pos = all_data[1.0]['choices']
        
        min_len = min(len(ch_neg), len(ch_pos))
        diff_choices = np.sum(ch_neg[:min_len] != ch_pos[:min_len])
        
        print(f"\nBehavioral Difference (ctx = -1.0 vs ctx = +1.0):")
        print(f"  Total choices compared: {min_len}")
        print(f"  Differing choices:      {diff_choices}")
        
        vg_neg = all_data[-1.0]['value_grid']
        vg_pos = all_data[1.0]['value_grid']
        vg_diff = np.nanmean(np.abs(vg_neg - vg_pos))
        print(f"\nValue Grid Difference (ctx = -1.0 vs ctx = +1.0):")
        print(f"  Mean absolute difference: {vg_diff:.6f}")

    print(f"\nPolicy {labels['half1_short']}/{labels['half2_short']} Pull Balance by Context:")
    print(f"  ctx      |{labels['half1_short']}|      |{labels['half2_short']}|      ratio    half1std    half2std")
    for ctx in contexts:
        stats = all_data.get(ctx, {}).get('pull_stats')
        if stats is None:
            continue
        ratio = stats['d2_choice_abs_mean'] / max(stats['d1_choice_abs_mean'], 1e-12)
        print(
            f"  {ctx:+.2f}   "
            f"{stats['d1_choice_abs_mean']:.6f}  "
            f"{stats['d2_choice_abs_mean']:.6f}  "
            f"{ratio:.4f}  "
            f"{stats['d1_std']:.6f}  "
            f"{stats['d2_std']:.6f}"
        )

    print("="*60 + "\n")
    # =====================================================================

    if baseline_data is None and 0.0 in all_data:
        baseline_data = all_data[0.0]

    policy_lim, value_lim = _compute_weight_limits(all_data, baseline_data, contexts)

    # Calculate global max for the policy half-split and total grids so colors are comparable
    grid_max_vals = []
    for ctx in contexts:
        if ctx in all_data and all_data[ctx].get('grid_V') is not None:
            grid_max_vals.extend([
                np.nanmax(np.abs(all_data[ctx]['grid_D1'])),
                np.nanmax(np.abs(all_data[ctx]['grid_D2'])),
                np.nanmax(np.abs(all_data[ctx]['grid_V']))
            ])
    global_grid_max = max(grid_max_vals) if grid_max_vals else 1.0

    import matplotlib.pyplot as plt
    n_contexts = len(contexts)
    fig = plt.figure(figsize=(3 * n_contexts, 33))
    gs = fig.add_gridspec(11, n_contexts, hspace=0.35, wspace=0.25)

    def get_title(ctx):
        return f'Ctx = {ctx:+.1f}'

    _plot_row_behavior(fig, gs, 0, contexts, all_data, get_title)
    _plot_row_values(fig, gs, 1, contexts, all_data)
    _plot_row_regression(fig, gs, 2, contexts, all_data, 'policy', 'Policy\nβEV')
    _plot_row_regression(fig, gs, 3, contexts, all_data, 'value', 'Value\nβEV')
    _plot_row_regression_keys(fig, gs, 4, contexts, all_data,
                              'beta_hh_ll_policy_d1', 'beta_ev_policy_d1', labels['half1_beta'])
    _plot_row_regression_keys(fig, gs, 5, contexts, all_data,
                              'beta_hh_ll_policy_d2', 'beta_ev_policy_d2', labels['half2_beta'])
    _plot_row_beta_vs_weights_keys(fig, gs, 6, contexts, all_data,
                                   'Wout_policy_d1', 'beta_hh_ll_policy_d1', 'red', policy_lim,
                                   labels['half1_weight'])
    _plot_row_beta_vs_weights_keys(fig, gs, 7, contexts, all_data,
                                   'Wout_policy_d2', 'beta_hh_ll_policy_d2', 'red', policy_lim,
                                   labels['half2_weight'])

    _plot_row_policy_grids(fig, gs, 8, contexts, all_data, 'grid_D1', labels['half1_row'], global_grid_max)
    _plot_row_policy_grids(fig, gs, 9, contexts, all_data, 'grid_D2', labels['half2_row'], global_grid_max)
    _plot_row_policy_grids(fig, gs, 10, contexts, all_data, 'grid_V', labels['total_row'], global_grid_max)

    plt.tight_layout(rect=[0, 0, 1, 1])

    import os
    savefile = os.path.join(figspath, 'mega_comparison_context_sweep.png')
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"\nSaved 11-row context mega-plot to {savefile}\n")
    plt.close()

def _plot_row_policy_grids(fig, gs, row, col_keys, all_data, grid_key, ylabel, global_max):
    """Plot Row: Policy Internal Value Grids (D1, D2, V) with swapped axes."""
    axes = []
    im = None
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        axes.append(ax)
        if key not in all_data or all_data[key].get(grid_key) is None:
            ax.set_xticks([]); ax.set_yticks([])
            continue
            
        data = all_data[key][grid_key]
        im = ax.imshow(data, cmap='RdBu', origin='lower', vmin=-global_max, vmax=global_max, aspect='auto')
        
        ax.set_yticks(range(5))
        ax.set_xticks(range(5))
        
        # Y-Axis (Expected Value)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_yticklabels(['0.4', '', '0.7', '', '1.0'], fontsize=8)
        else:
            ax.set_yticklabels([])
            
        # X-Axis (Probability) - NOW APPLIED TO ALL ROWS
        if idx == 0:
            ax.set_xlabel('Reward Probability', fontsize=9)
        ax.set_xticklabels(['10%', '30%', '50%', '70%', '90%'], fontsize=8, rotation=45)
            
    if im is not None:
        cax = _row_colorbar_axis(fig, axes[-1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Logits (Preference)', fontsize=9, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
    return axes


# ============================================================================
# OPTOGENETIC VTA STIMULATION PLOTTING FUNCTIONS
# ============================================================================

def plot_opto_mega_comparison(opto_offsets, trialsfiles, modelfile, figspath):
    """
    Optostimulation megaplot - reuses context megaplot with opto labels.

    Parameters
    ----------
    opto_offsets : list
        List of optostimulation offset values
    trialsfiles : dict
        Mapping from opto_offset → trial data filepath
    modelfile : str
        Path to model file
    figspath : str
        Directory to save figures
    """
    # Call context megaplot function with opto data
    plot_context_mega_comparison(opto_offsets, trialsfiles, modelfile, figspath)

    # Rename the saved file to reflect optostimulation
    old_file = os.path.join(figspath, 'context_mega_comparison.png')
    new_file = os.path.join(figspath, 'opto_mega_comparison.png')

    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"  Saved: {new_file}")


def plot_opto_choice_probability_curves(opto_offsets, trialsfiles, figspath):
    """
    Plot choice probability curves across optostimulation levels.

    Parameters
    ----------
    opto_offsets : list
        List of optostimulation offset values
    trialsfiles : dict
        Mapping from opto_offset → trial data filepath
    figspath : str
        Directory to save figures
    """
    # Call context choice curve functions with opto data.
    plot_context_choice_probability_curves(opto_offsets, trialsfiles, figspath)
    plot_context_choice_probability_mega(opto_offsets, trialsfiles, figspath)

    old_file = os.path.join(figspath, 'context_choice_probability_curves_mega.png')
    new_file = os.path.join(figspath, 'opto_choice_probability_mega.png')

    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"  Saved: {new_file}")
