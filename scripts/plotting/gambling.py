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
from matplotlib.colors import LinearSegmentedColormap

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
    """
    Load and unpack trial data into a consistent dict.

    Returns dict with keys: trials, A, R, M, and optionally
    Z_b, r_policy, r_value, RPE_objective, RPE_subjective.
    Also includes 'format': 'behavior', 'activity', or 'rpe'.
    """
    data = utils.load(trialsfile)
    result = {}

    if len(data) == 5:
        result['trials'], result['A'], result['R'], result['M'], _ = data
        result['format'] = 'behavior'
    elif len(data) == 10:
        (result['trials'], _, _, result['Z_b'], result['A'], result['R'],
         result['M'], _, result['r_policy'], result['r_value']) = data
        result['format'] = 'activity'
    elif len(data) >= 12:
        (result['trials'], _, _, result['Z_b'], result['A'], result['R'],
         result['M'], _, result['r_policy'], result['r_value']) = data[:10]
        result['RPE_objective'] = data[10]
        result['RPE_subjective'] = data[11]
        result['format'] = 'rpe'
    else:
        raise ValueError(f"Unexpected data format with {len(data)} elements")

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
    """Compute 5×5 predicted value grid from chosen options."""
    value_grid = np.zeros((5, 5))
    count_grid = np.zeros((5, 5))

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


def load_model_weights(modelfile):
    """Load output weights from a model file. Returns (Wout_policy, Wout_value)."""
    model_data = utils.load(modelfile)
    policy_params = model_data.get('best_policy_params', {})
    baseline_params = model_data.get('best_baseline_params', {})

    Wout_policy = to_numpy(policy_params.get('Wout', None))
    Wout_value = to_numpy(baseline_params.get('Wout', None))

    return Wout_policy, Wout_value


def format_kappa_str(kappa):
    """Format kappa value to a filename-safe string."""
    return f"{kappa:+.1f}".replace('.', 'p').replace('-', 'neg').replace('+', 'pos')


def compute_theoretical_evs():
    """
    Compute theoretical expected values for all 25 gambling options.

    Returns ev_grid : ndarray (5, 5) with rows=probability, cols=magnitude.
    """
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    magnitudes = np.linspace(0.4, 2.0, 5)
    return np.outer(probs, magnitudes)


def create_extended_value_colormap():
    """Custom colormap with viridis core and extended range for outlier values."""
    viridis = plt.cm.viridis

    colors = [
        '#000000', '#1f1f1f',
        viridis(0.0), viridis(0.5), viridis(1.0),
        '#ff8c00', '#ff0000',
    ]
    positions = [0.0, 0.035, 0.035, 0.07, 0.105, 0.55, 1.0]

    return LinearSegmentedColormap.from_list(
        'value_extended', list(zip(positions, colors))
    )

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

        ch = extract_choices(td['trials'], action_indices, delta_hh_lls, delta_evs)

        value_grid = compute_value_grid(td['trials'], action_indices, Z_b_np)

        all_data[key] = {
            'choices': ch['choices'], 'delta_hh_lls': ch['delta_hh_lls'],
            'delta_evs': ch['delta_evs'], 'value_grid': value_grid,
            'beta_hh_ll_policy': beta_hh_ll_policy, 'beta_ev_policy': beta_ev_policy,
            'beta_hh_ll_value': beta_hh_ll_value, 'beta_ev_value': beta_ev_value,
            'Wout_policy': Wout_policy, 'Wout_value': Wout_value
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
        pos = axes[-1].get_position()
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('P(Right)', fontsize=9, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
    return axes


def _plot_row_values(fig, gs, row, col_keys, all_data):
    """Plot Row: Predicted value heatmaps."""
    axes = []
    im = None
    value_cmap = create_extended_value_colormap()
    for idx, key in enumerate(col_keys):
        ax = fig.add_subplot(gs[row, idx])
        axes.append(ax)
        if key not in all_data:
            ax.set_xticks([]); ax.set_yticks([])
            continue
        im = ax.imshow(all_data[key]['value_grid'].T, cmap=value_cmap,
                       aspect='auto', origin='lower', vmin=0.0, vmax=10.0)
        prob_labels = ['10', '30', '50', '70', '90']
        ax.set_yticks(range(5)); ax.set_xticks(range(5))
        ax.set_yticklabels(prob_labels, fontsize=8)
        ax.set_xticklabels(prob_labels, fontsize=8)
        if idx == 0:
            ax.set_ylabel('EV', fontsize=10); ax.set_xlabel('HH-LL(%)', fontsize=9)
    if im is not None:
        pos = axes[-1].get_position()
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Predicted\nValue', fontsize=9, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_ticks([0.0, 0.4, 1.0, 5.0, 10.0])
        cbar.set_ticklabels(['0.0', '0.4', '1.0', '5.0', '10.0'])
    return axes


def _plot_row_regression(fig, gs, row, col_keys, all_data, network, ylabel):
    """Plot Row: Regression scatter (β_HH-LL vs β_EV)."""
    axes = []
    sc = None
    beta_key_hh = f'beta_hh_ll_{network}'
    beta_key_ev = f'beta_ev_{network}'
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
        pos = axes[-1].get_position()
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.height])
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label('Neuron\n(sorted by βEV)', fontsize=9, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=8)
        n = len(all_data[col_keys[-1] if col_keys[-1] in all_data else
                         next(k for k in reversed(col_keys) if k in all_data)][beta_key_ev])
        cbar.set_ticks([0, (n-1)/2, n-1])
        cbar.set_ticklabels(['1', '50', '100'])
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
            ax.set_ylabel(f'{network.capitalize()}\nOutput Weight', fontsize=10)
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
            trials_per_condition = 2  # 2 trials per condition = 625*2 = 1250 trials

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
        plot_heatmap(trialsfile, config['figspath'])

    elif action == 'mega-comparison':
        # Define all kappa values
        kappas = [
            -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0.0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
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
            print(f"   Example: python3 scripts/training/train.py tasks/gambling.py --suffix neg0p8 run scripts/plotting/gambling.py trials-a 5")
            return

        # Extract kappa values for which we found files (in order)
        kappa_values = [k for k in kappas if k in trialsfiles]

        print(f"\n🎨 Generating mega-comparison plot with {len(kappa_values)} models...")
        plot_mega_comparison(kappa_values, trialsfiles, modelfiles, config['figspath'])

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
            n_examples = 10

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

                runtools.run('trials-b', trials, pg, kappa_trialspath, dt_save=config['dt-save'])

                # Store path
                trialsfiles[kappa] = runtools.behaviorfile(kappa_trialspath)

            # Create comparison plots
            print(f"\n{'='*60}")
            print("Generating comparison plots...")
            print('='*60)

            plot_kappa_comparison(list(kappa_values), trialsfiles, config['figspath'])
            plot_kappa_summary(list(kappa_values), trialsfiles, config['figspath'])

            print("\nKappa sweep complete!")

        except Exception as e:
            print(f"Error in kappa-sweep: {e}")
            import traceback
            traceback.print_exc()
            print("\nUsage: kappa-sweep [kappa_min] [kappa_max] [n_kappas]")
            print("Example: kappa-sweep -0.8 0.8 9")

    else:
        print(f"Unrecognized action: {action}")
        print("Available actions:")
        print("  trials-b           - Generate trial data (behavior only)")
        print("  trials-a           - Generate trial data with neural activity")
        print("  behavior           - Plot behavioral heatmap")
        print("  neural-analysis    - Complete neural activity analysis (requires trials-a)")
        print("  kappa-comparison   - Compare multiple kappa values")
        print("  kappa-sweep        - Automated sweep across kappa values")


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

def plot_mega_comparison(kappa_values, trialsfiles, modelfiles, figspath, plot_title=None):
    """
    Create 8×9 mega-plot comparing all kappa values.
    """
    print("\nCREATING MEGA-PLOT: 8 ROWS × 9 KAPPA VALUES\n")

    expected_kappas = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    n_kappas = len(expected_kappas)

    all_data, baseline_data = _load_comparison_data(expected_kappas, trialsfiles, modelfiles)

    if baseline_data is None:
        print("Error: No baseline (κ=0) data found!")
        return

    policy_lim, value_lim = _compute_weight_limits(all_data, baseline_data, expected_kappas)

    fig = plt.figure(figsize=(27, 24))
    gs = fig.add_gridspec(8, n_kappas, hspace=0.35, wspace=0.25)

    def get_title(kappa):
        return f'κ={kappa:+.1f}'

    # Row 0: Behavioral heatmaps
    _plot_row_behavior(fig, gs, 0, expected_kappas, all_data, get_title)
    # Row 1: Predicted values
    _plot_row_values(fig, gs, 1, expected_kappas, all_data)
    # Row 2: Policy regression
    _plot_row_regression(fig, gs, 2, expected_kappas, all_data, 'policy', 'Policy\nβEV')
    # Row 3: Value regression
    _plot_row_regression(fig, gs, 3, expected_kappas, all_data, 'value', 'Value\nβEV')
    # Row 4: Policy output weights
    _plot_row_weights(fig, gs, 4, expected_kappas, all_data, baseline_data,
                      'policy', 'red', policy_lim, 'Policy\nOutput Weight',
                      'Policy Output Weight (κ=0)')
    # Row 5: Value output weights
    _plot_row_weights(fig, gs, 5, expected_kappas, all_data, baseline_data,
                      'value', 'blue', value_lim, 'Value\nOutput Weight',
                      'Value Output Weight (κ=0)')
    # Row 6: Policy β vs weights
    _plot_row_beta_vs_weights(fig, gs, 6, expected_kappas, all_data, 'policy', 'red', policy_lim)
    # Row 7: Value β vs weights
    _plot_row_beta_vs_weights(fig, gs, 7, expected_kappas, all_data, 'value', 'blue', value_lim)

    plt.tight_layout(rect=[0, 0, 1, 1])

    if plot_title == 'Neuromodulation Comparison':
        savefile = os.path.join(figspath, 'mega_comparison_neuromodulation.png')
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
