#!/usr/bin/env python3
"""Plot risk preference sensitivity across context/VTA levels."""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

from pyrl import utils


CONDITIONS = [
    {
        'name': 'Vanilla + context',
        'token': 'vanilla_context',
        'trials_dir': (
            'data_progression3/d1d2_plasticity_opal04/trials/'
            'gambling_d1d2_plasticity_pos_reg_opal04'
        ),
        'pattern': 'trials_activity_ctx*.pkl',
    },
    {
        'name': 'Vanilla + fake VTA',
        'token': 'vanilla_fake_vta',
        'trials_dir': (
            'data_progression3/d1d2_plasticity_opal04_fake_vta/trials/'
            'gambling_d1d2_plasticity_pos_reg_opal04_fake_vta'
        ),
        'pattern': 'trials_activity_opto*.pkl',
    },
    {
        'name': 'Natural RPE + fake VTA',
        'token': 'natural_fake_vta',
        'trials_dir': (
            'data_progression3/d1d2_plasticity_opal04_rpe_natural/trials/'
            'gambling_d1d2_plasticity_pos_reg_opal04_rpe_natural'
        ),
        'pattern': 'trials_activity_opto*.pkl',
    },
]


def to_numpy(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_trial_data(trialsfile):
    data = utils.load(trialsfile)
    if len(data) == 5:
        trials, A, _, _, perf = data
    elif len(data) >= 10:
        trials, _, _, _, A, _, _, perf = data[:8]
    else:
        raise ValueError(f'Unrecognized trial file format: {trialsfile}')
    return trials, A, perf


def action_indices(A):
    A_np = to_numpy(A)
    if A_np.ndim == 3:
        return np.argmax(A_np, axis=2)
    return A_np


def parse_level(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r'(?:ctx|opto)(neg|pos)([0-9]+p[0-9]+)', stem)
    if not match:
        raise ValueError(f'Could not parse level from {path}')
    sign = -1.0 if match.group(1) == 'neg' else 1.0
    return sign * float(match.group(2).replace('p', '.'))


def first_choices(A):
    actions = action_indices(A)
    choices = np.full(actions.shape[1], -1, dtype=int)
    for trial_i in range(actions.shape[1]):
        choice_times = np.where((actions[:, trial_i] == 1) | (actions[:, trial_i] == 2))[0]
        if choice_times.size:
            choices[trial_i] = actions[choice_times[0], trial_i]
    return choices


def risk_summary(trials, A, perf):
    choices = first_choices(A)
    perf_decisions = np.asarray(getattr(perf, 'decisions', []), dtype=bool)
    if perf_decisions.size != len(trials):
        perf_decisions = choices != -1
    matched_ev = []
    risky_choice = []
    decisions = []

    for i, trial in enumerate(trials):
        choice = choices[i]
        has_decision = bool(perf_decisions[i]) and choice in (1, 2)
        decisions.append(has_decision)

        prob_l = float(trial['prob_l'])
        prob_r = float(trial['prob_r'])
        ev_l = prob_l * float(trial['size_l'])
        ev_r = prob_r * float(trial['size_r'])
        is_matched_ev = abs(ev_l - ev_r) < 0.01 and prob_l != prob_r
        matched_ev.append(is_matched_ev)

        if not (has_decision and is_matched_ev):
            risky_choice.append(np.nan)
            continue

        chose_right = choice == 2
        right_is_risky = prob_r < prob_l
        risky_choice.append(chose_right if right_is_risky else not chose_right)

    decisions = np.asarray(decisions, dtype=bool)
    matched_ev = np.asarray(matched_ev, dtype=bool)
    risky_choice = np.asarray(risky_choice, dtype=float)
    mask = decisions & matched_ev
    riskiness = float(np.nanmean(risky_choice[mask])) if np.any(mask) else np.nan
    return {
        'riskiness': riskiness,
        'risk_averseness': 1.0 - riskiness if np.isfinite(riskiness) else np.nan,
        'completion': float(np.mean(decisions)),
        'n_matched_ev_decisions': int(np.sum(mask)),
    }


def collect_condition(condition):
    pattern = os.path.join(REPO_ROOT, condition['trials_dir'], condition['pattern'])
    files = sorted(glob.glob(pattern), key=parse_level)
    rows = []
    seen = set()
    for trialsfile in files:
        level = round(parse_level(trialsfile), 3)
        if abs(level) > 0.9001 or level in seen:
            continue
        seen.add(level)
        trials, A, perf = load_trial_data(trialsfile)
        stats = risk_summary(trials, A, perf)
        rows.append({'level': level, **stats, 'file': trialsfile})
    if not rows:
        raise SystemExit(f"No trial files found for {condition['name']}: {pattern}")
    return rows


def linear_r2(x, y):
    fit = np.polyfit(x, y, 1)
    pred = np.polyval(fit, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return fit, 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def write_csv(all_rows, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'condition', 'level', 'riskiness', 'risk_averseness',
                'completion', 'n_matched_ev_decisions', 'file',
            ],
        )
        writer.writeheader()
        for condition, rows in all_rows.items():
            for row in rows:
                writer.writerow({'condition': condition, **row})


def plot(all_rows, outfile):
    colors = {
        'Vanilla + context': '#1b9e77',
        'Vanilla + fake VTA': '#d95f02',
        'Natural RPE + fake VTA': '#7570b3',
    }
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)

    for condition, rows in all_rows.items():
        levels = np.array([r['level'] for r in rows], dtype=float)
        riskiness = np.array([r['riskiness'] for r in rows], dtype=float)
        order = np.argsort(levels)
        levels = levels[order]
        riskiness = riskiness[order]
        color = colors.get(condition, '#333333')
        fit, r2 = linear_r2(levels, riskiness)
        slope = np.diff(riskiness) / np.diff(levels)
        midpoints = (levels[:-1] + levels[1:]) / 2.0

        axes[0].plot(levels, riskiness, '-o', color=color, linewidth=2.2,
                     markersize=4.8, label=f'{condition} (linear R2={r2:.2f})')
        axes[0].plot(levels, np.polyval(fit, levels), '--', color=color,
                     linewidth=1.2, alpha=0.55)
        axes[1].plot(midpoints, slope, '-o', color=color, linewidth=2.0,
                     markersize=4.0, label=condition)

    axes[0].axhline(0.5, color='#222222', linewidth=1.0, linestyle=':', alpha=0.7)
    axes[0].set_ylabel('Riskiness\nP(risky choice | matched EV)', fontsize=12)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title('Risk preference across context / VTA level', fontsize=15, pad=10)

    axes[1].axhline(0.0, color='#222222', linewidth=1.0, linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Context / VTA level', fontsize=12)
    axes[1].set_ylabel('Local change\nΔriskiness / Δlevel', fontsize=12)
    axes[1].set_xticks(np.round(np.arange(-0.9, 1.0, 0.1), 1))

    for ax in axes:
        ax.set_xlim(-0.95, 0.95)
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10, width=1.0, length=4)

    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved risk sensitivity plot: {outfile}')


def smooth_curve(x, y, dense_x, smoothing=0.035):
    try:
        from scipy.interpolate import UnivariateSpline

        order = min(3, len(x) - 1)
        spline = UnivariateSpline(x, y, k=order, s=smoothing)
        return spline(dense_x)
    except Exception:
        degree = min(4, len(x) - 1)
        fit = np.polyfit(x, y, degree)
        return np.polyval(fit, dense_x)


def plot_smoothed(all_rows, outfile):
    colors = {
        'Vanilla + context': '#1b9e77',
        'Vanilla + fake VTA': '#d95f02',
        'Natural RPE + fake VTA': '#7570b3',
    }
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    dense_levels = np.linspace(-0.9, 0.9, 400)
    dense_midpoints = np.linspace(-0.85, 0.85, 360)

    for condition, rows in all_rows.items():
        levels = np.array([r['level'] for r in rows], dtype=float)
        riskiness = np.array([r['riskiness'] for r in rows], dtype=float)
        order = np.argsort(levels)
        levels = levels[order]
        riskiness = riskiness[order]
        color = colors.get(condition, '#333333')
        fit, r2 = linear_r2(levels, riskiness)

        risk_smooth = np.clip(smooth_curve(levels, riskiness, dense_levels), 0.0, 1.0)
        slope = np.diff(riskiness) / np.diff(levels)
        midpoints = (levels[:-1] + levels[1:]) / 2.0
        slope_smooth = smooth_curve(midpoints, slope, dense_midpoints, smoothing=0.12)

        axes[0].plot(dense_levels, risk_smooth, '-', color=color, linewidth=2.6,
                     label=f'{condition} (linear R2={r2:.2f})')
        axes[0].plot(levels, riskiness, 'o', color=color, markerfacecolor='white',
                     markeredgewidth=1.1, markersize=4.8, alpha=0.7)
        axes[1].plot(dense_midpoints, slope_smooth, '-', color=color, linewidth=2.4,
                     label=condition)
        axes[1].plot(midpoints, slope, 'o', color=color, markerfacecolor='white',
                     markeredgewidth=1.0, markersize=4.0, alpha=0.55)

    axes[0].axhline(0.5, color='#222222', linewidth=1.0, linestyle=':', alpha=0.7)
    axes[0].set_ylabel('Riskiness\nP(risky choice | matched EV)', fontsize=12)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title('Smoothed risk preference across context / VTA level', fontsize=15, pad=10)

    axes[1].axhline(0.0, color='#222222', linewidth=1.0, linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Context / VTA level', fontsize=12)
    axes[1].set_ylabel('Smoothed local change\nΔriskiness / Δlevel', fontsize=12)
    axes[1].set_xticks(np.round(np.arange(-0.9, 1.0, 0.1), 1))

    for ax in axes:
        ax.set_xlim(-0.95, 0.95)
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10, width=1.0, length=4)

    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved smoothed risk sensitivity plot: {outfile}')


def range_capture(rows, max_ctx):
    by_level = {round(float(row['level']), 1): float(row['riskiness']) for row in rows}
    full = by_level[0.9] - by_level[-0.9]
    captured = by_level[round(max_ctx, 1)] - by_level[round(-max_ctx, 1)]
    return captured / full if full else np.nan


def plot_ctx_range_optimization(all_rows, outfile, selected_max_ctx=0.6):
    colors = {
        'Vanilla + context': '#1b9e77',
        'Vanilla + fake VTA': '#d95f02',
        'Natural RPE + fake VTA': '#7570b3',
    }
    max_ctxs = np.round(np.arange(0.1, 1.0, 0.1), 1)
    capture_by_condition = {}
    for condition, rows in all_rows.items():
        capture_by_condition[condition] = np.array([
            range_capture(rows, max_ctx) for max_ctx in max_ctxs
        ])

    mean_capture = np.nanmean(np.vstack(list(capture_by_condition.values())), axis=0)
    marginal_gain = np.diff(mean_capture, prepend=0.0)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)
    for condition, capture in capture_by_condition.items():
        color = colors.get(condition, '#333333')
        axes[0].plot(max_ctxs, capture, 'o', color=color, markerfacecolor='white',
                     markeredgewidth=1.1, markersize=4.8, alpha=0.65)
        dense_x = np.linspace(max_ctxs.min(), max_ctxs.max(), 300)
        dense_y = np.clip(smooth_curve(max_ctxs, capture, dense_x, smoothing=0.015), 0.0, 1.05)
        axes[0].plot(dense_x, dense_y, '-', color=color, linewidth=2.1, label=condition)

    axes[0].plot(max_ctxs, mean_capture, '-o', color='#222222', linewidth=2.4,
                 markersize=5.2, label='Mean across conditions')
    axes[0].axhline(0.90, color='#8d5f2f', linewidth=1.2, linestyle='--', alpha=0.85)
    axes[0].axvline(selected_max_ctx, color='#8d5f2f', linewidth=1.4, linestyle='--', alpha=0.85)
    axes[0].text(selected_max_ctx + 0.015, 0.08, f'selected max ctx = ±{selected_max_ctx:.1f}',
                 color='#6f451f', fontsize=10, rotation=90, va='bottom')
    axes[0].set_ylabel('Fraction of full ±0.9\nrisk modulation captured', fontsize=12)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title('Choosing a practical maximum ctx range', fontsize=15, pad=10)
    axes[0].legend(frameon=False, fontsize=8.5, loc='lower right')

    axes[1].bar(max_ctxs, marginal_gain, width=0.055, color='#6f8f72', alpha=0.72,
                edgecolor='#39563f', linewidth=0.6)
    axes[1].axvline(selected_max_ctx, color='#8d5f2f', linewidth=1.4, linestyle='--', alpha=0.85)
    axes[1].axhline(0.05, color='#8d5f2f', linewidth=1.0, linestyle=':', alpha=0.75)
    axes[1].set_ylabel('Added captured range\nfrom next 0.1 step', fontsize=12)
    axes[1].set_xlabel('Symmetric tested range: -max ctx to +max ctx', fontsize=12)
    axes[1].set_xticks(max_ctxs)

    for ax in axes:
        ax.set_xlim(0.05, 0.95)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=10, width=1.0, length=4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved ctx range optimization plot: {outfile}')


def main():
    outdir = os.path.join(REPO_ROOT, 'reports', 'assets')
    all_rows = {
        condition['name']: collect_condition(condition)
        for condition in CONDITIONS
    }
    write_csv(all_rows, os.path.join(outdir, 'risk_sensitivity_by_level.csv'))
    plot(all_rows, os.path.join(outdir, 'risk_sensitivity_by_level.png'))
    plot_smoothed(all_rows, os.path.join(outdir, 'risk_sensitivity_by_level_smoothed.png'))
    plot_ctx_range_optimization(
        all_rows,
        os.path.join(outdir, 'ctx_range_optimization.png'),
        selected_max_ctx=0.6,
    )

    for condition, rows in all_rows.items():
        levels = np.array([r['level'] for r in rows], dtype=float)
        riskiness = np.array([r['riskiness'] for r in rows], dtype=float)
        order = np.argsort(levels)
        fit, r2 = linear_r2(levels[order], riskiness[order])
        slopes = np.diff(riskiness[order]) / np.diff(levels[order])
        print(
            f"{condition}: risk {riskiness[order][0]:.3f} -> {riskiness[order][-1]:.3f}, "
            f"linear slope={fit[0]:.3f}, R2={r2:.3f}, "
            f"max local slope={np.max(slopes):.3f}, min local slope={np.min(slopes):.3f}"
        )


if __name__ == '__main__':
    main()
