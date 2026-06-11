#!/usr/bin/env python3
"""Fit prospect-theory parameters independently across context levels."""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.analysis.replicate_sasaki2024 import load_all_ctx_data  # noqa: E402


def _choice_arrays(trials, choices):
    p_l, p_r, r_l, r_r, y = [], [], [], [], []
    for i, tr in enumerate(trials):
        c = choices[i]
        if c not in (1, 2):
            continue
        p_l.append(float(tr['prob_l']))
        p_r.append(float(tr['prob_r']))
        r_l.append(float(tr['size_l']))
        r_r.append(float(tr['size_r']))
        y.append(1.0 if c == 2 else 0.0)
    return map(np.asarray, (p_l, p_r, r_l, r_r, y))


def _fit_fixed_gamma(trials, choices, fixed_gamma=1.0):
    """Fit alpha and beta with gamma fixed."""
    from scipy.optimize import minimize
    from scipy.special import expit

    p_l, p_r, r_l, r_r, y = _choice_arrays(trials, choices)
    if len(y) < 30:
        return None

    p_l = np.clip(p_l, 1e-9, 1 - 1e-9)
    p_r = np.clip(p_r, 1e-9, 1 - 1e-9)
    wl = np.exp(-((-np.log(p_l)) ** fixed_gamma))
    wr = np.exp(-((-np.log(p_r)) ** fixed_gamma))

    def neg_ll(params):
        alpha, beta = params
        if alpha <= 0.05 or beta <= 0:
            return 1e9
        vl = np.clip(r_l, 0, None) ** alpha
        vr = np.clip(r_r, 0, None) ** alpha
        pr = np.clip(expit(beta * (wr * vr - wl * vl)), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    best = None
    for a0, b0 in [(0.5, 4.0), (1.0, 2.0), (1.5, 2.0), (2.5, 1.0)]:
        res = minimize(
            neg_ll,
            [a0, b0],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5},
        )
        if best is None or res.fun < best.fun:
            best = res
    return None if best is None else (float(best.x[0]), float(best.x[1]), float(best.fun))


def _fit_fixed_alpha(trials, choices, fixed_alpha=1.0):
    """Fit gamma and beta with alpha fixed."""
    from scipy.optimize import minimize
    from scipy.special import expit

    p_l, p_r, r_l, r_r, y = _choice_arrays(trials, choices)
    if len(y) < 30:
        return None

    p_l = np.clip(p_l, 1e-9, 1 - 1e-9)
    p_r = np.clip(p_r, 1e-9, 1 - 1e-9)
    vl = np.clip(r_l, 0, None) ** fixed_alpha
    vr = np.clip(r_r, 0, None) ** fixed_alpha

    def neg_ll(params):
        gamma, beta = params
        if gamma <= 0.05 or gamma > 6 or beta <= 0:
            return 1e9
        wl = np.exp(-((-np.log(p_l)) ** gamma))
        wr = np.exp(-((-np.log(p_r)) ** gamma))
        pr = np.clip(expit(beta * (wr * vr - wl * vl)), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    best = None
    for g0, b0 in [(0.5, 4.0), (0.8, 3.0), (1.0, 2.0), (1.5, 2.0)]:
        res = minimize(
            neg_ll,
            [g0, b0],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5},
        )
        if best is None or res.fun < best.fun:
            best = res
    return None if best is None else (float(best.x[0]), float(best.x[1]), float(best.fun))


def _plot(rows, outpath, fixed_gamma, fixed_alpha):
    levels = []
    alpha_fixed_gamma = []
    gamma_fixed_alpha = []
    alpha_beta = []
    gamma_beta = []

    for row in rows:
        a_fit = _fit_fixed_gamma(row['trials'], row['choices'], fixed_gamma=fixed_gamma)
        g_fit = _fit_fixed_alpha(row['trials'], row['choices'], fixed_alpha=fixed_alpha)
        if a_fit is None or g_fit is None:
            continue
        levels.append(row['level'])
        alpha_fixed_gamma.append(a_fit[0])
        alpha_beta.append(a_fit[1])
        gamma_fixed_alpha.append(g_fit[0])
        gamma_beta.append(g_fit[1])

    levels = np.asarray(levels)
    order = np.argsort(levels)
    levels = levels[order]
    alpha_fixed_gamma = np.asarray(alpha_fixed_gamma)[order]
    gamma_fixed_alpha = np.asarray(gamma_fixed_alpha)[order]
    alpha_beta = np.asarray(alpha_beta)[order]
    gamma_beta = np.asarray(gamma_beta)[order]

    norm = plt.Normalize(levels.min(), levels.max())
    colors = plt.cm.coolwarm(norm(levels))

    ref_trials = rows[0]['trials']
    all_rewards = [float(tr['size_l']) for tr in ref_trials] + [
        float(tr['size_r']) for tr in ref_trials
    ]
    x_max = float(np.max(all_rewards)) * 1.05
    x_grid = np.linspace(0, x_max, 200)
    p_grid = np.linspace(0.01, 0.99, 200)

    fig = plt.figure(figsize=(17, 4.5))
    axes = [fig.add_axes([0.05 + i * 0.215, 0.12, 0.185, 0.76]) for i in range(4)]
    cax = fig.add_axes([0.935, 0.14, 0.013, 0.72])

    ax = axes[0]
    ax.plot(p_grid, p_grid, '--', color='#aaaaaa', lw=1.1, zorder=1)
    for level, gamma, color in zip(levels, gamma_fixed_alpha, colors):
        w = np.exp(-((-np.log(p_grid)) ** gamma))
        ax.plot(p_grid, w, color=color, lw=1.4, alpha=0.75, zorder=2)
    for target in [levels[0], levels[len(levels) // 2], levels[-1]]:
        idx = int(np.argmin(np.abs(levels - target)))
        gamma = gamma_fixed_alpha[idx]
        w = np.exp(-((-np.log(p_grid)) ** gamma))
        ax.plot(
            p_grid, w, color=colors[idx], lw=2.4, zorder=3,
            label=f'ctx={levels[idx]:+.1f}  gamma={gamma:.2f}'
        )
    ax.set_xlabel('probability')
    ax.set_ylabel('decision weight w(p)')
    ax.set_title(f'Probability weighting\n(alpha={fixed_alpha:g} fixed)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc='upper left')

    ax = axes[1]
    ax.plot(x_grid / x_max, x_grid / x_max, '--', color='#aaaaaa', lw=1.1, zorder=1)
    for level, alpha, color in zip(levels, alpha_fixed_gamma, colors):
        v = x_grid ** alpha
        ax.plot(x_grid / x_max, v / v.max(), color=color, lw=1.4, alpha=0.75, zorder=2)
    for target in [levels[0], levels[len(levels) // 2], levels[-1]]:
        idx = int(np.argmin(np.abs(levels - target)))
        alpha = alpha_fixed_gamma[idx]
        v = x_grid ** alpha
        ax.plot(
            x_grid / x_max, v / v.max(), color=colors[idx], lw=2.4, zorder=3,
            label=f'ctx={levels[idx]:+.1f}  alpha={alpha:.2f}'
        )
    ax.set_xlabel('normalised reward')
    ax.set_ylabel('utility v(x) (norm.)')
    ax.set_title(f'Utility function\n(gamma={fixed_gamma:g} fixed)')
    ax.legend(fontsize=7.5, loc='upper left')

    ax = axes[2]
    ax.plot(levels, alpha_fixed_gamma, color='#444444', lw=1.4, zorder=1)
    ax.scatter(levels, alpha_fixed_gamma, c=colors, s=55, edgecolors='white', linewidths=0.7, zorder=3)
    ax.axhline(1.0, color='#999999', lw=1.0, ls=':')
    ax.set_xlabel('context level')
    ax.set_ylabel('alpha')
    ax.set_title(f'alpha vs context\n(gamma={fixed_gamma:g} fixed)')

    ax = axes[3]
    ax.plot(levels, gamma_fixed_alpha, color='#444444', lw=1.4, zorder=1)
    ax.scatter(levels, gamma_fixed_alpha, c=colors, s=55, edgecolors='white', linewidths=0.7, zorder=3)
    ax.axhline(1.0, color='#999999', lw=1.0, ls=':')
    ax.set_xlabel('context level')
    ax.set_ylabel('gamma')
    ax.set_title(f'gamma vs context\n(alpha={fixed_alpha:g} fixed)')

    sm = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label='ctx input')
    fig.savefig(outpath, dpi=240, bbox_inches='tight')
    plt.close(fig)

    csv_path = os.path.splitext(outpath)[0] + '.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('level,alpha_fixed_gamma,beta_alpha_fit,gamma_fixed_alpha,beta_gamma_fit\n')
        for values in zip(levels, alpha_fixed_gamma, alpha_beta, gamma_fixed_alpha, gamma_beta):
            f.write(','.join(f'{v:.8g}' for v in values) + '\n')
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trials_dir')
    parser.add_argument('--out', default='independent_prospect_params.png')
    parser.add_argument('--fixed-gamma', type=float, default=1.0)
    parser.add_argument('--fixed-alpha', type=float, default=1.0)
    args = parser.parse_args()

    rows = load_all_ctx_data(args.trials_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    csv_path = _plot(rows, args.out, args.fixed_gamma, args.fixed_alpha)
    print(f'Saved figure: {os.path.abspath(args.out)}')
    print(f'Saved CSV: {os.path.abspath(csv_path)}')


if __name__ == '__main__':
    main()
