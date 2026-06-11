#!/usr/bin/env python
"""
Visualize how network activity changes under different dopamine context levels.

Loads pre-computed trial pkl files (ctx -0.9 → +0.9) and generates:

  Fig 1 — PCA population trajectories
    Mean firing-rate trajectory in PC1-PC2 space for each ctx level,
    colored by time and overlaid. Shows how dopamine reshapes dynamics.

  Fig 2 — Neuron activity heatmaps  (ctx = -0.9, -0.45, 0, +0.45, +0.9)
    Neurons (y) × time (x), sorted by ctx-sensitivity. RdBu colormap.
    Shows which neurons are recruited/suppressed.

  Fig 3 — D1 vs D2 mean activity across ctx levels
    First N/2 neurons (D1) vs last N/2 (D2) average activation vs ctx.
    Demonstrates the push-pull gain modulation.

  Fig 4 — Per-neuron ctx tuning (sorted scatter)
    Correlation of each neuron's mean decision-period activity with ctx level.
    D1 (orange) vs D2 (teal) populations coloured.

  Fig 5 — Single-trial activity snapshots
    For the 5 ctx levels: raster-style firing rate for 10 example trials,
    showing how population patterns shift.

  Fig 6 — Ctx-modulated firing rate distributions
    Violin plots of decision-period firing rates for D1 / D2 at each ctx level.

Usage:
    python scripts/analysis/visualize_network_activity.py
    python scripts/analysis/visualize_network_activity.py --outdir reports/activity_viz
"""

import argparse
import os
import sys
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
from pyrl import utils

TRIALS_DIR = os.path.join(
    REPO_ROOT,
    'data_progression3/d1d2_plasticity_opal04/trials/'
    'gambling_d1d2_plasticity_pos_reg_opal04'
)
DEFAULT_OUT = os.path.join(REPO_ROOT, 'reports/figure_replication/figs')

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 180,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.18,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
    'font.size': 9.5,
    'axes.titlesize': 10,
    'axes.titleweight': 'semibold',
    'axes.labelsize': 9,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'lines.linewidth': 1.7,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def parse_ctx(path):
    m = re.search(r'ctx(neg|pos)(\d+p\d+)', os.path.basename(path))
    if not m:
        return 0.0
    sign = -1.0 if m.group(1) == 'neg' else 1.0
    return round(sign * float(m.group(2).replace('p', '.')), 3)


def load_ctx(path):
    d = utils.load(path)
    return dict(
        trials  = d[0],
        A       = to_np(d[4]),   # (T, B, 3)
        r_policy= to_np(d[8]),   # (T, B, N)
        D1pull  = to_np(d[13]) if len(d) > 13 else None,
        D2pull  = to_np(d[14]) if len(d) > 14 else None,
    )


def ctx_cmap(level, max_abs=0.9):
    t = np.clip(0.5 + 0.5 * level / max_abs, 0, 1)
    return plt.cm.coolwarm(t)


def first_choices(A):
    acts = np.argmax(A, axis=2)
    ch = np.full(A.shape[1], -1, dtype=int)
    for j in range(A.shape[1]):
        ct = np.where((acts[:, j] == 1) | (acts[:, j] == 2))[0]
        if ct.size:
            ch[j] = acts[ct[0], j]
    return ch


# ── Load all ctx levels ────────────────────────────────────────────────────────
def load_all(trials_dir, max_abs=0.91):
    rows = []
    for fn in sorted(os.listdir(trials_dir)):
        if not fn.endswith('.pkl') or 'ctx' not in fn:
            continue
        path = os.path.join(trials_dir, fn)
        lv = parse_ctx(path)
        if abs(lv) > max_abs:
            continue
        d = load_ctx(path)
        d['level'] = lv
        rows.append(d)
    rows.sort(key=lambda r: r['level'])
    return rows


# ── Figure 1: PCA population trajectories ─────────────────────────────────────
def fig_pca_trajectories(rows, outdir, targets=(-0.9, -0.45, 0.0, 0.45, 0.9)):
    from sklearn.decomposition import PCA

    # Fit PCA on the neutral ctx mean activity
    row0 = min(rows, key=lambda r: abs(r['level']))
    r0 = row0['r_policy']          # (T, B, N)
    T, B, N = r0.shape
    mean_traj_0 = r0.mean(axis=1)  # (T, N)

    pca = PCA(n_components=3)
    pca.fit(mean_traj_0)

    time_ms = np.arange(T) * 10
    epochs  = {'fixation': (0, 250), 'stimulus': (250, 500), 'decision': (500, T * 10)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    for ax_idx, (ax, dims) in enumerate(zip(axes, [(0, 1), (0, 2)])):
        for row in rows:
            lv = row['level']
            if not any(abs(lv - t) < 0.08 for t in targets):
                continue
            traj = pca.transform(row['r_policy'].mean(axis=1))  # (T, 3)
            x, y = traj[:, dims[0]], traj[:, dims[1]]

            # Colour line by time
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segs   = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segs, cmap='plasma', linewidth=1.8,
                                norm=plt.Normalize(0, T * 10), alpha=0.85)
            lc.set_array(time_ms[:-1])
            ax.add_collection(lc)

            # Start / end markers
            col = ctx_cmap(lv)
            ax.scatter(x[0],  y[0],  s=50, color=col, zorder=5, marker='o',
                       edgecolors='white', linewidths=0.7)
            ax.scatter(x[-1], y[-1], s=80, color=col, zorder=5, marker='*',
                       edgecolors='white', linewidths=0.6,
                       label=f'ctx={lv:+.2f}')

        ax.set_xlabel(f'PC{dims[0]+1}  ({pca.explained_variance_ratio_[dims[0]]:.0%})')
        ax.set_ylabel(f'PC{dims[1]+1}  ({pca.explained_variance_ratio_[dims[1]]:.0%})')
        ax.set_title(f'Population trajectory  PC{dims[0]+1}–PC{dims[1]+1}\n'
                     '(○ = t=0,  ★ = end,  colour = time via plasma)')
        ax.autoscale(); ax.legend(fontsize=7.5, loc='upper right')

    # Shared colorbar for time
    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(0, T * 10))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.01)
    cb.set_label('time (ms)')

    # Ctx colorbar legend
    sm2 = plt.cm.ScalarMappable(cmap='coolwarm',
                                 norm=plt.Normalize(-0.9, 0.9))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=axes, shrink=0.35, pad=0.04)
    cb2.set_label('ctx level')

    fig.suptitle('PCA population trajectories under different dopamine contexts',
                 fontsize=11, fontweight='semibold', y=1.01)
    path = os.path.join(outdir, 'activity_pca_trajectories.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  saved', path)


# ── Figure 2: Activity heatmap — neurons × all ctx levels (decision period) ────
def fig_heatmaps(rows, outdir):
    """
    Single large heatmap: rows = neurons (sorted by ctx-sensitivity),
    columns = all 19 ctx levels.  Only the decision period is averaged.

    Layout:
      Top marginal  — D1 (orange) and D2 (teal) mean activity across ctx
      Main heatmap  — neurons × ctx, RdBu_r, sorted by tuning slope
      Right marginal — each neuron's linear tuning slope (bar chart)
    """
    dec   = slice(50, None)   # decision period (t=50 → end, ~500–760 ms)
    levels = np.array([r['level'] for r in rows])
    N_ctx  = len(levels)
    N      = rows[0]['r_policy'].shape[2]
    half   = N // 2

    # Mean decision-period activity: (N_ctx, N)
    mean_act = np.array([r['r_policy'][dec].mean(axis=(0, 1)) for r in rows])

    # Sort neurons by linear ctx-tuning slope
    slopes   = np.polyfit(levels, mean_act, 1)[0]   # (N,)
    sort_idx = np.argsort(slopes)[::-1]             # most DA-excited first
    mean_sorted = mean_act[:, sort_idx]             # (N_ctx, N) sorted

    # D1 / D2 means across ctx
    d1_mean = mean_act[:, :half].mean(axis=1)   # (N_ctx,)
    d2_mean = mean_act[:, half:].mean(axis=1)

    # ── Figure layout via explicit axes ────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8.5))
    # proportions: top marginal 15%, main heatmap 75%, right marginal 10%
    ax_top  = fig.add_axes([0.10, 0.84, 0.72, 0.12])   # top marginal
    ax_main = fig.add_axes([0.10, 0.08, 0.72, 0.74])   # main heatmap
    ax_right= fig.add_axes([0.84, 0.08, 0.06, 0.74])   # right marginal (slopes)
    ax_cbar = fig.add_axes([0.92, 0.08, 0.018, 0.74])  # colorbar

    # ── Top marginal: D1 and D2 across ctx ────────────────────────────────────
    ax_top.plot(levels, d1_mean, 'o-', color='#a84d2a', lw=2, ms=5,
                label='D1 mean (neurons ranked ↑)')
    ax_top.plot(levels, d2_mean, 'o-', color='#1f6f68', lw=2, ms=5,
                label='D2 mean (neurons ranked ↓)')
    ax_top.axhline(0, color='#ccc', lw=0.8, ls=':')
    ax_top.axvline(0, color='#ccc', lw=0.8, ls='--', alpha=0.7)
    ax_top.set_xlim(levels[0] - 0.05, levels[-1] + 0.05)
    ax_top.set_xticklabels([])
    ax_top.set_ylabel('mean rate\n(decision)', fontsize=8.5)
    ax_top.legend(fontsize=8, loc='upper left', ncol=2)
    ax_top.grid(True, alpha=0.18, linewidth=0.5, linestyle='--')
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)

    # ── Main heatmap: neurons × ctx ────────────────────────────────────────────
    # vmin/vmax: clip symmetrically at 99th percentile for clarity
    vmax = np.percentile(np.abs(mean_sorted), 99)
    vmax = max(vmax, 0.05)

    im = ax_main.imshow(
        mean_sorted.T,                    # shape (N, N_ctx) — rows=neurons, cols=ctx
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax,
        interpolation='nearest',
        extent=[levels[0], levels[-1], N, 0],
    )
    # D1 / D2 divider
    ax_main.axhline(half, color='#111', lw=1.4, ls='-', alpha=0.6)
    ax_main.text(levels[-1] + 0.01, half / 2,
                 'D1\n(top ranked)', va='center', ha='left', fontsize=8,
                 color='#a84d2a', transform=ax_main.transData)
    ax_main.text(levels[-1] + 0.01, half + half / 2,
                 'D2\n(bottom ranked)', va='center', ha='left', fontsize=8,
                 color='#1f6f68', transform=ax_main.transData)

    ax_main.set_xlabel('dopamine context level', fontsize=9.5)
    ax_main.set_ylabel('neuron  (sorted by ctx-sensitivity, high→low)', fontsize=9.5)
    ax_main.axvline(0, color='#888', lw=1, ls='--', alpha=0.5)

    # ── Right marginal: tuning slope per neuron ────────────────────────────────
    slopes_sorted = slopes[sort_idx]
    neuron_pos    = np.arange(N)
    colors_bar    = ['#a84d2a' if s > 0 else '#1f6f68' for s in slopes_sorted]
    ax_right.barh(neuron_pos, slopes_sorted, height=0.9,
                  color=colors_bar, alpha=0.8, edgecolor='none')
    ax_right.axvline(0, color='#333', lw=0.8)
    ax_right.set_ylim(N, 0)
    ax_right.set_xlabel('slope\n(rate/ctx)', fontsize=8)
    ax_right.set_yticklabels([])
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.grid(axis='x', alpha=0.2, linewidth=0.5)
    ax_right.axhline(half, color='#111', lw=1.4, ls='-', alpha=0.6)

    # ── Colorbar ───────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label('mean firing rate\n(decision period)', fontsize=8.5)

    fig.suptitle(
        'Network activity during the decision period — all 19 dopamine context levels\n'
        'Neurons sorted top-to-bottom by ctx-tuning slope  ·  '
        'Red = more active, Blue = less active',
        fontsize=10.5, fontweight='semibold', y=0.99
    )
    path = os.path.join(outdir, 'activity_heatmaps.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('  saved', path)


# ── Figure 3: D1 vs D2 mean activity across ctx levels ────────────────────────
def fig_d1_d2_activation(rows, outdir):
    levels = np.array([r['level'] for r in rows])
    dec    = slice(50, None)
    N      = rows[0]['r_policy'].shape[2]
    half   = N // 2

    d1_mean = np.array([r['r_policy'][dec, :, :half].mean() for r in rows])
    d2_mean = np.array([r['r_policy'][dec, :, half:].mean() for r in rows])
    d1_std  = np.array([r['r_policy'][dec, :, :half].mean(axis=(0, 2)).std() for r in rows])
    d2_std  = np.array([r['r_policy'][dec, :, half:].mean(axis=(0, 2)).std() for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel 1: absolute activity
    ax = axes[0]
    ax.fill_between(levels, d1_mean - d1_std, d1_mean + d1_std, alpha=0.15, color='#a84d2a')
    ax.fill_between(levels, d2_mean - d2_std, d2_mean + d2_std, alpha=0.15, color='#1f6f68')
    ax.plot(levels, d1_mean, 'o-', color='#a84d2a', lw=2, ms=5, label='D1 (neurons 0–49)')
    ax.plot(levels, d2_mean, 'o-', color='#1f6f68', lw=2, ms=5, label='D2 (neurons 50–99)')
    ax.set_xlabel('dopamine context level')
    ax.set_ylabel('mean firing rate (decision period)')
    ax.set_title('D1 vs D2 mean activity\nacross dopamine levels')
    ax.legend(fontsize=8.5)

    # Panel 2: D1 - D2 balance (push-pull index)
    ax = axes[1]
    balance = d1_mean - d2_mean
    scatter_c = [ctx_cmap(l) for l in levels]
    for i in range(len(levels) - 1):
        ax.plot(levels[i:i+2], balance[i:i+2], '-', color=ctx_cmap(levels[i]), lw=2.2,
                solid_capstyle='round')
    ax.scatter(levels, balance, c=scatter_c, s=50, zorder=4,
               edgecolors='white', linewidths=0.7)
    ax.axhline(0, color='#bbb', lw=1, ls=':', label='balanced')
    # linear trend
    fit = np.polyfit(levels, balance, 1)
    xs  = np.array([levels.min(), levels.max()])
    ax.plot(xs, np.polyval(fit, xs), '--', color='#888', lw=1.2, alpha=0.7,
            label=f'slope {fit[0]:+.3f}')
    ax.set_xlabel('dopamine context level')
    ax.set_ylabel('D1 − D2 activity  (push-pull index)')
    ax.set_title('Push-pull balance\n(positive = D1 dominant → risk-seeking)')
    ax.legend(fontsize=8.5)

    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=plt.Normalize(levels.min(), levels.max()))
    sm.set_array([])
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.91, 0.14, 0.015, 0.72])
    fig.colorbar(sm, cax=cax, label='ctx')

    fig.suptitle('D1 / D2 population activity across dopamine contexts\n'
                 '(shaded = ±1 SD across trials)',
                 fontsize=10.5, fontweight='semibold')
    path = os.path.join(outdir, 'activity_d1d2_balance.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  saved', path)


# ── Figure 4: per-neuron ctx tuning scatter ────────────────────────────────────
def fig_neuron_tuning(rows, outdir):
    from numpy.linalg import lstsq

    levels = np.array([r['level'] for r in rows])
    dec    = slice(50, None)
    N      = rows[0]['r_policy'].shape[2]
    half   = N // 2

    # Mean decision-period activity per neuron per ctx: (n_ctx, N)
    mean_act = np.array([r['r_policy'][dec, :, :].mean(axis=(0, 1)) for r in rows])

    # Fit linear tuning slope per neuron
    slopes  = np.polyfit(levels, mean_act, 1)[0]   # (N,)
    baseline= mean_act[np.argmin(np.abs(levels))].copy()

    # Also compute β_HH from neural regression (proxy for risk tuning)
    row0 = min(rows, key=lambda r: abs(r['level']))
    r_pol = row0['r_policy']
    T0, B0, _ = r_pol.shape
    dh_list, valid_idx = [], []
    for i, tr in enumerate(row0['trials']):
        pl, pr = float(tr['prob_l']), float(tr['prob_r'])
        evl, evr = pl*float(tr['size_l']), pr*float(tr['size_r'])
        if pl<=0 or pr<=0 or evl<=0 or evr<=0: continue
        dh_list.append(np.log(pr)-np.log(pl))
        valid_idx.append(i)
    dh  = np.array(dh_list)
    X   = np.column_stack([dh, np.ones(len(dh))])
    act = r_pol[dec, :, :].mean(axis=0)[valid_idx, :]
    coef, _, _, _ = lstsq(X, act, rcond=None)
    beta_hh = coef[0]   # (N,) risk encoding

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel 1: ctx tuning slope per neuron
    ax = axes[0]
    ax.scatter(np.arange(half), slopes[:half], s=28, alpha=0.75, color='#a84d2a',
               edgecolors='none', label='D1 (0–49)')
    ax.scatter(np.arange(half, N), slopes[half:], s=28, alpha=0.75, color='#1f6f68',
               edgecolors='none', label='D2 (50–99)')
    ax.axhline(0, color='#bbb', lw=1, ls=':')
    ax.set_xlabel('neuron index')
    ax.set_ylabel('ctx tuning slope  (firing rate / ctx unit)')
    ax.set_title('Per-neuron ctx sensitivity\n(positive = more active at high DA)')
    ax.legend(fontsize=8.5)

    # Panel 2: ctx slope vs risk tuning (β_HH) scatter
    ax = axes[1]
    ax.scatter(beta_hh[:half], slopes[:half], s=28, alpha=0.7, color='#a84d2a',
               edgecolors='none', label='D1')
    ax.scatter(beta_hh[half:], slopes[half:], s=28, alpha=0.7, color='#1f6f68',
               edgecolors='none', label='D2')
    ax.axhline(0, color='#bbb', lw=0.8, ls=':')
    ax.axvline(0, color='#bbb', lw=0.8, ls=':')
    ax.set_xlabel('β_HH  (risk encoding coefficient)')
    ax.set_ylabel('ctx tuning slope')
    ax.set_title('Risk encoding vs ctx sensitivity\n(per neuron)')
    ax.legend(fontsize=8.5)
    # Correlation annotation
    r = np.corrcoef(beta_hh, slopes)[0, 1]
    ax.text(0.04, 0.96, f'r = {r:.2f}', transform=ax.transAxes,
            va='top', fontsize=9, color='#555')

    fig.suptitle('Per-neuron dopamine context tuning', fontsize=10.5, fontweight='semibold')
    path = os.path.join(outdir, 'activity_neuron_tuning.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  saved', path)


# ── Figure 5: timecourse overlay — D1 & D2 population means ───────────────────
def fig_timecourse_overlay(rows, outdir, targets=(-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)):
    time_ms = np.arange(rows[0]['r_policy'].shape[0]) * 10
    T = len(time_ms)
    N = rows[0]['r_policy'].shape[2]
    half = N // 2

    plot_rows = [min(rows, key=lambda r: abs(r['level'] - t)) for t in targets]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels_added = set()

    for ax, pop_slice, pop_name in zip(axes,
                                        [slice(0, half), slice(half, N)],
                                        ['D1 population (0–49)', 'D2 population (50–99)']):
        for row in plot_rows:
            lv  = row['level']
            col = ctx_cmap(lv)
            # mean over trials and neurons in population
            pop_mean = row['r_policy'][:, :, pop_slice].mean(axis=(1, 2))  # (T,)
            pop_sem  = row['r_policy'][:, :, pop_slice].mean(axis=2).std(axis=1) / \
                       np.sqrt(row['r_policy'].shape[1])
            lbl = f'ctx={lv:+.1f}'
            ax.fill_between(time_ms, pop_mean - pop_sem, pop_mean + pop_sem,
                            alpha=0.13, color=col)
            ax.plot(time_ms, pop_mean, '-', color=col, lw=1.9, alpha=0.9,
                    label=lbl)

        for t_ms, ls, lbl in [(250, '--', 'stim'), (500, ':', 'decision')]:
            ax.axvline(t_ms, color='#999', lw=0.9, ls=ls, alpha=0.8)
            ax.text(t_ms + 6, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] != 1 else 0.97,
                    lbl, fontsize=7.5, color='#888', va='top')

        ax.set_xlabel('time (ms)')
        ax.set_ylabel('mean firing rate ± SEM')
        ax.set_title(pop_name)
        ax.legend(fontsize=7.5, ncol=2, loc='lower right')

    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=plt.Normalize(-0.9, 0.9))
    sm.set_array([])
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.91, 0.14, 0.015, 0.72])
    fig.colorbar(sm, cax=cax, label='ctx level')

    fig.suptitle('Population firing rate timecourses — D1 vs D2 across dopamine levels\n'
                 '(shaded = ±1 SEM across trials)',
                 fontsize=10.5, fontweight='semibold')
    path = os.path.join(outdir, 'activity_timecourse_overlay.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  saved', path)


# ── Figure 6: firing rate distributions per ctx ────────────────────────────────
def fig_violin_distributions(rows, outdir, targets=(-0.9, -0.45, 0.0, 0.45, 0.9)):
    plot_rows = [min(rows, key=lambda r: abs(r['level'] - t)) for t in targets]
    dec = slice(50, None)
    N   = plot_rows[0]['r_policy'].shape[2]
    half = N // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    lv_labels = [f'{r["level"]:+.2f}' for r in plot_rows]

    for ax, pop_slice, title, col in zip(
            axes,
            [slice(0, half), slice(half, N)],
            ['D1 population (neurons 0–49)', 'D2 population (neurons 50–99)'],
            ['#a84d2a', '#1f6f68']):

        data = [row['r_policy'][dec, :, pop_slice].flatten() for row in plot_rows]
        colors = [ctx_cmap(r['level']) for r in plot_rows]

        parts = ax.violinplot(data, positions=range(len(plot_rows)),
                              showmedians=True, showextrema=False,
                              widths=0.7)
        for i, (pc, c) in enumerate(zip(parts['bodies'], colors)):
            pc.set_facecolor(c)
            pc.set_alpha(0.72)
            pc.set_edgecolor('white')
        parts['cmedians'].set_color('#222')
        parts['cmedians'].set_linewidth(1.5)

        ax.set_xticks(range(len(plot_rows)))
        ax.set_xticklabels(lv_labels)
        ax.set_xlabel('dopamine context level')
        ax.set_ylabel('firing rate (decision period, all trials & neurons)')
        ax.set_title(title)

    fig.suptitle('Firing rate distributions — D1 and D2 populations across dopamine contexts',
                 fontsize=10.5, fontweight='semibold')
    path = os.path.join(outdir, 'activity_violin_distributions.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  saved', path)


# ── Main ───────────────────────────────────────────────────────────────────────
def fig_risky_vs_safe(rows, outdir, ctx_levels=(0.0, 0.6, -0.6)):
    """
    For each requested ctx level: split trials into risky-choice and safe-choice,
    then plot mean ± SEM timecourses for:
      Row 1 — D1 population firing rate  (neurons 0 : N//2)
      Row 2 — D2 population firing rate  (neurons N//2 : N)
      Row 3 — D1pull − D2pull net drive  (sum across output actions)

    Risky = chose the lower-probability option on matched-EV trials.
    Safe  = chose the higher-probability option on matched-EV trials.
    All other trials (unmatched EV, no choice) are excluded.
    """
    n_ctx   = len(ctx_levels)
    row0    = min(rows, key=lambda r: abs(r['level']))
    T, _, N = row0['r_policy'].shape
    half    = N // 2
    dt      = 10          # ms per timestep (gambling task)
    time_ms = np.arange(T) * dt

    # Epoch boundaries (timesteps, not ms)
    FIX_END  = 25    # 0–24  fixation
    STIM_END = 50    # 25–49 stimulus
    # decision: 50–76

    COLOR_RISKY = '#a84d2a'   # rust
    COLOR_SAFE  = '#1f6f68'   # teal

    fig, axes = plt.subplots(
        3, n_ctx,
        figsize=(5.2 * n_ctx, 10.5),
        sharex=True, sharey='row',
    )
    if n_ctx == 1:
        axes = axes[:, np.newaxis]

    row_titles = [
        'D1 population  (mean firing rate)',
        'D2 population  (mean firing rate)',
        'Choice advantage  (D1−D2)[chosen] − (D1−D2)[unchosen]',
    ]

    from matplotlib.transforms import blended_transform_factory

    for col_i, target_ctx in enumerate(ctx_levels):
        row = min(rows, key=lambda r: abs(r['level'] - target_ctx))
        lv  = row['level']

        trials  = row['trials']
        A       = row['A']          # (T, B, 3)
        r_pol   = row['r_policy']   # (T, B, N)
        D1pull  = row['D1pull']     # (T, B, 3) or None
        D2pull  = row['D2pull']

        # ── Classify trials; store (trial_idx, chosen_action) tuples ──────────
        risky_info, safe_info = [], []
        for i, tr in enumerate(trials):
            acts = np.argmax(A[:, i, :], axis=1)
            choice_ts = np.where((acts == 1) | (acts == 2))[0]
            if choice_ts.size == 0:
                continue
            c  = int(acts[choice_ts[0]])
            pl = float(tr['prob_l']); pr = float(tr['prob_r'])
            evl = pl * float(tr['size_l']); evr = pr * float(tr['size_r'])
            if abs(evl - evr) > 0.015 or abs(pl - pr) < 1e-6:
                continue
            risky = (c == 2 and pr < pl) or (c == 1 and pl < pr)
            (risky_info if risky else safe_info).append((i, c))   # (trial_idx, chosen_action)

        if len(risky_info) < 5 or len(safe_info) < 5:
            for row_i in range(3):
                axes[row_i, col_i].text(0.5, 0.5, 'too few trials',
                    ha='center', va='center', transform=axes[row_i, col_i].transAxes)
            continue

        risky_idx = np.array([x[0] for x in risky_info])
        safe_idx  = np.array([x[0] for x in safe_info])

        def ms_pop(x, idx):
            """Mean ± SEM over trials, average over last axis (neurons)."""
            sub = x[:, idx, :].mean(axis=2)   # (T, n)
            return sub.mean(1), sub.std(1) / np.sqrt(len(idx))

        def ms_chosen(idx_info):
            """Mean ± SEM of D1pull−D2pull for each trial's CHOSEN action."""
            series = np.stack(
                [D1pull[:, i, c] - D2pull[:, i, c] for i, c in idx_info],
                axis=1)                          # (T, n)
            return series.mean(1), series.std(1) / np.sqrt(len(idx_info))

        # Row 0: D1 population firing rate
        m_r,  s_r  = ms_pop(r_pol[:, :, :half], risky_idx)
        m_s,  s_s  = ms_pop(r_pol[:, :, :half], safe_idx)
        # Row 1: D2 population firing rate
        m_r2, s_r2 = ms_pop(r_pol[:, :, half:], risky_idx)
        m_s2, s_s2 = ms_pop(r_pol[:, :, half:], safe_idx)
        # Row 2: choice advantage = (D1−D2)[chosen] − (D1−D2)[unchosen] per trial
        # Positive = network drives chosen action more than unchosen.
        # Normalises out the background DA bias so the decision signal is visible.
        if D1pull is not None and D2pull is not None:
            def ms_advantage(info):
                series = np.stack(
                    [(D1pull[:, i, c] - D2pull[:, i, c]) -
                     (D1pull[:, i, 3-c] - D2pull[:, i, 3-c])   # unchosen = 1+2-c
                     for i, c in info], axis=1)   # (T, n)
                return series.mean(1), series.std(1) / np.sqrt(len(info))
            m_rn, s_rn = ms_advantage(risky_info)
            m_sn, s_sn = ms_advantage(safe_info)
        else:
            m_rn = m_sn = np.zeros(T)
            s_rn = s_sn = np.zeros(T)

        plot_data = [
            (axes[0, col_i], m_r,  s_r,  m_s,  s_s),
            (axes[1, col_i], m_r2, s_r2, m_s2, s_s2),
            (axes[2, col_i], m_rn, s_rn, m_sn, s_sn),
        ]

        for ax, mr, sr, ms, ss in plot_data:
            ax.axvspan(0,            FIX_END  * dt, color='#eef2f8', alpha=0.9, zorder=0)
            ax.axvspan(FIX_END * dt, STIM_END * dt, color='#eef8f0', alpha=0.9, zorder=0)
            ax.axvspan(STIM_END* dt, T        * dt, color='#fdf2ee', alpha=0.9, zorder=0)
            ax.fill_between(time_ms, mr - sr, mr + sr, alpha=0.18, color=COLOR_RISKY)
            ax.fill_between(time_ms, ms - ss, ms + ss, alpha=0.18, color=COLOR_SAFE)
            ax.plot(time_ms, mr, '-', color=COLOR_RISKY, lw=2.2,
                    label=f'risky  (n={len(risky_info)})')
            ax.plot(time_ms, ms, '-', color=COLOR_SAFE,  lw=2.2,
                    label=f'safe   (n={len(safe_info)})')
            for t_ms in [FIX_END * dt, STIM_END * dt]:
                ax.axvline(t_ms, color='#ccc', lw=0.9, ls='--', zorder=5)
            ax.axhline(0, color='#ccc', lw=0.8, ls=':', zorder=1)

        # Epoch labels inside top row
        trans = blended_transform_factory(axes[0, col_i].transData,
                                          axes[0, col_i].transAxes)
        for mid_t, lbl in [(FIX_END * dt / 2,          'fixation'),
                           ((FIX_END + STIM_END)*dt/2,  'stimulus'),
                           ((STIM_END + T)*dt/2,         'decision')]:
            axes[0, col_i].text(mid_t, 0.97, lbl, ha='center', va='top',
                                fontsize=8, color='#555', fontweight='semibold',
                                transform=trans, zorder=6)

        axes[0, col_i].set_title(f'ctx = {lv:+.2f}', fontsize=11, pad=22)
        axes[0, col_i].legend(fontsize=8.5, loc='lower right')
        axes[2, col_i].set_xlabel('time (ms)')

    for row_i, title in enumerate(row_titles):
        axes[row_i, 0].set_ylabel(title, fontsize=9)

    fig.suptitle(
        'Population activity split by choice:  risky vs safe  (matched-EV trials only)\n'
        'Shaded = ±1 SEM across trials',
        fontsize=11, fontweight='semibold', y=1.01
    )
    fig.tight_layout()
    path = os.path.join(outdir, 'activity_risky_vs_safe.png')
    fig.savefig(path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print('  saved', path)

    # ── Companion figure: difference (risky − safe) reveals choice signal ──────
    # The absolute traces are dominated by initial-state settling.
    # Subtracting safe from risky removes the common-mode decrease and exposes
    # the choice-selective modulation directly.
    fig2, axes2 = plt.subplots(
        3, n_ctx,
        figsize=(5.2 * n_ctx, 10.5),
        sharex=True, sharey='row',
    )
    if n_ctx == 1:
        axes2 = axes2[:, np.newaxis]

    diff_row_titles = [
        'D1 population Δ  (risky − safe firing rate)',
        'D2 population Δ  (risky − safe firing rate)',
        'Choice advantage Δ  (risky − safe)\n(D1−D2)[chosen] − (D1−D2)[unchosen]',
    ]

    for col_i, target_ctx in enumerate(ctx_levels):
        row = min(rows, key=lambda r: abs(r['level'] - target_ctx))
        lv  = row['level']
        trials  = row['trials']
        A       = row['A']
        r_pol   = row['r_policy']
        D1pull  = row['D1pull']
        D2pull  = row['D2pull']

        # Store (trial_idx, chosen_action) for correct action indexing
        risky_info, safe_info = [], []
        for i, tr in enumerate(trials):
            acts = np.argmax(A[:, i, :], axis=1)
            ct   = np.where((acts == 1) | (acts == 2))[0]
            if ct.size == 0: continue
            c  = int(acts[ct[0]])
            pl = float(tr['prob_l']); pr = float(tr['prob_r'])
            evl = pl * float(tr['size_l']); evr = pr * float(tr['size_r'])
            if abs(evl - evr) > 0.015 or abs(pl - pr) < 1e-6: continue
            risky = (c == 2 and pr < pl) or (c == 1 and pl < pr)
            (risky_info if risky else safe_info).append((i, c))

        if len(risky_info) < 5 or len(safe_info) < 5:
            for row_i in range(3):
                axes2[row_i, col_i].text(0.5, 0.5, 'too few trials',
                    ha='center', va='center', transform=axes2[row_i, col_i].transAxes)
            continue

        risky_idx = np.array([x[0] for x in risky_info])
        safe_idx  = np.array([x[0] for x in safe_info])

        def pop_diff(pop_slice):
            """Δ mean firing rate for a neuron population."""
            sub_r = r_pol[:, risky_idx, pop_slice].mean(axis=2)
            sub_s = r_pol[:, safe_idx,  pop_slice].mean(axis=2)
            diff  = sub_r.mean(1) - sub_s.mean(1)
            sem   = np.sqrt(sub_r.var(1)/len(risky_idx) + sub_s.var(1)/len(safe_idx))
            return diff, sem

        def advantage_series(info):
            """(D1−D2)[chosen] − (D1−D2)[unchosen] per trial."""
            return np.stack(
                [(D1pull[:, i, c] - D2pull[:, i, c]) -
                 (D1pull[:, i, 3-c] - D2pull[:, i, 3-c])
                 for i, c in info], axis=1)   # (T, n)

        d1_diff, d1_sem = pop_diff(slice(None, half))
        d2_diff, d2_sem = pop_diff(slice(half, None))
        if D1pull is not None and D2pull is not None:
            r_adv = advantage_series(risky_info)  # (T, n_risky)
            s_adv = advantage_series(safe_info)   # (T, n_safe)
            dn_diff = r_adv.mean(1) - s_adv.mean(1)
            dn_sem  = np.sqrt(r_adv.var(1)/len(risky_info) + s_adv.var(1)/len(safe_info))
        else:
            dn_diff = dn_sem = np.zeros(T)

        from matplotlib.transforms import blended_transform_factory

        for row_i, (ax, diff, sem) in enumerate(zip(
                axes2[:, col_i],
                [d1_diff, d2_diff, dn_diff],
                [d1_sem,  d2_sem,  dn_sem])):

            ax.axvspan(0,            FIX_END  * dt, color='#eef2f8', alpha=0.9, zorder=0)
            ax.axvspan(FIX_END * dt, STIM_END * dt, color='#eef8f0', alpha=0.9, zorder=0)
            ax.axvspan(STIM_END* dt, T        * dt, color='#fdf2ee', alpha=0.9, zorder=0)

            ax.axhline(0, color='#888', lw=1.1, ls='--', zorder=2)
            ax.fill_between(time_ms, diff - sem, diff + sem,
                            alpha=0.22, color='#444', zorder=3)
            ax.plot(time_ms, diff, '-', color='#1a1a1a', lw=2.2, zorder=4,
                    label='risky − safe')

            for t_ms in [FIX_END * dt, STIM_END * dt]:
                ax.axvline(t_ms, color='#ccc', lw=0.9, ls='--', zorder=5)

            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8.5, loc='lower right')

            ax.set_xlabel('time (ms)') if row_i == 2 else None

        # Epoch labels inside top row
        trans2 = blended_transform_factory(axes2[0, col_i].transData,
                                           axes2[0, col_i].transAxes)
        for mid_t, lbl in [(FIX_END * dt / 2,          'fixation'),
                           ((FIX_END + STIM_END)*dt/2,  'stimulus'),
                           ((STIM_END + T)*dt/2,         'decision')]:
            axes2[0, col_i].text(mid_t, 0.97, lbl, ha='center', va='top',
                                 fontsize=8, color='#555', fontweight='semibold',
                                 transform=trans2, zorder=6)

        axes2[0, col_i].set_title(f'ctx = {lv:+.2f}', fontsize=11, pad=22)

    for row_i, title in enumerate(diff_row_titles):
        axes2[row_i, 0].set_ylabel(title, fontsize=9)

    fig2.suptitle(
        'Choice-selective signal:  risky − safe  (matched-EV trials only)\n'
        'Common-mode settling removed — this is the activity that encodes the choice\n'
        'Shaded = propagated SEM',
        fontsize=11, fontweight='semibold', y=1.01
    )
    fig2.tight_layout()
    path2 = os.path.join(outdir, 'activity_risky_vs_safe_diff.png')
    fig2.savefig(path2, bbox_inches='tight', dpi=180)
    plt.close(fig2)
    print('  saved', path2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials-dir', default=TRIALS_DIR)
    ap.add_argument('--outdir', default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('Loading pkl files ...')
    rows = load_all(args.trials_dir)
    print(f'  {len(rows)} ctx levels: {[r["level"] for r in rows]}')
    T, B, N = rows[0]['r_policy'].shape
    print(f'  Shape: T={T}, B={B}, N={N}')

    print('\nFig 1: PCA trajectories ...')
    fig_pca_trajectories(rows, args.outdir)

    print('Fig 2: Activity heatmaps ...')
    fig_heatmaps(rows, args.outdir)

    print('Fig 3: D1/D2 activation vs ctx ...')
    fig_d1_d2_activation(rows, args.outdir)

    print('Fig 4: Per-neuron ctx tuning ...')
    fig_neuron_tuning(rows, args.outdir)

    print('Fig 5: Timecourse overlay ...')
    fig_timecourse_overlay(rows, args.outdir)

    print('Fig 6: Violin distributions ...')
    fig_violin_distributions(rows, args.outdir)

    print('Fig 7: Risky vs safe population activity ...')
    fig_risky_vs_safe(rows, args.outdir, ctx_levels=(0.0, 0.6, -0.6))

    print(f'\nAll figures saved to {args.outdir}')


if __name__ == '__main__':
    main()
