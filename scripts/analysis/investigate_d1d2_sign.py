#!/usr/bin/env python
"""
Investigate the D1/D2 sign reversal:
  - Do D1 neurons actually drive the chosen action?
  - Is the net D1−D2 pull higher for the chosen action on risky vs safe trials?
  - Where does the sign come from when summing over all actions?

Produces four figures:
  sign_1_action_specific.png  — D1pull and D2pull per action at choice time,
                                risky vs safe, for ctx=0 and ctx=+0.6
  sign_2_chosen_vs_unchosen.png — (D1−D2)[chosen] vs (D1−D2)[unchosen] over time
  sign_3_total_vs_chosen.png  — shows why summing all actions flips the sign
  sign_4_logits.png           — raw action logits over time, risky vs safe

Run from repo root:
    python scripts/analysis/investigate_d1d2_sign.py
"""

import os, sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
from pyrl import utils

TRIALS_DIR = os.path.join(REPO_ROOT,
    'data_progression3/d1d2_plasticity_opal04/trials/'
    'gambling_d1d2_plasticity_pos_reg_opal04')
OUT = os.path.join(REPO_ROOT, 'reports/figure_replication/figs')

plt.rcParams.update({
    'figure.dpi': 160, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': True,
    'grid.alpha': 0.18, 'grid.linewidth': 0.5, 'grid.linestyle': '--',
    'font.family': 'sans-serif', 'font.size': 9.5,
    'axes.titlesize': 10, 'axes.titleweight': 'semibold',
    'lines.linewidth': 1.9, 'savefig.bbox': 'tight',
})

RUST = '#a84d2a'; TEAL = '#1f6f68'; GOLD = '#b8860b'; PURPLE = '#5b2d8e'

def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    except: pass
    return np.asarray(x)

def parse_ctx(path):
    m = re.search(r'ctx(neg|pos)(\d+p\d+)', os.path.basename(path))
    if not m: return 0.0
    return round((-1 if m.group(1)=='neg' else 1)*float(m.group(2).replace('p','.')), 3)

def load_ctx(path):
    d = utils.load(path)
    return dict(
        level   = parse_ctx(path),
        trials  = d[0],
        A       = to_np(d[4]),    # (T, B, 3)  one-hot actions
        Z       = to_np(d[2]),    # (T, B, 3)  policy pre-softmax logits
        Z_b     = to_np(d[3]),    # (T, B)     value
        r_policy= to_np(d[8]),    # (T, B, N)
        D1pull  = to_np(d[13]) if len(d) > 13 else None,  # (T, B, 3)
        D2pull  = to_np(d[14]) if len(d) > 14 else None,
    )

def classify_trials(trials, A):
    """Return (risky_idx, safe_idx, choice_ts) — matched-EV only."""
    risky, safe, ch_ts = [], [], []
    acts = np.argmax(A, axis=2)  # (T, B)
    for i, tr in enumerate(trials):
        ct = np.where((acts[:,i]==1)|(acts[:,i]==2))[0]
        if ct.size == 0: continue
        c  = int(acts[ct[0], i])
        pl = float(tr['prob_l']); pr = float(tr['prob_r'])
        evl = pl*float(tr['size_l']); evr = pr*float(tr['size_r'])
        if abs(evl-evr) > 0.015 or abs(pl-pr) < 1e-6: continue
        risky_choice = (c==2 and pr<pl) or (c==1 and pl<pr)
        chosen_action  = c                          # 1=left, 2=right
        unchosen_action = 2 if c==1 else 1
        (risky if risky_choice else safe).append(
            (i, ct[0], chosen_action, unchosen_action))
    return risky, safe

def mean_sem(x, axis=0):
    m = x.mean(axis=axis)
    s = x.std(axis=axis) / np.sqrt(x.shape[axis])
    return m, s

def epoch_bg(ax, T, dt=10):
    ax.axvspan(0,      25*dt, color='#eef2f8', alpha=0.9, zorder=0)
    ax.axvspan(25*dt,  50*dt, color='#eef8f0', alpha=0.9, zorder=0)
    ax.axvspan(50*dt,  T*dt,  color='#fdf2ee', alpha=0.9, zorder=0)
    for t in [25*dt, 50*dt]:
        ax.axvline(t, color='#ccc', lw=0.9, ls='--', zorder=5)

def plot_line(ax, time_ms, data, label, color, lw=2.0):
    m, s = mean_sem(data, axis=1)
    ax.fill_between(time_ms, m-s, m+s, alpha=0.18, color=color)
    ax.plot(time_ms, m, '-', color=color, lw=lw, label=label)

def main():
    # Load two ctx levels
    target_levels = {0.0: None, 0.6: None, -0.6: None}
    for fn in sorted(os.listdir(TRIALS_DIR)):
        if not fn.endswith('.pkl') or 'ctx' not in fn: continue
        lv = parse_ctx(os.path.join(TRIALS_DIR, fn))
        for t in target_levels:
            if target_levels[t] is None and abs(lv - t) < 0.08:
                target_levels[t] = load_ctx(os.path.join(TRIALS_DIR, fn))

    ctx_items = [(lv, d) for lv, d in sorted(target_levels.items()) if d is not None]
    n_ctx = len(ctx_items)

    T  = ctx_items[0][1]['A'].shape[0]
    dt = 10
    time_ms = np.arange(T) * dt

    # ═══════════════════════════════════════════════════════════════════════════
    # Fig 1 — Action-specific D1/D2 pull at the choice timestep
    # ═══════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(2, n_ctx, figsize=(5.5*n_ctx, 9.0), sharey='row')
    action_names = ['fixate', 'left', 'right']

    for col, (lv, d) in enumerate(ctx_items):
        if d['D1pull'] is None:
            continue
        risky, safe = classify_trials(d['trials'], d['A'])

        # At each trial, grab D1pull and D2pull at the choice timestep
        for row, (group, gname, gcol) in enumerate([(risky,'risky',RUST),(safe,'safe',TEAL)]):
            ax = axes1[row, col]
            epoch_bg(ax, T, dt)
            ax.axhline(0, color='#aaa', lw=0.8, ls=':')

            for act_i, act_name in enumerate(action_names):
                d1 = d['D1pull'][:, [g[0] for g in group], act_i]  # (T, n_group)
                d2 = d['D2pull'][:, [g[0] for g in group], act_i]
                net= d1 - d2
                col_line = [RUST, TEAL, GOLD][act_i]
                plot_line(ax, time_ms, d1,  f'D1[{act_name}]',   col_line,  lw=1.7)
                # dashed for D2
                m2, s2 = mean_sem(d2, axis=1)
                ax.plot(time_ms, m2, '--', color=col_line, lw=1.2, alpha=0.7,
                        label=f'D2[{act_name}]')

            ax.set_title(f'ctx={lv:+.2f} | {gname}  (n={len(group)})', fontsize=9.5)
            ax.set_xlabel('time (ms)')
            if col == 0:
                ax.set_ylabel('D1 pull (solid) / D2 pull (dashed)\nper output action')
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, ncol=2, loc='lower left')

    fig1.suptitle('D1 and D2 pull per action  (solid=D1, dashed=D2)\n'
                  'Is D1 actually high for the chosen action on risky trials?',
                  fontsize=11, fontweight='semibold')
    fig1.tight_layout()
    p = os.path.join(OUT, 'sign_1_action_specific.png')
    fig1.savefig(p); plt.close(fig1); print('saved', p)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fig 2 — (D1−D2) for the CHOSEN action vs the UNCHOSEN action over time
    # ═══════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, n_ctx, figsize=(5.5*n_ctx, 8.5), sharey='row', sharex=True)

    for col, (lv, d) in enumerate(ctx_items):
        if d['D1pull'] is None: continue
        risky, safe = classify_trials(d['trials'], d['A'])
        net_pull = d['D1pull'] - d['D2pull']  # (T, B, 3)

        for row, (group, gname, gcol) in enumerate([(risky,'risky',RUST),(safe,'safe',TEAL)]):
            ax = axes2[row, col]
            epoch_bg(ax, T, dt)
            ax.axhline(0, color='#aaa', lw=0.8, ls=':')

            chosen_net, unchosen_net, total_net = [], [], []
            for trial_i, choice_t, ch_act, unch_act in group:
                chosen_net.append(net_pull[:, trial_i, ch_act])
                unchosen_net.append(net_pull[:, trial_i, unch_act])
                total_net.append(net_pull[:, trial_i, :].sum(axis=1))

            chosen_net   = np.stack(chosen_net,   axis=1)  # (T, n)
            unchosen_net = np.stack(unchosen_net, axis=1)
            total_net    = np.stack(total_net,    axis=1)

            plot_line(ax, time_ms, chosen_net,   'chosen action (D1−D2)',   gcol,   lw=2.2)
            plot_line(ax, time_ms, unchosen_net, 'unchosen action (D1−D2)', '#888', lw=1.6)
            plot_line(ax, time_ms, total_net,    'sum all actions',         PURPLE, lw=1.3)

            ax.set_title(f'ctx={lv:+.2f} | {gname}  (n={len(group)})', fontsize=9.5)
            ax.set_xlabel('time (ms)')
            if col == 0:
                ax.set_ylabel('D1−D2 net pull')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')

    fig2.suptitle('(D1−D2) pull for chosen vs unchosen action\n'
                  'Purple = sum over all actions (what the diff-plot was showing)',
                  fontsize=11, fontweight='semibold')
    fig2.tight_layout()
    p = os.path.join(OUT, 'sign_2_chosen_vs_unchosen.png')
    fig2.savefig(p); plt.close(fig2); print('saved', p)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fig 3 — The key: (D1−D2)[chosen] Δ(risky−safe) vs total Δ
    # This shows exactly where the sign reversal comes from
    # ═══════════════════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(1, n_ctx, figsize=(5.5*n_ctx, 4.8), sharex=True)
    if n_ctx == 1: axes3 = [axes3]

    for col, (lv, d) in enumerate(ctx_items):
        if d['D1pull'] is None: continue
        ax = axes3[col]
        risky, safe = classify_trials(d['trials'], d['A'])
        net_pull = d['D1pull'] - d['D2pull']   # (T, B, 3)

        def get_chosen_series(group):
            ch, un, tot = [], [], []
            for trial_i, _, ch_act, unch_act in group:
                ch.append(net_pull[:, trial_i, ch_act])
                un.append(net_pull[:, trial_i, unch_act])
                tot.append(net_pull[:, trial_i, :].sum(1))
            return (np.stack(ch, 1), np.stack(un, 1), np.stack(tot, 1))

        r_ch, r_un, r_tot = get_chosen_series(risky)
        s_ch, s_un, s_tot = get_chosen_series(safe)

        epoch_bg(ax, T, dt)
        ax.axhline(0, color='#aaa', lw=0.8, ls=':')

        # Δ = risky_mean − safe_mean for each quantity
        def delta_plot(r_data, s_data, label, color, lw=2.0, ls='-'):
            rm, rs_ = mean_sem(r_data, 1)
            sm, ss  = mean_sem(s_data, 1)
            diff = rm - sm
            sem  = np.sqrt(rs_**2 + ss**2)
            ax.fill_between(time_ms, diff-sem, diff+sem, alpha=0.15, color=color)
            ax.plot(time_ms, diff, ls, color=color, lw=lw, label=label)

        delta_plot(r_ch,  s_ch,  'Δ chosen action (D1−D2)',    RUST,   lw=2.4)
        delta_plot(r_un,  s_un,  'Δ unchosen action (D1−D2)',  TEAL,   lw=1.8)
        delta_plot(r_tot, s_tot, 'Δ sum all actions',           PURPLE, lw=1.4, ls='--')

        ax.set_xlabel('time (ms)')
        ax.set_title(f'ctx = {lv:+.2f}\n(risky − safe) per quantity')
        if col == 0:
            ax.set_ylabel('Δ (D1−D2) pull  [risky − safe]')
            ax.legend(fontsize=8.5)

    fig3.suptitle('Where does the sign come from?\n'
                  'Δ for the chosen action vs the unchosen action vs the total sum\n'
                  'Positive = risky trials have higher pull',
                  fontsize=11, fontweight='semibold')
    fig3.tight_layout()
    p = os.path.join(OUT, 'sign_3_total_vs_chosen.png')
    fig3.savefig(p); plt.close(fig3); print('saved', p)

    # ═══════════════════════════════════════════════════════════════════════════
    # Fig 4 — Raw logits over time, risky vs safe
    # This is the ground truth: which action wins
    # ═══════════════════════════════════════════════════════════════════════════
    fig4, axes4 = plt.subplots(2, n_ctx, figsize=(5.5*n_ctx, 8.5), sharex=True, sharey='row')

    for col, (lv, d) in enumerate(ctx_items):
        risky, safe = classify_trials(d['trials'], d['A'])
        Z = d['Z']   # (T, B, 3) pre-softmax logits

        for row, (group, gname, gcol) in enumerate([(risky,'risky',RUST),(safe,'safe',TEAL)]):
            ax = axes4[row, col]
            epoch_bg(ax, T, dt)
            ax.axhline(0, color='#aaa', lw=0.8, ls=':')

            ch_logit, un_logit, fix_logit = [], [], []
            for trial_i, _, ch_act, unch_act in group:
                ch_logit.append(Z[:, trial_i, ch_act])
                un_logit.append(Z[:, trial_i, unch_act])
                fix_logit.append(Z[:, trial_i, 0])

            plot_line(ax, time_ms, np.stack(ch_logit,  1), 'chosen action logit',   gcol,   2.2)
            plot_line(ax, time_ms, np.stack(un_logit,  1), 'unchosen action logit', '#888', 1.6)
            plot_line(ax, time_ms, np.stack(fix_logit, 1), 'fixate logit',          PURPLE, 1.2)

            ax.set_title(f'ctx={lv:+.2f} | {gname}  (n={len(group)})', fontsize=9.5)
            ax.set_xlabel('time (ms)')
            if col == 0:
                ax.set_ylabel('action logit (pre-softmax)')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='lower left')

    fig4.suptitle('Raw action logits over time — chosen vs unchosen vs fixate\n'
                  'Ground truth: what the network is actually outputting',
                  fontsize=11, fontweight='semibold')
    fig4.tight_layout()
    p = os.path.join(OUT, 'sign_4_logits.png')
    fig4.savefig(p); plt.close(fig4); print('saved', p)

    print('\nDone. Open the sign_*.png files to investigate.')


if __name__ == '__main__':
    main()
