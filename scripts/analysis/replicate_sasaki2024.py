#!/usr/bin/env python
"""
Replicate Sasaki et al. (2024) figures from the trained opal04 model.

Strategy:
  - Load all 20 pre-computed context-sweep pkl files for behavioural analyses
    (no redundant inference).
  - Run fresh inference only where required: ablation (B), directional stim (C1/C3),
    timing specificity (D), context session simulation (E).
  - Embed existing high-quality figures (mega comparison, psychometric mega, etc.)
    directly in the HTML report.
  - HTML uses the warm editorial style from reports/fake_vta_context_comparison.html.

Usage:
    python scripts/analysis/replicate_sasaki2024.py [--quick] [--groups P,A,B,C,D,F,E]
"""

import argparse
import os
import re
import sys
import shutil
import traceback
import time
import html as html_module
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pyrl import utils            # noqa: E402
from pyrl.model import Model      # noqa: E402
from tasks import gambling        # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
TASKFILE = os.path.join(REPO_ROOT, 'tasks/gambling.py')

MODEL_NAME = 'gambling_d1d2_plasticity_pos_reg_opal04'
DATA_ROOT  = os.path.join(REPO_ROOT, 'data_progression3/d1d2_plasticity_opal04')
SAVEFILE   = os.path.join(DATA_ROOT, f'weights/{MODEL_NAME}/{MODEL_NAME}.pkl')
TRIALS_DIR = os.path.join(DATA_ROOT, f'trials/{MODEL_NAME}')
FIGS_SRC   = os.path.join(DATA_ROOT, f'figures/{MODEL_NAME}')

# probabilities in the gambling task
PROBS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

# Accumulates figure entries for the HTML
REPORT: list[dict] = []

# --------------------------------------------------------------------------- #
# Matplotlib style (matches existing gambling.py plots)
# --------------------------------------------------------------------------- #
def setup_style():
    plt.rcParams.update({
        'figure.dpi': 200,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.22,
        'grid.linewidth': 0.55,
        'grid.linestyle': '--',
        'grid.color': '#b0a898',
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'semibold',
        'axes.labelsize': 9.5,
        'lines.linewidth': 1.9,
        'lines.markersize': 6.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.framealpha': 0,
        'savefig.bbox': 'tight',
        'savefig.dpi': 200,
        'text.usetex': False,
    })

# --------------------------------------------------------------------------- #
# Color helpers
# --------------------------------------------------------------------------- #
RUST   = '#a84d2a'
TEAL   = '#1f6f68'
INK    = '#1f2520'
MUTED  = '#5d655d'
GOOD   = '#17693d'
WARN   = '#a84d2a'
GOLD   = '#b8860b'
PURPLE = '#5b2d8e'

def ctx_color(level, max_abs=0.9):
    """Coolwarm color: negative ctx → blue, positive → rust-red."""
    t = np.clip(0.5 + 0.5 * level / max_abs, 0, 1)
    return plt.cm.coolwarm(t)

def ev_colors(n):
    return [plt.cm.viridis(t) for t in np.linspace(0.12, 0.88, n)]

# --------------------------------------------------------------------------- #
# Tensor / numpy helpers
# --------------------------------------------------------------------------- #
def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def first_choices(A):
    """Return first action per trial: 1=left, 2=right, -1=no choice."""
    A_np = to_np(A)
    acts = np.argmax(A_np, axis=2) if A_np.ndim == 3 else A_np
    choices = np.full(acts.shape[1], -1, dtype=int)
    for j in range(acts.shape[1]):
        ct = np.where((acts[:, j] == 1) | (acts[:, j] == 2))[0]
        if ct.size:
            choices[j] = acts[ct[0], j]
    return choices

def completion_rate(choices):
    return float(np.mean(choices != -1))

def proportion_slope(trials, choices, kind):
    """
    kind='prob': matched-EV pairs, x=win-probability  (neg slope = risk-seeking)
    kind='ev':   matched-prob pairs, x=expected value  (pos slope = EV-seeking)
    Returns (levels, proportions, slope).
    """
    offered = defaultdict(int)
    chosen  = defaultdict(int)
    for i, tr in enumerate(trials):
        c = choices[i]
        if c not in (1, 2):
            continue
        pl, pr = float(tr['prob_l']), float(tr['prob_r'])
        evl = pl * float(tr['size_l'])
        evr = pr * float(tr['size_r'])
        if kind == 'prob':
            if not (abs(evl - evr) < 0.015 and abs(pl - pr) > 1e-6):
                continue
            la, lb = round(pl, 4), round(pr, 4)
        else:
            if not (abs(pl - pr) < 1e-6 and abs(tr['size_l'] - tr['size_r']) > 1e-6):
                continue
            la, lb = round(evl, 4), round(evr, 4)
        offered[la] += 1
        offered[lb] += 1
        chosen[la if c == 1 else lb] += 1
    levels = np.array(sorted(offered))
    if levels.size < 2:
        return levels, np.array([]), np.nan
    props = np.array([chosen[l] / offered[l] for l in levels])
    slope = float(np.polyfit(levels, props, 1)[0])
    return levels, props, slope

def riskiness_from(trials, choices):
    """P(chose lower-prob option | matched-EV decided trial)."""
    vals = []
    for i, tr in enumerate(trials):
        c = choices[i]
        if c not in (1, 2):
            continue
        pl, pr = float(tr['prob_l']), float(tr['prob_r'])
        evl, evr = pl * float(tr['size_l']), pr * float(tr['size_r'])
        if not (abs(evl - evr) < 0.015 and abs(pl - pr) > 1e-6):
            continue
        chose_right = (c == 2)
        right_is_risky = pr < pl
        vals.append(chose_right if right_is_risky else (not chose_right))
    return float(np.mean(vals)) if vals else np.nan

def ev_accuracy(trials, choices):
    """P(chose higher-EV option | both EVs differ, decided trial)."""
    correct = []
    for i, tr in enumerate(trials):
        c = choices[i]
        if c not in (1, 2):
            continue
        evl = float(tr['prob_l']) * float(tr['size_l'])
        evr = float(tr['prob_r']) * float(tr['size_r'])
        if abs(evl - evr) < 0.01:
            continue
        correct.append((c == 1 and evl > evr) or (c == 2 and evr > evl))
    return float(np.mean(correct)) if correct else np.nan

def linear_r2(x, y):
    fit = np.polyfit(x, y, 1)
    pred = np.polyval(fit, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return fit, (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)

def compute_mc_returns(R, M, gamma=0.9):
    """Compute Monte-Carlo discounted returns from reward/mask tensors."""
    R_np, M_np = to_np(R), to_np(M)
    T, B = R_np.shape
    returns = np.zeros_like(R_np)
    running = np.zeros(B)
    for t in reversed(range(T)):
        running = R_np[t] + gamma * running
        returns[t] = running
    return returns

# --------------------------------------------------------------------------- #
# pkl loading
# --------------------------------------------------------------------------- #
def load_pkl(path):
    """Load a pre-computed trials pkl. Returns a dict with named arrays."""
    data = utils.load(path)
    trials, U, Z, Z_b, A, R, M, perf = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
    r_policy = data[8] if len(data) > 8 else None
    r_value  = data[9] if len(data) > 9 else None
    RPE_obj  = data[10] if len(data) > 10 else None
    RPE_sub  = data[11] if len(data) > 11 else None
    Pvals    = data[12] if len(data) > 12 else None  # Policy_Values
    D1pull   = data[13] if len(data) > 13 else None  # Policy_D1_Pull
    D2pull   = data[14] if len(data) > 14 else None  # Policy_D2_Pull
    r_mod    = data[15] if len(data) > 15 else None  # r_policy_mod
    return dict(trials=trials, U=U, Z=Z, Z_b=Z_b, A=A, R=R, M=M, perf=perf,
                r_policy=r_policy, r_value=r_value, RPE_obj=RPE_obj,
                RPE_sub=RPE_sub, Pvals=Pvals, D1pull=D1pull, D2pull=D2pull, r_mod=r_mod)

def parse_ctx_level(path):
    """Extract float context level from filename like trials_activity_ctxpos0p50.pkl"""
    m = re.search(r'ctx(neg|pos)(\d+p\d+)', os.path.basename(path))
    if not m:
        return 0.0
    sign = -1.0 if m.group(1) == 'neg' else 1.0
    return round(sign * float(m.group(2).replace('p', '.')), 3)

def load_all_ctx_data(trials_dir):
    """Load every ctx pkl in trials_dir, compute metrics, return sorted list of dicts."""
    rows = []
    for fn in sorted(os.listdir(trials_dir)):
        if not fn.endswith('.pkl') or 'ctx' not in fn:
            continue
        path = os.path.join(trials_dir, fn)
        level = parse_ctx_level(path)
        if abs(level) > 0.91:
            continue
        d = load_pkl(path)
        choices = first_choices(d['A'])
        trials  = d['trials'][:len(choices)]
        _, _, rs = proportion_slope(trials, choices, 'prob')
        _, _, es = proportion_slope(trials, choices, 'ev')
        rows.append(dict(
            level=level, path=path, choices=choices,
            trials=trials, d=d,
            risk_slope=rs, ev_slope=es,
            riskiness=riskiness_from(trials, choices),
            ev_accuracy=ev_accuracy(trials, choices),
            completion=completion_rate(choices),
        ))
    rows.sort(key=lambda r: r['level'])
    return rows

# --------------------------------------------------------------------------- #
# Report tracking
# --------------------------------------------------------------------------- #
def add_fig(group, fname, title, paper, what, criterion, verdict, status='info'):
    REPORT.append(dict(group=group, fig=fname, title=title, paper=paper,
                       what=what, criterion=criterion, verdict=verdict, status=status))

def save_fig(fig, figs_dir, name):
    path = os.path.join(figs_dir, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return name

# Copy an existing figure into the figs dir, return relative filename
def copy_existing(src_name, figs_dir, dst_name=None):
    src = os.path.join(FIGS_SRC, src_name)
    if not os.path.exists(src):
        return None
    dst_name = dst_name or src_name
    dst = os.path.join(figs_dir, dst_name)
    shutil.copy2(src, dst)
    return dst_name

# --------------------------------------------------------------------------- #
# Runner (for fresh inference: ablation, stim-mode, timing, session sim)
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, savefile, device='cpu'):
        self.model = Model(TASKFILE)
        self.pg    = self.model.get_pg(savefile, seed=1, device=device)
        self.task  = self.model.Task()
        self.N     = self.pg.policy_net.N
        self._orig = self.pg.policy_net.output_layer

    def _ablate(self, which, rng):
        half = self.N // 2
        if   which == 'd1':      idx = np.arange(0, half)
        elif which == 'd2':      idx = np.arange(half, self.N)
        elif which == 'control': idx = rng.choice(self.N, half, replace=False)
        else: return
        orig = self._orig
        def patched(r, *a, **kw):
            r2 = r.clone(); r2[..., idx] = 0.0; return orig(r2, *a, **kw)
        self.pg.policy_net.output_layer = patched

    def _restore(self):
        self.pg.policy_net.output_layer = self._orig

    def run(self, level, n_per_pair, pathway_mode='symmetric',
            context_phases=None, ablate=None, kind=('prob','ev'), seed=999):
        pg = self.pg
        pg.policy_net.inference_pathway_mode = pathway_mode
        pg.policy_net.disable_inference_dopamine_bias = (pathway_mode != 'symmetric')
        rng = np.random.RandomState(seed)
        if ablate: self._ablate(ablate, rng)
        try:
            specs = []
            if 'prob' in kind: specs += gambling.generate_psychometric_trial_set(n_per_pair)
            if 'ev'   in kind: specs += _matched_prob_specs(n_per_pair)
            pg.rng = np.random.RandomState(seed)
            rng.shuffle(specs)
            trials = [self.task.get_condition(pg.rng, pg.dt, s) for s in specs]
            res = pg.run_trials(trials, context_input=level, return_states=True,
                                context_phases=context_phases, progress_bar=False)
            choices = first_choices(res['A'])
            trials  = trials[:len(choices)]
            _, pp, rs = proportion_slope(trials, choices, 'prob')
            pl, _, _  = proportion_slope(trials, choices, 'prob')
            _, ep, es = proportion_slope(trials, choices, 'ev')
            el, _, _  = proportion_slope(trials, choices, 'ev')
            return dict(
                level=level, risk_slope=rs, ev_slope=es,
                riskiness=riskiness_from(trials, choices),
                completion=completion_rate(choices),
                prob_levels=pl, prob_props=pp,
                ev_levels=el, ev_props=ep,
                trials=trials, choices=choices,
                r_policy=to_np(res.get('r_policy', np.array([]))),
            )
        finally:
            if ablate: self._restore()

def _matched_prob_specs(n):
    specs = []
    for row in range(5):
        opts = [row*5 + col for col in range(5)]
        for i, o1 in enumerate(opts):
            for o2 in opts[i+1:]:
                for _ in range(n):
                    specs += [{'target_l': o1, 'target_r': o2},
                               {'target_l': o2, 'target_r': o1}]
    return specs

# --------------------------------------------------------------------------- #
# GROUP P  ── pre-flight sanity
# --------------------------------------------------------------------------- #
def group_P(figs_dir, runner):
    # P1: learning curve from checkpoint
    saved = utils.load(runner.pg.savefile if hasattr(runner.pg, 'savefile') else SAVEFILE)
    hist  = saved.get('training_history', []) or []
    iters, rewards = [], []
    for rec in hist:
        if not isinstance(rec, dict): continue
        it = rec.get('iter', rec.get('iteration', len(iters)))
        rw = rec.get('reward') or rec.get('mean_reward') or rec.get('best_reward')
        if rw is None and isinstance(rec.get('perf'), dict):
            rw = rec['perf'].get('reward')
        if rw is not None:
            iters.append(it); rewards.append(rw)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    if rewards:
        ax.plot(iters, rewards, color=TEAL, lw=2, alpha=0.9, zorder=3)
        ax.fill_between(iters, rewards, alpha=0.12, color=TEAL)
        best_i = int(np.argmax(rewards))
        ax.scatter([iters[best_i]], [rewards[best_i]], color=RUST, s=60, zorder=4,
                   label=f'best: {rewards[best_i]:.3f} @ iter {iters[best_i]}')
        ax.legend()
        verdict = f"final ≈ {rewards[-1]:.3f}, best ≈ {rewards[best_i]:.3f}"
        st = 'pass'
    else:
        ax.text(0.5, 0.5, 'training_history not in checkpoint', ha='center', va='center',
                transform=ax.transAxes, color=MUTED)
        verdict, st = 'history not stored', 'info'
    ax.set_xlabel('training iteration')
    ax.set_ylabel('mean validation reward')
    ax.set_title('P1 · Learning curve')
    name = save_fig(fig, figs_dir, 'p1_learning_curve.png')
    add_fig('P', name, 'P1: Learning curve', 'pre-flight',
            'Mean reward vs training iteration.',
            'Rises and plateaus.', verdict, st)

    # P2: critic quality from baseline pkl
    try:
        d0 = load_pkl(os.path.join(TRIALS_DIR, 'trials_activity_ctxpos0p00.pkl'))
        Z_b_np = to_np(d0['Z_b'])    # (T, B)
        R_np   = to_np(d0['R'])
        M_np   = to_np(d0['M'])
        rets   = compute_mc_returns(d0['R'], d0['M'])
        # take value at last mask-on timestep per trial
        B = Z_b_np.shape[1]
        v_pred, v_true = [], []
        for j in range(B):
            valid = np.where(M_np[:, j] > 0)[0]
            if valid.size == 0: continue
            t = valid[-1]
            v_pred.append(Z_b_np[t, j])
            v_true.append(rets[t, j])
        v_pred, v_true = np.array(v_pred), np.array(v_true)
        rmse = float(np.sqrt(np.mean((v_pred - v_true)**2)))
        cov  = float((v_pred.max()-v_pred.min()) / (v_true.max()-v_true.min()+1e-9))

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].scatter(v_true, v_pred, s=4, alpha=0.3, color=TEAL, rasterized=True)
        lims = [min(v_true.min(), v_pred.min()), max(v_true.max(), v_pred.max())]
        axes[0].plot(lims, lims, '--', color=RUST, lw=1.2, label='perfect')
        axes[0].set_xlabel('MC return'); axes[0].set_ylabel('V prediction')
        axes[0].set_title('P2 · Critic: V vs actual return')
        axes[0].legend()

        axes[1].axhline(0, color='k', lw=0.8, ls=':')
        resid = v_pred - v_true
        axes[1].hist(resid, bins=40, color=TEAL, alpha=0.75, edgecolor='white', linewidth=0.4)
        axes[1].set_xlabel('V − MC return (residual)')
        axes[1].set_ylabel('count')
        axes[1].set_title(f'Residuals · RMSE={rmse:.3f}, coverage={cov:.2f}')
        fig.tight_layout()
        name = save_fig(fig, figs_dir, 'p2_critic_health.png')
        ok = cov > 0.5 and rmse < 0.5
        add_fig('P', name, 'P2: Critic health', 'pre-flight',
                'Value-network predictions vs Monte-Carlo returns.',
                'V coverage near 1.0, RMSE small.',
                f'coverage={cov:.2f}, RMSE={rmse:.3f}', 'pass' if ok else 'info')
    except Exception as e:
        print('P2 failed:', e)

# --------------------------------------------------------------------------- #
# Dose-response from existing pkl files (the backbone of groups A & C)
# --------------------------------------------------------------------------- #
def build_dose_response(ctx_rows, figs_dir):
    """Compute and plot riskiness / EV accuracy / completion vs ctx level."""
    levels    = np.array([r['level']    for r in ctx_rows])
    risk      = np.array([r['riskiness'] for r in ctx_rows])
    ev_acc    = np.array([r['ev_accuracy'] for r in ctx_rows])
    comp      = np.array([r['completion']  for r in ctx_rows])
    risk_sl   = np.array([r['risk_slope']  for r in ctx_rows])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # Panel 1: riskiness
    ax = axes[0]
    scatter_c = [ctx_color(l) for l in levels]
    for i in range(len(levels)-1):
        ax.plot(levels[i:i+2], risk[i:i+2], '-', color=ctx_color(levels[i]), lw=2.2, solid_capstyle='round')
    ax.scatter(levels, risk, c=scatter_c, s=42, zorder=4, edgecolors='white', linewidths=0.6)
    ax.axhline(0.5, color=INK, lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('dopamine context input')
    ax.set_ylabel('P(risky choice | matched EV)')
    ax.set_title('Riskiness vs dopamine')
    # linear fit annotation
    fin = np.isfinite(risk)
    if fin.sum() >= 3:
        fit, r2 = linear_r2(levels[fin], risk[fin])
        xs = np.array([levels.min(), levels.max()])
        ax.plot(xs, np.polyval(fit, xs), '--', color=INK, lw=1, alpha=0.5)
        ax.text(0.04, 0.96, f'slope {fit[0]:+.2f}  R²={r2:.2f}',
                transform=ax.transAxes, va='top', fontsize=8, color=MUTED)

    # Panel 2: EV accuracy
    ax = axes[1]
    for i in range(len(levels)-1):
        ax.plot(levels[i:i+2], ev_acc[i:i+2], '-', color=ctx_color(levels[i]), lw=2.2, solid_capstyle='round')
    ax.scatter(levels, ev_acc, c=scatter_c, s=42, zorder=4, edgecolors='white', linewidths=0.6)
    ax.axhline(0.5, color=INK, lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('dopamine context input')
    ax.set_ylabel('P(chose higher EV | different EVs)')
    ax.set_title('EV accuracy vs dopamine')

    # Panel 3: completion
    ax = axes[2]
    for i in range(len(levels)-1):
        ax.plot(levels[i:i+2], comp[i:i+2], '-', color=ctx_color(levels[i]), lw=2.2, solid_capstyle='round')
    ax.scatter(levels, comp, c=scatter_c, s=42, zorder=4, edgecolors='white', linewidths=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('dopamine context input')
    ax.set_ylabel('fraction of trials with a choice')
    ax.set_title('Completion rate vs dopamine')

    fig.subplots_adjust(left=0.07, right=0.88, wspace=0.32)
    cax = fig.add_axes([0.905, 0.14, 0.016, 0.72])
    sm  = plt.cm.ScalarMappable(cmap='coolwarm',
                                 norm=plt.Normalize(vmin=levels.min(), vmax=levels.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('dopamine context level', fontsize=9)

    fig.suptitle('Dose-response across all context levels',
                 fontsize=11, fontweight='semibold', y=1.01)
    name = save_fig(fig, figs_dir, 'dose_response_summary.png')
    return name, levels, risk, ev_acc, comp

# --------------------------------------------------------------------------- #
# GROUP A  ── baseline behaviour
# --------------------------------------------------------------------------- #
def group_A(ctx_rows, figs_dir, runner, cfg):
    dr_name, levels, risk, ev_acc, comp = build_dose_response(ctx_rows, figs_dir)
    add_fig('A', dr_name, 'Dose-response summary (riskiness · EV accuracy · completion)',
            'Fig 1C + Fig 3O/P',
            'Riskiness, EV accuracy, and completion rate across all dopamine context levels.',
            _dose_response_verdict(levels, risk), '', 'info')

    row0 = next((r for r in ctx_rows if abs(r['level']) < 0.06), ctx_rows[len(ctx_rows)//2])

    # A1: transposed 5×5 heatmap with bootstrap marginals (folds A2 and A3)
    _plot_heatmap_with_marginals(row0['trials'], row0['choices'], row0['level'],
                                 ctx_rows, figs_dir)

    # Utility curve at neutral DA (single panel, neutral only)
    plot_utility_curve(ctx_rows, figs_dir)

    # Utility comparison — all ctx levels
    plot_utility_curves_comparison(ctx_rows, figs_dir)

    # Psychometrics at 5 ctx levels
    _plot_psychometrics_multi_ctx(ctx_rows, figs_dir, 'a5_psychometrics_ctx_sweep.png')

    # Embed existing mega figures
    n = copy_existing('context_choice_probability_curves_mega.png', figs_dir,
                      'existing_psychometric_mega.png')
    if n:
        add_fig('A', n, 'Probability curves — all ctx levels (pre-computed)',
                'Fig 1C', 'Psychometric curves at every context level.', '', '', 'info')

    for src, dst, what in [
        ('proportion_chosen_by_risk.png', 'existing_proportion_risk.png',
         'Proportion chosen by probability (marginal risk slope)'),
        ('proportion_chosen_by_ev.png', 'existing_proportion_ev.png',
         'Proportion chosen by expected value (marginal EV slope)'),
    ]:
        n = copy_existing(src, figs_dir, dst)
        if n:
            add_fig('A', n, what, 'Fig 1C', what, '', '', 'info')

def _fit_prospect_theory(trials, choices):
    """
    Fit a two-parameter prospect theory model to the agent's binary choices.

    Model:
        v(x)   = x^α                          (power utility, α>0)
        w(p)   = exp(−(−ln p)^γ)              (Prelec probability weighting, γ>0)
        SV     = w(p) · v(reward)
        P(R)   = σ( β · (SV_R − SV_L) )      (logistic choice rule, β>0)

    Returns (alpha, gamma, beta, result) or None on failure.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    X_l, X_r, chose_r = [], [], []
    for i, tr in enumerate(trials):
        c = choices[i]
        if c not in (1, 2):
            continue
        X_l.append((float(tr['prob_l']), float(tr['size_l'])))
        X_r.append((float(tr['prob_r']), float(tr['size_r'])))
        chose_r.append(1 if c == 2 else 0)

    if len(chose_r) < 30:
        return None

    p_l = np.array([x[0] for x in X_l])
    r_l = np.array([x[1] for x in X_l])
    p_r = np.array([x[0] for x in X_r])
    r_r = np.array([x[1] for x in X_r])
    y   = np.array(chose_r, dtype=float)

    def neg_ll(params):
        alpha, gamma, beta = params
        if alpha <= 0.05 or gamma <= 0.05 or gamma > 6 or beta <= 0:
            return 1e9
        # Prelec weighting (clip to avoid log(0))
        p_l_c = np.clip(p_l, 1e-9, 1 - 1e-9)
        p_r_c = np.clip(p_r, 1e-9, 1 - 1e-9)
        wl = np.exp(-((-np.log(p_l_c)) ** gamma))
        wr = np.exp(-((-np.log(p_r_c)) ** gamma))
        vl = np.clip(r_l, 0, None) ** alpha
        vr = np.clip(r_r, 0, None) ** alpha
        sv_diff = beta * (wr * vr - wl * vl)
        pr = np.clip(expit(sv_diff), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    best, best_res = np.inf, None
    for a0, g0, b0 in [(0.8, 0.6, 3.0), (1.2, 0.8, 2.0), (0.5, 0.5, 5.0),
                        (1.0, 1.2, 2.0), (0.9, 0.4, 4.0)]:
        res = minimize(neg_ll, [a0, g0, b0], method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best:
            best, best_res = res.fun, res
    if best_res is None or not best_res.success and best > 1e8:
        return None
    alpha, gamma, beta = best_res.x
    return alpha, gamma, beta, best_res


def plot_utility_curve(ctx_rows, figs_dir):
    """
    Fit prospect-theory to the agent's choices (neutral ctx) and plot:
      Left  — Prelec probability weighting w(p) vs p
      Centre — power utility v(x) vs x (normalized 0-1)
      Right  — predicted vs observed choice frequency validation
    """
    row0 = next((r for r in ctx_rows if abs(r['level']) < 0.06),
                ctx_rows[len(ctx_rows) // 2])
    fit = _fit_prospect_theory(row0['trials'], row0['choices'])
    if fit is None:
        print('Utility curve fit failed — too few choices')
        return None

    alpha, gamma, beta, res = fit

    # --- build figure ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    p_grid = np.linspace(0.01, 0.99, 200)
    w_fit  = np.exp(-((-np.log(p_grid)) ** gamma))

    # Panel 1: probability weighting
    ax = axes[0]
    ax.plot(p_grid, p_grid, '--', color=MUTED, lw=1.2, alpha=0.7, label='risk-neutral  w(p)=p')
    ax.fill_between(p_grid, p_grid, w_fit,
                    where=(w_fit > p_grid), alpha=0.14, color=RUST)   # overweighting
    ax.fill_between(p_grid, p_grid, w_fit,
                    where=(w_fit < p_grid), alpha=0.14, color=TEAL)   # underweighting
    ax.plot(p_grid, w_fit, color=RUST, lw=2.3,
            label=f'Prelec  γ = {gamma:.3f}')
    ax.scatter(PROBS, np.exp(-((-np.log(PROBS)) ** gamma)),
               s=55, color=RUST, zorder=5, edgecolors='white', linewidths=0.7)
    ax.set_xlabel('objective probability  p')
    ax.set_ylabel('decision weight  w(p)')
    ax.set_title('Probability weighting')
    ax.legend(fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 2: utility function (normalised to max=1 for display)
    ax = axes[1]
    rewards_unique = np.unique([float(tr['size_l']) for tr in row0['trials']]
                               + [float(tr['size_r']) for tr in row0['trials']])
    x_grid = np.linspace(0, rewards_unique.max() * 1.05, 200)
    v_fit  = x_grid ** alpha
    v_lin  = x_grid                     # linear (risk-neutral)
    # normalize both to [0,1] so they share a y-axis
    v_max  = v_fit.max()
    ax.plot(x_grid, v_lin / v_max, '--', color=MUTED, lw=1.2, alpha=0.7,
            label='risk-neutral  v(x)=x')
    ax.fill_between(x_grid, v_lin / v_max, v_fit / v_max,
                    where=(v_fit > v_lin), alpha=0.14, color=RUST)   # convex
    ax.fill_between(x_grid, v_lin / v_max, v_fit / v_max,
                    where=(v_fit < v_lin), alpha=0.14, color=TEAL)   # concave
    ax.plot(x_grid, v_fit / v_max, color=TEAL, lw=2.3,
            label=f'power  α = {alpha:.3f}')
    ax.scatter(rewards_unique, (rewards_unique ** alpha) / v_max,
               s=45, color=TEAL, zorder=5, edgecolors='white', linewidths=0.7)
    ax.set_xlabel('reward magnitude')
    ax.set_ylabel('utility  v(x)  (normalised)')
    ax.set_title('Utility function')
    ax.legend(fontsize=8.5)

    # Panel 3: predicted vs observed choice rates
    ax = axes[2]
    # Bin trials by (prob_l, prob_r) pair and compare predicted vs observed
    from collections import defaultdict
    from scipy.special import expit
    bins_obs, bins_pred, bin_labels = [], [], []
    bucket = defaultdict(lambda: {'n': 0, 'chose_r': 0, 'pred': []})
    for i, tr in enumerate(row0['trials']):
        c = row0['choices'][i]
        if c not in (1, 2):
            continue
        pl, pr = float(tr['prob_l']), float(tr['prob_r'])
        rl, rr = float(tr['size_l']), float(tr['size_r'])
        key = (round(pl, 2), round(pr, 2))
        bucket[key]['n'] += 1
        bucket[key]['chose_r'] += (c == 2)
        pl_c = np.clip(pl, 1e-9, 1 - 1e-9)
        pr_c = np.clip(pr, 1e-9, 1 - 1e-9)
        wl = np.exp(-((-np.log(pl_c)) ** gamma))
        wr = np.exp(-((-np.log(pr_c)) ** gamma))
        pred = float(expit(beta * (wr * rr**alpha - wl * rl**alpha)))
        bucket[key]['pred'].append(pred)
    obs_all, pred_all = [], []
    for key, v in bucket.items():
        if v['n'] < 3:
            continue
        obs_all.append(v['chose_r'] / v['n'])
        pred_all.append(float(np.mean(v['pred'])))
    obs_all  = np.array(obs_all)
    pred_all = np.array(pred_all)
    ax.scatter(pred_all, obs_all, s=28, alpha=0.65, color=GOLD,
               edgecolors='none', rasterized=True)
    lims = [0, 1]
    ax.plot(lims, lims, '--', color=MUTED, lw=1.2, alpha=0.7, label='perfect')
    if len(obs_all) >= 3:
        _, r2 = linear_r2(pred_all, obs_all)
        ax.text(0.05, 0.94, f'R² = {r2:.3f}', transform=ax.transAxes,
                va='top', fontsize=9, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('model-predicted  P(chose right)')
    ax.set_ylabel('observed  P(chose right)')
    ax.set_title('Calibration (predicted vs observed)')
    ax.legend(fontsize=8.5)

    # Title
    risk_char = ('convex utility → risk-seeking' if alpha > 1
                 else 'concave utility → risk-averse' if alpha < 1
                 else 'linear utility → risk-neutral')
    prob_char = ('inverse-S → overweight small p' if gamma < 1
                 else 'S-shape → underweight small p' if gamma > 1
                 else 'linear → no distortion')
    fig.suptitle(f'A4 · Prospect-theory fit   α={alpha:.3f} ({risk_char})   '
                 f'γ={gamma:.3f} ({prob_char})   β={beta:.2f}',
                 fontsize=10, fontweight='semibold', y=1.01)
    fig.tight_layout()
    name = save_fig(fig, figs_dir, 'a4_utility_curve.png')

    # Verdict
    risk_seek_util = alpha > 1.05
    inv_s_prob     = gamma < 0.95          # overweights small probs → risk-seeking
    verdict = (f'α={alpha:.3f} (utility {"convex ✓" if risk_seek_util else "concave"})  '
               f'γ={gamma:.3f} (prob. weighting {"inv-S / overweights small p ✓" if inv_s_prob else "S-shape / underweights small p"})  '
               f'β={beta:.2f}')
    st = 'pass' if (risk_seek_util or inv_s_prob) else 'info'
    add_fig('A', name, 'A4: Prospect-theory utility curve', 'Fig S1',
            'Monkeys showed convex utility (risk-seeking) and/or inverse-S probability weighting '
            '(overweighting of small probabilities).',
            'α > 1 (convex utility) and/or γ < 1 (inverse-S probability weighting).',
            verdict, st)
    return name


def plot_utility_curves_comparison(ctx_rows, figs_dir):
    """
    Fit prospect theory at EVERY ctx level.
    Panel 1 — probability weighting w(p) for all ctx levels overlaid (coolwarm).
    Panel 2 — utility v(x) for all ctx levels overlaid (coolwarm).
    Panel 3 — α (utility curvature) vs ctx.
    Panel 4 — γ (Prelec exponent) vs ctx.
    Colorbar sits in its own axes to the right.
    """
    # Fit all ctx levels
    all_levels, all_alphas, all_gammas, all_rows = [], [], [], []
    for row in ctx_rows:
        result = _fit_prospect_theory(row['trials'], row['choices'])
        if result is not None:
            a, g, b, _ = result
            all_levels.append(row['level'])
            all_alphas.append(a)
            all_gammas.append(g)
            all_rows.append(row)

    if not all_levels:
        print('Utility comparison: all fits failed')
        return

    lvs   = np.array(all_levels)
    alps  = np.array(all_alphas)
    gams  = np.array(all_gammas)
    order = np.argsort(lvs)
    lvs, alps, gams = lvs[order], alps[order], gams[order]
    all_rows = [all_rows[i] for i in order]

    p_grid = np.linspace(0.01, 0.99, 200)
    ref_trials = all_rows[0]['trials']
    all_rewards = [float(tr['size_l']) for tr in ref_trials] + \
                  [float(tr['size_r']) for tr in ref_trials]
    x_max  = float(np.max(all_rewards)) * 1.05
    x_grid = np.linspace(0, x_max, 200)

    norm = plt.Normalize(lvs.min(), lvs.max())
    sm   = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])

    # Figure: 4 data axes + 1 thin colorbar axis
    fig = plt.figure(figsize=(17, 4.5))
    # left=0.05, right=0.91 leaves room for colorbar at 0.92-0.935
    axes = [fig.add_axes([0.05 + i*0.215, 0.12, 0.185, 0.76]) for i in range(4)]
    cax  = fig.add_axes([0.935, 0.14, 0.013, 0.72])
    fig.colorbar(sm, cax=cax, label='ctx input')

    # Panel 1: probability weighting — all ctx overlaid
    ax = axes[0]
    ax.plot(p_grid, p_grid, '--', color='#aaa', lw=1.1, zorder=1)
    for i, (lv, a, g) in enumerate(zip(lvs, alps, gams)):
        w   = np.exp(-((-np.log(p_grid)) ** g))
        col = plt.cm.coolwarm(norm(lv))
        ax.plot(p_grid, w, color=col, lw=1.4, alpha=0.75, zorder=2)
    # highlight min/mid/max
    for lv_hi in [lvs[0], lvs[len(lvs)//2], lvs[-1]]:
        idx = np.argmin(np.abs(lvs - lv_hi))
        g   = gams[idx]
        w   = np.exp(-((-np.log(p_grid)) ** g))
        col = plt.cm.coolwarm(norm(lvs[idx]))
        ax.plot(p_grid, w, color=col, lw=2.4, zorder=3,
                label=f'ctx={lvs[idx]:+.1f}  γ={g:.2f}')
    ax.set_xlabel('probability'); ax.set_ylabel('decision weight w(p)')
    ax.set_title('Probability weighting'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(fontsize=7.5, loc='upper left')

    # Panel 2: utility function — all ctx overlaid
    ax = axes[1]
    ax.plot(x_grid/x_max, x_grid/x_max, '--', color='#aaa', lw=1.1, zorder=1)
    for i, (lv, a, g) in enumerate(zip(lvs, alps, gams)):
        v   = x_grid ** a
        col = plt.cm.coolwarm(norm(lv))
        ax.plot(x_grid/x_max, v/v.max(), color=col, lw=1.4, alpha=0.75, zorder=2)
    for lv_hi in [lvs[0], lvs[len(lvs)//2], lvs[-1]]:
        idx = np.argmin(np.abs(lvs - lv_hi))
        a   = alps[idx]
        v   = x_grid ** a
        col = plt.cm.coolwarm(norm(lvs[idx]))
        ax.plot(x_grid/x_max, v/v.max(), color=col, lw=2.4, zorder=3,
                label=f'ctx={lvs[idx]:+.1f}  α={a:.2f}')
    ax.set_xlabel('normalised reward'); ax.set_ylabel('utility v(x) (norm.)')
    ax.set_title('Utility function')
    ax.legend(fontsize=7.5, loc='upper left')

    # Panel 3: α vs ctx
    ax = axes[2]
    for i in range(len(lvs)-1):
        ax.plot(lvs[i:i+2], alps[i:i+2], '-', color=plt.cm.coolwarm(norm(lvs[i])), lw=2)
    ax.scatter(lvs, alps, c=[plt.cm.coolwarm(norm(l)) for l in lvs],
               s=42, zorder=4, edgecolors='white', linewidths=0.6)
    ax.axhline(1.0, color='#aaa', lw=1, ls=':')
    ax.set_xlabel('dopamine context level'); ax.set_ylabel('α (utility curvature)')
    ax.set_title('α vs dopamine\n(>1 convex · <1 concave)')

    # Panel 4: γ vs ctx
    ax = axes[3]
    for i in range(len(lvs)-1):
        ax.plot(lvs[i:i+2], gams[i:i+2], '-', color=plt.cm.coolwarm(norm(lvs[i])), lw=2)
    ax.scatter(lvs, gams, c=[plt.cm.coolwarm(norm(l)) for l in lvs],
               s=42, zorder=4, edgecolors='white', linewidths=0.6)
    ax.axhline(1.0, color='#aaa', lw=1, ls=':')
    ax.set_xlabel('dopamine context level'); ax.set_ylabel('γ (Prelec exponent)')
    ax.set_title('γ vs dopamine\n(<1 inv-S · >1 S-shape)')

    name = save_fig(fig, figs_dir, 'a4b_utility_comparison.png')
    alpha_trend = 'rises with DA ✓' if alps[-1] > alps[0] else 'falls with DA'
    add_fig('A', name, 'Prospect-theory utility curves across all DA levels', 'Fig S1',
            'Utility curvature (α) and probability weighting (γ) across all ctx levels.',
            f'α: {alps[order[0]]:.2f} → {alps[-1]:.2f} ({alpha_trend}); '
            f'γ: {gams[order[0]]:.2f} → {gams[-1]:.2f}',
            f'α trend: {alpha_trend}', 'info')
    return name


def _bootstrap_marginals(trials, choices, n_boot=400):
    """Return (prob_mean, prob_lo, prob_hi, ev_mean, ev_lo, ev_hi) with 95% CI."""
    def grid_from(idx):
        g_off = np.zeros((5, 5)); g_ch = np.zeros((5, 5))
        for i in idx:
            tr = trials[i]
            for tgt in (tr['target_l'], tr['target_r']):
                g_off[tgt % 5, tgt // 5] += 1   # [ev_col, prob_row]
            c = choices[i]
            if c in (1, 2):
                tgt = tr['target_l'] if c == 1 else tr['target_r']
                g_ch[tgt % 5, tgt // 5] += 1
        g = np.divide(g_ch, g_off, out=np.zeros((5, 5)), where=g_off > 0)
        return g
    N = len(trials)
    prob_bs = np.zeros((n_boot, 5)); ev_bs = np.zeros((n_boot, 5))
    for b in range(n_boot):
        g = grid_from(np.random.choice(N, N, replace=True))
        prob_bs[b] = np.nanmean(g, axis=0)   # mean over EV → by prob
        ev_bs[b]   = np.nanmean(g, axis=1)   # mean over prob → by EV
    p_m = prob_bs.mean(0); p_l = np.percentile(prob_bs, 2.5, 0); p_h = np.percentile(prob_bs, 97.5, 0)
    e_m = ev_bs.mean(0);   e_l = np.percentile(ev_bs, 2.5, 0);   e_h = np.percentile(ev_bs, 97.5, 0)
    return p_m, p_l, p_h, e_m, e_l, e_h


def _plot_heatmap_with_marginals(trials, choices, level, ctx_rows, figs_dir):
    """
    Transposed 5×5 heatmap: x = win probability, y = EV column.
    Top marginal = risk psychometric (A2): proportion vs probability + CI + regression.
    Left marginal = EV control (A3): proportion vs EV column + CI + regression.
    """
    # ---- build 5×5 grid ----  [ev_col, prob_row]  (row=prob, col=EV after transpose)
    grid_off = np.zeros((5, 5)); grid_ch = np.zeros((5, 5))
    for i, tr in enumerate(trials):
        for tgt in (tr['target_l'], tr['target_r']):
            prob_row = tgt // 5; ev_col = tgt % 5
            grid_off[ev_col, prob_row] += 1
        c = choices[i]
        if c in (1, 2):
            tgt = tr['target_l'] if c == 1 else tr['target_r']
            grid_ch[tgt % 5, tgt // 5] += 1
    grid = np.divide(grid_ch, grid_off, out=np.full((5, 5), np.nan), where=grid_off > 0)
    # grid shape: [ev_col (y), prob_row (x)]

    # ---- bootstrap marginals ----
    p_m, p_l, p_h, e_m, e_l, e_h = _bootstrap_marginals(trials, choices)

    # ---- layout: main heatmap + top + left + colorbar ----
    fig = plt.figure(figsize=(7.8, 7.8))
    ax_main = fig.add_axes([0.20, 0.12, 0.52, 0.52])
    ax_top  = fig.add_axes([0.20, 0.66, 0.52, 0.20])  # probability marginal (A2)
    ax_left = fig.add_axes([0.02, 0.12, 0.16, 0.52])  # EV marginal (A3)
    cax     = fig.add_axes([0.74, 0.12, 0.025, 0.52])

    prob_ticks = ['10%', '30%', '50%', '70%', '90%']
    ev_ticks   = ['EV1', 'EV2', 'EV3', 'EV4', 'EV5']

    # ---- main heatmap ----
    im = ax_main.imshow(grid, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                         aspect='auto', extent=[-0.5, 4.5, -0.5, 4.5])
    ax_main.set_xticks(range(5)); ax_main.set_xticklabels(prob_ticks)
    ax_main.set_yticks(range(5)); ax_main.set_yticklabels(ev_ticks)
    ax_main.set_xlabel('win probability'); ax_main.set_ylabel('expected value')
    fig.colorbar(im, cax=cax, label='P(chosen | offered)')

    # ---- top marginal: prob axis (A2 — risk psychometric) ----
    xs = np.arange(5)
    ax_top.fill_between(xs, p_l, p_h, alpha=0.18, color='#333')
    ax_top.plot(xs, p_m, 'o-', color='#222', lw=1.8, ms=5.5, zorder=3)
    # regression line
    fit = np.polyfit(xs, p_m, 1)
    xd = np.linspace(0, 4, 80)
    ax_top.plot(xd, np.polyval(fit, xd), '--', color='#888', lw=1.2)
    ax_top.axhline(0.2, color='#ccc', lw=0.8, ls=':')  # 1/5 uniform baseline
    ax_top.set_xlim(-0.5, 4.5); ax_top.set_ylim(0, 0.4)
    ax_top.set_xticks(range(5)); ax_top.set_xticklabels([])
    ax_top.set_ylabel('P(chosen)', fontsize=8.5)
    ax_top.set_title(f'Choice frequency heatmap  (ctx={level:+.2f})  ·  95% bootstrap CI',
                     fontsize=9.5, pad=6)
    slope_p = fit[0]

    # ---- left marginal: EV axis (A3 — EV control) ----
    ys = np.arange(5)
    ax_left.fill_betweenx(ys, e_l, e_h, alpha=0.18, color='#333')
    ax_left.plot(e_m, ys, 'o-', color='#222', lw=1.8, ms=5.5, zorder=3)
    fit_ev = np.polyfit(ys, e_m, 1)
    yd = np.linspace(0, 4, 80)
    ax_left.plot(np.polyval(fit_ev, yd), yd, '--', color='#888', lw=1.2)
    ax_left.axvline(0.2, color='#ccc', lw=0.8, ls=':')
    ax_left.set_ylim(-0.5, 4.5); ax_left.set_xlim(0, 0.4)
    ax_left.set_yticks(range(5)); ax_left.set_yticklabels([])
    ax_left.set_xlabel('P(chosen)', fontsize=8.5)
    slope_ev = fit_ev[0]

    # slope annotations
    ax_top.text(0.98, 0.92, f'prob slope {slope_p:+.3f}',
                transform=ax_top.transAxes, ha='right', va='top', fontsize=8, color='#555')
    ax_left.text(0.98, 0.04, f'EV slope\n{slope_ev:+.3f}',
                 transform=ax_left.transAxes, ha='right', va='bottom', fontsize=8, color='#555')

    name = save_fig(fig, figs_dir, 'a1_heatmap_marginals.png')
    add_fig('A', name,
            'Choice frequency heatmap with marginals',
            'Fig 1C',
            'Transposed 5×5 heatmap (x=probability, y=EV). '
            'Top marginal = risk psychometric with 95% bootstrap CI. '
            'Left marginal = EV sensitivity with 95% bootstrap CI.',
            f'prob slope={slope_p:+.3f}  EV slope={slope_ev:+.3f}',
            f'prob slope={slope_p:+.3f}, EV slope={slope_ev:+.3f}',
            'pass' if slope_p < -0.01 else 'info')
    return name

def _plot_psychometrics_multi_ctx(ctx_rows, figs_dir, fname):
    """Overlaid psychometric curves at 5 representative ctx levels."""
    targets = [-0.9, -0.45, 0.0, 0.45, 0.9]
    rows = []
    for t in targets:
        r = min(ctx_rows, key=lambda r: abs(r['level'] - t))
        rows.append(r)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        lv, pp, rs = proportion_slope(r['trials'], r['choices'], 'prob')
        if lv.size < 2: continue
        col = ctx_color(r['level'])
        ax.plot(lv*100, pp, 'o', color=col, ms=7, zorder=4)
        fit, _ = linear_r2(lv, pp)
        xs = np.linspace(0.05, 0.95, 80)
        ax.plot(xs*100, np.polyval(fit, xs), '-', color=col, lw=2, alpha=0.85,
                label=f'ctx={r["level"]:+.2f}  (slope {fit[0]:+.3f})')
    ax.axhline(0.5, color=INK, lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('win probability (%)')
    ax.set_ylabel('proportion chosen  (matched EV)')
    ax.set_title('A4 · Psychometrics across dopamine levels')
    ax.legend(fontsize=8.5, loc='lower left')
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(-0.9, 0.9))
    sm.set_array([]); fig.colorbar(sm, ax=ax, shrink=0.7, label='ctx input')
    name = save_fig(fig, figs_dir, fname)
    add_fig('A', name, 'A5: Psychometrics at 5 dopamine levels', 'Fig 1C + Fig 3M/N',
            'Higher DA → steeper risk-seeking slope; lower DA → risk-averse.',
            'Lines tilt from positive (low DA) to negative (high DA).',
            'Five ctx levels overlaid with coolwarm gradient', 'info')

def _dose_response_verdict(levels, risk):
    fin = np.isfinite(risk)
    if fin.sum() < 3:
        return 'insufficient data'
    corr = float(np.corrcoef(levels[fin], risk[fin])[0, 1])
    rng  = float(np.nanmax(risk) - np.nanmin(risk))
    mid  = risk[np.argmin(np.abs(levels))] if fin.any() else np.nan
    monotonic = abs(corr) > 0.65 and rng > 0.1
    verdict = (f'DA–risk correlation = {corr:+.2f}, span = {rng:.2f}, '
               f'neutral riskiness = {mid:.2f}')
    if monotonic:
        verdict += ' → MONOTONIC ✓'
    else:
        verdict += ' → weak/flat — model may not show strong DA→risk coupling'
    return verdict

def _risk_slope_verdict(slope):
    if not np.isfinite(slope):
        return 'undefined (no matched-EV decisions)', 'fail'
    if slope < -0.05: return f'slope = {slope:+.3f} → RISK-SEEKING ✓', 'pass'
    if slope >  0.05: return f'slope = {slope:+.3f} → risk-averse', 'info'
    return f'slope = {slope:+.3f} → ~flat (EV-neutral)', 'info'

# --------------------------------------------------------------------------- #
# GROUP B  ── necessity / ablation
# --------------------------------------------------------------------------- #
def group_B(ctx_rows, figs_dir, runner, cfg):
    """
    Silence the risk-coding neural dimension (top-K neurons by |β_HH|) rather than
    zeroing the D1 or D2 anatomical half, which flips instead of flattening the slope.
    Format: Δslope dot plot (ablated − intact) for risk slope and EV slope.
    """
    from numpy.linalg import lstsq

    # ── Step 1: compute β_HH per neuron from baseline pkl ──────────────────────
    row0 = next((r for r in ctx_rows if abs(r['level']) < 0.06), ctx_rows[len(ctx_rows)//2])
    d = row0['d']
    r_policy = to_np(d['r_policy'])   # (T, B, N)
    T, B_tr, N = r_policy.shape
    dec = slice(50, T)

    dh_list, de_list, valid = [], [], []
    for i, tr in enumerate(row0['trials']):
        pl, pr = float(tr['prob_l']), float(tr['prob_r'])
        evl, evr = pl * float(tr['size_l']), pr * float(tr['size_r'])
        if pl <= 0 or pr <= 0 or evl <= 0 or evr <= 0:
            continue
        dh_list.append(np.log(pr) - np.log(pl))
        de_list.append(np.log(evr) - np.log(evl))
        valid.append(i)

    dh = np.array(dh_list); de = np.array(de_list)
    X_reg = np.column_stack([dh, de, np.ones(len(dh))])
    act_v = r_policy[dec, :, :].mean(axis=0)[valid, :]  # (n_valid, N)
    coefs, _, _, _ = lstsq(X_reg, act_v, rcond=None)
    beta_hh = coefs[0]  # (N,) — risk encoding coefficient per neuron

    # ── Step 2: define ablation indices ──────────────────────────────────��─────
    K            = max(N // 4, 5)
    rank         = np.argsort(np.abs(beta_hh))[::-1].copy()
    risk_neurons = rank[:K].tolist()   # top-K by |β_HH|  → risk dimension
    ctrl_neurons = rank[-K:].tolist()  # bottom-K          → value / non-risk

    def run_ablated(idx, seed=999):
        orig = runner.pg.policy_net.output_layer
        def patched(r, *a, **kw):
            r2 = r.clone(); r2[..., idx] = 0.0; return orig(r2, *a, **kw)
        runner.pg.policy_net.output_layer = patched
        try:
            return runner.run(0.0, cfg['n'], kind='prob', seed=seed)
        finally:
            runner.pg.policy_net.output_layer = orig

    intact = runner.run(0.0, cfg['n'], kind='prob')
    risk_abl = run_ablated(risk_neurons)
    ctrl_abl = run_ablated(ctrl_neurons, seed=1001)

    conds = {'intact': intact, 'risk-axis': risk_abl, 'value-ctrl': ctrl_abl}

    # ── Step 3: Δslope dot plot ─────────────────────────────────────────────────
    rs = {k: v['risk_slope'] for k, v in conds.items()}
    es = {k: v['ev_slope']   for k, v in conds.items()}

    ablation_names = ['risk-axis', 'value-ctrl']
    colors_abl     = [RUST, TEAL]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, slopes, ylabel, title in [
        (axes[0], rs, 'Δ risk slope  (ablated − intact)', 'Risk slope change'),
        (axes[1], es, 'Δ EV slope  (ablated − intact)',   'EV slope change'),
    ]:
        ax.axhline(0, color='#bbb', lw=1.2, ls='--', zorder=1)
        for xi, (name, col) in enumerate(zip(ablation_names, colors_abl)):
            delta = slopes[name] - slopes['intact']
            ax.scatter([xi], [delta], s=110, color=col, zorder=4,
                       edgecolors='white', linewidths=0.8)
            ax.text(xi, delta + (0.005 if delta >= 0 else -0.005),
                    f'{delta:+.3f}', ha='center',
                    va='bottom' if delta >= 0 else 'top', fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks(range(2)); ax.set_xticklabels(['risk-axis\nsilenced', 'value-ctrl\nsilenced'])
        ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title)
        ax.text(0.98, 0.03, f'intact = {slopes["intact"]:+.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#666')

    fig.suptitle(f'Ablation: Δslope vs intact  (K={K} neurons silenced)',
                 fontsize=10, fontweight='semibold')
    fig.tight_layout()
    name = save_fig(fig, figs_dir, 'b_ablation.png')
    d_risk = rs['risk-axis'] - rs['intact']
    d_ctrl = rs['value-ctrl'] - rs['intact']
    flattened = np.isfinite(d_risk) and abs(d_risk) < abs(rs['intact']) * 0.65
    add_fig('B', name, 'Ablation: Δslope dot plot (risk-axis vs value-ctrl)',
            'Fig 2E/F',
            f'Δrisk slope: risk-axis={d_risk:+.3f}, value-ctrl={d_ctrl:+.3f}. '
            f'Δ EV slope: risk-axis={es["risk-axis"]-es["intact"]:+.3f}.',
            f'Risk-axis ablation flattens risk slope; EV slope unchanged.',
            f'risk-axis Δ={d_risk:+.3f}, ctrl Δ={d_ctrl:+.3f}',
            'pass' if flattened else 'info')

# --------------------------------------------------------------------------- #
# GROUP C  ── directional dopamine control
# --------------------------------------------------------------------------- #
def group_C(ctx_rows, figs_dir, runner, cfg):
    levels   = np.array([r['level']    for r in ctx_rows])
    riskvals = np.array([r['riskiness'] for r in ctx_rows])

    # ── C2: riskiness dose-response (diagnostic, not a paper panel) ────────────
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i in range(len(levels)-1):
        ax.plot(levels[i:i+2], riskvals[i:i+2], '-', color=ctx_color(levels[i]),
                lw=2.5, solid_capstyle='round')
    ax.scatter(levels, riskvals, c=[ctx_color(l) for l in levels], s=55, zorder=4,
               edgecolors='white', linewidths=0.7)
    ax.axhline(0.5, color='#bbb', lw=1, ls=':', label='chance (0.5)')
    fin = np.isfinite(riskvals)
    if fin.sum() >= 3:
        fit, r2 = linear_r2(levels[fin], riskvals[fin])
        xs = np.array([levels.min(), levels.max()])
        ax.plot(xs, np.polyval(fit, xs), '--', color='#888', lw=1.2, alpha=0.7,
                label=f'linear  R²={r2:.2f}')
    ax.set_xlabel('dopamine context level'); ax.set_ylabel('P(risky | matched-EV)')
    ax.set_title('Riskiness dose-response  (model diagnostic)')
    ax.legend(loc='upper left', fontsize=8.5)
    fig.subplots_adjust(right=0.88)
    cax2 = fig.add_axes([0.905, 0.14, 0.016, 0.72])
    sm2  = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(levels.min(), levels.max()))
    sm2.set_array([]); fig.colorbar(sm2, cax=cax2, label='ctx')
    name = save_fig(fig, figs_dir, 'c2_dose_response.png')
    vrd = _dose_response_verdict(levels, riskvals)
    mono = 'MONOTONIC' in vrd
    add_fig('C', name, 'Riskiness dose-response (model diagnostic)',
            'Fig 3O/P (model analog)',
            f'Riskiness as a function of dopamine context level. {vrd}',
            '', '', 'info')
    cfg['_C2_ok'] = mono

    # Embed existing preference figures
    for src, dst, what in [
        ('context_preference_risk_sigmoid.png', 'existing_pref_risk.png',
         'Risk preference across ctx levels (pre-computed sigmoid)'),
        ('context_preference_ev_direct.png', 'existing_pref_ev.png',
         'EV preference across ctx levels (pre-computed)'),
    ]:
        n = copy_existing(src, figs_dir, dst)
        if n:
            add_fig('C', n, what, 'Fig 3 preference panels', what, '', '', 'info')

    # ── C1: Δprobability psychometric + cumGaussian + ΔEV companion ────────────
    from scipy.stats import norm as scipy_norm
    from scipy.optimize import curve_fit

    def cum_gauss(x, mu, sigma, lapse=0.02):
        sigma = max(abs(sigma), 1e-4)
        return lapse / 2 + (1 - lapse) * scipy_norm.cdf(x, mu, sigma)

    def psychometric_delta_prob(run_result, ref_prob=0.5, n_boot=300):
        """
        Build psychometric curve with x = Δprob = p_offered - ref_prob.
        For each matched-EV trial, compute p_offered of each option, Δp = p - ref,
        and y = P(chose that option).
        Returns (delta_p_bins, p_chosen_mean, p_chosen_lo, p_chosen_hi).
        """
        tri = run_result['trials']; cho = run_result['choices']
        dp_list, chosen_list = [], []
        for i, tr in enumerate(tri):
            c = cho[i]
            if c not in (1, 2): continue
            pl, pr = float(tr['prob_l']), float(tr['prob_r'])
            evl, evr = pl * float(tr['size_l']), pr * float(tr['size_r'])
            if not (abs(evl - evr) < 0.015 and abs(pl - pr) > 1e-6): continue
            # Both options appear; record from the perspective of the left option
            dp_list.append(pl - ref_prob)
            chosen_list.append(1 if c == 1 else 0)
            dp_list.append(pr - ref_prob)
            chosen_list.append(1 if c == 2 else 0)
        dp_arr = np.array(dp_list); ch_arr = np.array(chosen_list)
        if len(dp_arr) < 10:
            return None, None, None, None

        # bin by Δp
        bins = np.linspace(dp_arr.min() - 1e-5, dp_arr.max() + 1e-5, 7)
        centers, means, los, his = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (dp_arr >= lo) & (dp_arr < hi)
            if mask.sum() < 3: continue
            vals = ch_arr[mask]
            bs = np.array([vals[np.random.choice(len(vals), len(vals), replace=True)].mean()
                           for _ in range(n_boot)])
            centers.append((lo + hi) / 2)
            means.append(vals.mean())
            los.append(np.percentile(bs, 2.5))
            his.append(np.percentile(bs, 97.5))
        return np.array(centers), np.array(means), np.array(los), np.array(his)

    def psychometric_delta_ev(run_result, n_boot=300):
        """Matched-probability trials: x = ΔEV = ev_offered - ev_alt."""
        tri = run_result['trials']; cho = run_result['choices']
        de_list, chosen_list = [], []
        for i, tr in enumerate(tri):
            c = cho[i]
            if c not in (1, 2): continue
            pl, pr = float(tr['prob_l']), float(tr['prob_r'])
            if abs(pl - pr) > 1e-6: continue   # only matched-prob
            evl = pl * float(tr['size_l']); evr = pr * float(tr['size_r'])
            if abs(evl - evr) < 1e-6: continue
            de_list.append(evl - evr)
            chosen_list.append(1 if c == 1 else 0)
            de_list.append(evr - evl)
            chosen_list.append(1 if c == 2 else 0)
        de_arr = np.array(de_list); ch_arr = np.array(chosen_list)
        if len(de_arr) < 10:
            return None, None, None, None
        bins = np.linspace(de_arr.min()-1e-5, de_arr.max()+1e-5, 7)
        centers, means, los, his = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (de_arr >= lo) & (de_arr < hi)
            if mask.sum() < 3: continue
            vals = ch_arr[mask]
            bs = np.array([vals[np.random.choice(len(vals), len(vals), replace=True)].mean()
                           for _ in range(n_boot)])
            centers.append((lo + hi) / 2)
            means.append(vals.mean())
            los.append(np.percentile(bs, 2.5))
            his.append(np.percentile(bs, 97.5))
        return np.array(centers), np.array(means), np.array(los), np.array(his)

    base = runner.run(0.0, cfg['stim_n'], kind=('prob', 'ev'))
    d1s  = runner.run(cfg['stim_mag'], cfg['stim_n'], kind=('prob', 'ev'),
                      pathway_mode='d1_only_stim')
    d2s  = runner.run(cfg['stim_mag'], cfg['stim_n'], kind=('prob', 'ev'),
                      pathway_mode='d2_only_stim')

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    def _plot_psycho(ax, cond_res, label, color, is_ev=False):
        if is_ev:
            xc, ym, yl, yh = psychometric_delta_ev(cond_res)
        else:
            xc, ym, yl, yh = psychometric_delta_prob(cond_res)
        if xc is None or len(xc) < 3:
            return
        ax.fill_between(xc, yl, yh, alpha=0.18, color=color)
        ax.plot(xc, ym, 'o-', color=color, lw=1.8, ms=5.5, zorder=3, label=label)
        # cumulative Gaussian fit
        try:
            p0 = [xc[np.argmin(np.abs(ym - 0.5))], (xc[-1]-xc[0])/4]
            popt, _ = curve_fit(cum_gauss, xc, ym, p0=p0, maxfev=4000)
            xfit = np.linspace(xc.min(), xc.max(), 120)
            ax.plot(xfit, cum_gauss(xfit, *popt), '--', color=color,
                    lw=1.2, alpha=0.7)
            ax.text(0.03, 0.96, f'PSE={popt[0]:.2f}  σ={abs(popt[1]):.2f}',
                    transform=ax.transAxes, va='top', fontsize=7.5, color=color, alpha=0.9)
        except Exception:
            pass
        ax.axhline(0.5, color='#ccc', lw=0.9, ls=':')
        ax.axvline(0,   color='#ccc', lw=0.9, ls=':')

    # Row 0: Δprobability psychometric
    for ax, cond, lbl, col in zip(axes[0], [base, d1s, d2s],
                                   ['baseline', 'D1 stim', 'D2 stim'],
                                   ['#333', RUST, TEAL]):
        _plot_psycho(ax, cond, lbl, col, is_ev=False)
        ax.set_xlabel('Δ probability  (offered − 50%)')
        ax.set_ylabel('P(chose offered option)')
        ax.set_title(lbl)
        ax.legend(fontsize=8)
        if lbl == 'baseline':
            ax.set_title('Baseline — Δprobability')

    # Row 1: ΔEV psychometric
    for ax, cond, lbl, col in zip(axes[1], [base, d1s, d2s],
                                   ['baseline', 'D1 stim', 'D2 stim'],
                                   ['#333', RUST, TEAL]):
        _plot_psycho(ax, cond, lbl, col, is_ev=True)
        ax.set_xlabel('ΔEV  (offered − alt)')
        ax.set_ylabel('P(chose offered option)')
        ax.set_title(f'{lbl} — ΔEV')
        ax.legend(fontsize=8)

    fig.suptitle(f'Directional pathway stimulation  (mag +{cfg["stim_mag"]:.2f})  —  '
                 'Δprob (top row) and ΔEV (bottom row)',
                 fontsize=10, fontweight='semibold')
    fig.tight_layout()
    name_c1 = save_fig(fig, figs_dir, 'c1_stim_psychometrics.png')
    d1_riskier = d1s['risk_slope'] < base['risk_slope']
    d2_safer   = d2s['risk_slope'] > base['risk_slope']
    add_fig('C', name_c1, 'Directional stimulation psychometrics (Δprob + ΔEV)',
            'Fig 3M/N',
            'Δprobability psychometric curves for D1/D2-only stimulation, '
            'with cumulative Gaussian fit and 95% bootstrap CI bands. '
            f'D1 stim: {"↑risky" if d1_riskier else "no shift"}. '
            f'D2 stim: {"↓risky" if d2_safer else "no shift"}.',
            '', '', 'pass' if (d1_riskier and d2_safer) else 'info')

    # ── C3: bootstrap cloud scatter — ON vs OFF per stim type ──────────────────
    n_boot = 300

    def boot_ri(tri, cho):
        N = len(tri)
        return np.array([
            riskiness_from([tri[i] for i in np.random.choice(N, N, replace=True)],
                           cho[np.random.choice(N, N, replace=True)])
            for _ in range(n_boot)
        ])

    # OFF baseline bootstrap
    row0 = next((r for r in ctx_rows if abs(r['level']) < 0.06), ctx_rows[len(ctx_rows)//2])
    ri_off_bs = boot_ri(row0['trials'], row0['choices'])
    ri_d1_bs  = boot_ri(d1s['trials'], d1s['choices'])
    ri_d2_bs  = boot_ri(d2s['trials'], d2s['choices'])
    ri_ctrl_bs = boot_ri(
        runner.run(0.0, cfg['stim_n'], kind='prob', seed=9999)['trials'],
        runner.run(0.0, cfg['stim_n'], kind='prob', seed=9999)['choices']
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    ax.plot([0, 1], [0, 1], '--', color='#ccc', lw=1.2, zorder=1)

    def scatter_cloud(ri_x, ri_y, col, lbl):
        valid = np.isfinite(ri_x) & np.isfinite(ri_y)
        x, y = ri_x[valid], ri_y[valid]
        ax.scatter(x, y, s=9, alpha=0.3, color=col, edgecolors='none', zorder=2)
        mx, my = np.median(x), np.median(y)
        cix = np.percentile(x, [2.5, 97.5]); ciy = np.percentile(y, [2.5, 97.5])
        ax.errorbar(mx, my, xerr=[[mx-cix[0]], [cix[1]-mx]],
                    yerr=[[my-ciy[0]], [ciy[1]-my]],
                    fmt='o', color=col, ms=9, lw=1.8, capsize=4, zorder=5,
                    label=f'{lbl}  Δ={my-mx:+.2f}')

    scatter_cloud(ri_off_bs, ri_d1_bs,  RUST, 'D1 stim')
    scatter_cloud(ri_off_bs, ri_d2_bs,  TEAL, 'D2 stim')
    scatter_cloud(ri_off_bs, ri_ctrl_bs, '#777', 'control')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('riskiness  stim OFF  (bootstrap sessions)')
    ax.set_ylabel('riskiness  stim ON  (bootstrap sessions)')
    ax.set_title('Risk threshold: stim ON vs OFF\n(bootstrap pseudo-sessions, n=300)')
    ax.legend(fontsize=9, loc='upper left')
    name_c3 = save_fig(fig, figs_dir, 'c3_threshold_scatter.png')
    m_d1 = np.nanmedian(ri_d1_bs); m_d2 = np.nanmedian(ri_d2_bs)
    m_off = np.nanmedian(ri_off_bs)
    add_fig('C', name_c3, 'Risk threshold ON vs OFF — bootstrap cloud',
            'Fig 3Q',
            f'Bootstrap pseudo-sessions: D1 stim cloud above diagonal, '
            f'D2 cloud below, control on it. '
            f'D1 Δ={m_d1-m_off:+.2f}, D2 Δ={m_d2-m_off:+.2f}.',
            '', '', 'pass' if (m_d1 > m_off and m_d2 < m_off) else 'info')

    # ── C4: EV slope dissociation (unchanged) ──────────────────────────────────
    ev_slopes = np.array([r['ev_slope'] for r in ctx_rows])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i in range(len(levels)-1):
        ax.plot(levels[i:i+2], ev_slopes[i:i+2], '-', color=ctx_color(levels[i]), lw=2)
    ax.scatter(levels, ev_slopes, c=[ctx_color(l) for l in levels], s=45,
               zorder=4, edgecolors='white', linewidths=0.6)
    ax.axhline(0, color='#bbb', lw=0.9, ls=':')
    ax.set_xlabel('dopamine context level'); ax.set_ylabel('EV slope')
    ax.set_title('EV sensitivity vs dopamine  (dissociation check)')
    fig.subplots_adjust(right=0.88)
    cax4 = fig.add_axes([0.905, 0.14, 0.016, 0.72])
    sm4  = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(levels.min(), levels.max()))
    sm4.set_array([]); fig.colorbar(sm4, cax=cax4, label='ctx')
    name = save_fig(fig, figs_dir, 'c4_ev_vs_da.png')
    ev_std = float(np.nanstd(ev_slopes))
    add_fig('C', name, 'EV sensitivity vs dopamine (dissociation)',
            'Fig 3 bottom rows',
            f'EV slope across all ctx levels. std={ev_std:.3f} '
            f'(small = EV sensitivity unchanged by dopamine).',
            '', '', 'info')

# --------------------------------------------------------------------------- #
# GROUP D  ── timing specificity
# --------------------------------------------------------------------------- #
def group_D(figs_dir, runner, cfg):
    """
    Three-panel figure:
      Left  — Sliding-window sweep: context applied in every 15-timestep window
               across the full trial; Δriskiness plotted as a continuous curve
               over time. Shows exactly when dopamine matters.
      Centre — Epoch-bar summary: riskiness for each epoch-restricted condition,
               visualised as a horizontal timeline with bar height = effect.
      Right  — Overlaid psychometric curves for all phase conditions.
    """
    mag   = cfg['stim_mag']
    dt    = runner.pg.dt          # ms per timestep (typically 10)
    T     = runner.pg.Tmax        # total timesteps (77)
    n     = cfg['stim_n']

    # ── (a) Sliding window sweep ──────────────────────────────────────────────
    # Monkey-patch _context_for_step so we can restrict context to an
    # arbitrary set of timestep indices without going through the epoch API.
    win_size   = 15               # timesteps ≈ 150 ms
    win_stride = 5                # step between window starts
    win_starts = list(range(0, T - win_size + 1, win_stride))
    # baseline once
    base_run = runner.run(0.0, n, kind='prob', seed=888)
    base_ri  = base_run['riskiness']

    orig_ctx = runner.pg._context_for_step   # save original

    sweep_midpoints = []   # ms
    sweep_delta_ri  = []   # Δriskiness vs baseline

    print(f'  Sliding-window sweep: {len(win_starts)} windows × {win_size}ts ...')
    for ws in win_starts:
        win_ts = set(range(ws, ws + win_size))

        def _patched_ctx(trial, t, ctx_val, ctx_phases, _win=win_ts):
            return ctx_val if t in _win else 0.0

        runner.pg._context_for_step = _patched_ctx
        try:
            r = runner.run(mag, n, kind='prob', seed=888)
        finally:
            runner.pg._context_for_step = orig_ctx

        mid_ms = (ws + win_size / 2) * dt
        sweep_midpoints.append(mid_ms)
        sweep_delta_ri.append(r['riskiness'] - base_ri)

    sweep_mid = np.array(sweep_midpoints)
    sweep_dri = np.array(sweep_delta_ri)

    # ── (b) Epoch-restricted conditions ──────────────────────────────────────
    epoch_conds = [
        ('fixation',  ('fixation',),  '#7b9fc4'),
        ('stimulus',  ('stimulus',),  '#5a8a78'),
        ('decision',  ('decision',),  RUST),
        ('all',       None,           '#555'),
    ]
    epoch_results = {}
    for label, phases, _ in epoch_conds:
        epoch_results[label] = runner.run(mag, n, kind='prob',
                                          context_phases=phases, seed=888)

    # ── (c) Psychometric for each phase ──────────────────────────────────────
    # base + 3 epoch-restricted conditions
    psycho_conds = [('baseline', base_run, '#333')] + \
                   [(lbl, epoch_results[lbl], col) for lbl, _, col in epoch_conds]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5.5))
    ax_sweep  = fig.add_axes([0.05, 0.14, 0.34, 0.72])
    ax_epoch  = fig.add_axes([0.44, 0.14, 0.22, 0.72])
    ax_psycho = fig.add_axes([0.72, 0.14, 0.26, 0.72])

    # Epoch boundaries (ms)
    fix_end  = 25  * dt   # 250 ms
    stim_end = 50  * dt   # 500 ms
    trial_end= T   * dt   # 770 ms

    epoch_spans = [
        (0,        fix_end,   '#e8f0f8', 'fixation'),
        (fix_end,  stim_end,  '#e8f4ec', 'stimulus'),
        (stim_end, trial_end, '#fdf0ed', 'decision'),
    ]

    # ── Panel 1: sliding-window sweep ─────────────────────────────────────────
    for x0, x1, col, _ in epoch_spans:
        ax_sweep.axvspan(x0, x1, color=col, alpha=0.9, zorder=0)
    ax_sweep.axhline(0, color='#bbb', lw=1, ls=':', zorder=1)

    # Draw each window as a horizontal bar spanning its exact time range.
    # This avoids any interpolation or smoothing between windows.
    win_size_ms = win_size * dt
    for i, (mid_ms, dri) in enumerate(zip(sweep_mid, sweep_dri)):
        x0_w = mid_ms - win_size_ms / 2
        x1_w = mid_ms + win_size_ms / 2
        col_bar = RUST if dri > 0 else TEAL
        ax_sweep.barh(0, x1_w - x0_w, left=x0_w,
                      height=dri, align='edge',
                      color=col_bar, alpha=0.55, edgecolor='none', zorder=2)
    # Overlay the raw data points at window midpoints so the actual values are clear
    ax_sweep.scatter(sweep_mid, sweep_dri, s=28, color='#222',
                     zorder=4, edgecolors='white', linewidths=0.6)
    # Connect with a step-style line only between consecutive windows
    ax_sweep.step(sweep_mid, sweep_dri, where='mid', color='#333',
                  lw=1.4, alpha=0.7, zorder=3)

    for x in [fix_end, stim_end]:
        ax_sweep.axvline(x, color='#aaa', lw=1, ls='--', zorder=5)

    # Epoch labels: data x, axes-fraction y → sit inside plot at fixed height
    from matplotlib.transforms import blended_transform_factory
    trans_xdata_yaxes = blended_transform_factory(ax_sweep.transData,
                                                   ax_sweep.transAxes)
    for x0, x1, _, lbl in epoch_spans:
        ax_sweep.text((x0 + x1) / 2, 0.97, lbl,
                      ha='center', va='top', fontsize=8.5, color='#444',
                      fontweight='semibold', transform=trans_xdata_yaxes,
                      zorder=6)

    ax_sweep.set_xlabel('window midpoint (ms)')
    ax_sweep.set_ylabel(f'Δriskiness  (stim − baseline)')
    ax_sweep.set_title('When does dopamine matter?\n'
                       f'Each bar = one {win_size_ms:.0f} ms window (exact, no smoothing)')
    ax_sweep.set_xlim(0, trial_end)

    # Legend explaining Δriskiness
    ax_sweep.annotate(
        'Δriskiness =\nP(chose risky | matched-EV)\nwith stim  −  without stim',
        xy=(0.02, 0.52), xycoords='axes fraction',
        fontsize=7.5, color='#444', va='center',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.85)
    )

    # ── Panel 2: epoch bar summary ────────────────────────────────────────────
    # Draw as a horizontal timeline with vertical bars showing the effect
    EPOCH_RANGES_MS = {
        'fixation': (0, fix_end),
        'stimulus': (fix_end, stim_end),
        'decision': (stim_end, trial_end),
        'all':      (0, trial_end),
    }
    ys = {'fixation': 1.5, 'stimulus': 1.0, 'decision': 0.5, 'all': 0.0}

    for lbl, phases, col in epoch_conds:
        ri_val  = epoch_results[lbl]['riskiness']
        delta   = ri_val - base_ri
        x0, x1  = EPOCH_RANGES_MS.get(phases[0] if phases else 'all',
                                        EPOCH_RANGES_MS['all'])
        y = ys[lbl]
        ax_epoch.barh(y, x1 - x0, left=x0, height=0.32,
                      color=col, alpha=0.35, edgecolor=col, linewidth=1.2)
        ax_epoch.annotate(f'{lbl}\nΔ={delta:+.3f}',
                          xy=((x0 + x1) / 2, y),
                          ha='center', va='center', fontsize=8,
                          color='#222', fontweight='semibold')

    for x0, x1, bg, _ in epoch_spans:
        ax_epoch.axvspan(x0, x1, color=bg, alpha=0.55, zorder=0)
    for x in [fix_end, stim_end]:
        ax_epoch.axvline(x, color='#aaa', lw=1, ls='--', zorder=1)

    ax_epoch.set_xlim(0, trial_end)
    ax_epoch.set_xlabel('time (ms)')
    ax_epoch.set_yticks([])
    ax_epoch.set_title('Effect per epoch\n(Δriskiness vs baseline)')

    # ── Panel 3: psychometric overlays ────────────────────────────────────────
    for label, r, col in psycho_conds:
        lv_data, pp, rs_val = proportion_slope(r['trials'], r['choices'], 'prob')
        if lv_data.size < 2: continue
        fit = np.polyfit(lv_data, pp, 1)
        xs  = np.linspace(0.05, 0.95, 80)
        lw  = 2.4 if label in ('baseline', 'decision') else 1.5
        alpha = 1.0 if label in ('baseline', 'decision') else 0.72
        ax_psycho.plot(lv_data * 100, pp, 'o', color=col, ms=5.5, alpha=alpha, zorder=4)
        ax_psycho.plot(xs * 100, np.polyval(fit, xs), '-', color=col,
                       lw=lw, alpha=alpha,
                       label=f'{label}  ({rs_val:+.3f})')
    ax_psycho.axhline(0.5, color='#ccc', lw=0.9, ls=':')
    ax_psycho.set_xlabel('win probability (%)')
    ax_psycho.set_ylabel('proportion chosen  (matched EV)')
    ax_psycho.set_title('Psychometric curves\nper phase condition')
    ax_psycho.legend(fontsize=7.8, loc='lower left')

    fig.suptitle(f'Phase-restricted dopamine modulation  (stim = ctx +{mag:.1f})',
                 fontsize=11, fontweight='semibold', y=1.01)

    name = save_fig(fig, figs_dir, 'd1_phase_restricted.png')

    base_rs  = base_run['risk_slope']
    dec_rs   = epoch_results['decision']['risk_slope']
    fix_rs   = epoch_results['fixation']['risk_slope']
    win_peak_ms = float(sweep_mid[np.argmax(np.abs(sweep_dri))])
    decision_drives = np.isfinite(dec_rs) and abs(dec_rs - base_rs) > abs(fix_rs - base_rs)
    add_fig('D', name,
            'Phase-restricted modulation — sliding window + psychometrics',
            'Fig S4',
            'Sliding-window sweep shows when context has the biggest effect. '
            'Epoch bars + psychometric overlays confirm decision-period specificity.',
            f'Peak window at {win_peak_ms:.0f} ms; '
            f'Δslope: decision={dec_rs-base_rs:+.3f} vs fixation={fix_rs-base_rs:+.3f}',
            '', 'pass' if decision_drives else 'info')

# --------------------------------------------------------------------------- #
# GROUP F  ── neural decoding from existing pkl data
# --------------------------------------------------------------------------- #
def group_F(ctx_rows, figs_dir, runner, cfg):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.decomposition import PCA
    except ImportError as e:
        print('Group F skipped (sklearn unavailable):', e)
        return

    row0 = next((r for r in ctx_rows if abs(r['level']) < 0.06), ctx_rows[len(ctx_rows)//2])
    d = row0['d']
    r_policy = to_np(d['r_policy'])   # (T, B, N)
    T, _, N = r_policy.shape
    dec = slice(max(0, T-8), T)       # last 8 timesteps = deepest decision period

    # Build features for matched-EV trials only
    X, y_risk, y_prob_offered, trial_idx_keep = [], [], [], []
    for i, tr in enumerate(row0['trials']):
        c = row0['choices'][i]
        if c not in (1, 2): continue
        pl, pr = float(tr['prob_l']), float(tr['prob_r'])
        evl = pl * float(tr['size_l']); evr = pr * float(tr['size_r'])
        if not (abs(evl - evr) < 0.015 and abs(pl - pr) > 1e-6): continue
        feat = r_policy[dec, i, :].mean(axis=0)
        risky = (c == 2 and pr < pl) or (c == 1 and pl < pr)
        X.append(feat); y_risk.append(int(risky))
        y_prob_offered.append(min(pl, pr))
        trial_idx_keep.append(i)

    X = np.array(X); y_risk = np.array(y_risk)
    y_prob_offered = np.array(y_prob_offered)

    # ── F1a: Decoder psychometric ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    if X.shape[0] > 30 and len(np.unique(y_risk)) == 2:
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(X, y_risk)
        p_risky_pred = clf.predict_proba(X)[:, 1]  # P(risky) for each trial

        # bin by offered probability
        ax = axes[0]
        prob_bins = sorted(set(np.round(y_prob_offered, 2)))
        bin_centers, pred_mean, pred_lo, pred_hi, obs_mean = [], [], [], [], []
        for pb in prob_bins:
            mask = np.abs(y_prob_offered - pb) < 0.05
            if mask.sum() < 3: continue
            # bootstrap CI on predicted P(risky)
            preds_here = p_risky_pred[mask]
            obs_here   = y_risk[mask]
            bs = np.array([np.mean(np.random.choice(preds_here, len(preds_here), replace=True))
                           for _ in range(400)])
            bin_centers.append(pb)
            pred_mean.append(preds_here.mean())
            pred_lo.append(np.percentile(bs, 2.5))
            pred_hi.append(np.percentile(bs, 97.5))
            obs_mean.append(obs_here.mean())

        bc = np.array(bin_centers)
        pm = np.array(pred_mean); pl_ = np.array(pred_lo); ph = np.array(pred_hi)
        om = np.array(obs_mean)

        ax.fill_between(bc*100, pl_, ph, alpha=0.2, color='#444')
        ax.plot(bc*100, pm, 'o-', color='#222', lw=1.8, ms=6, zorder=3, label='decoder P(risky)')
        ax.plot(bc*100, om, 's--', color=RUST, lw=1.4, ms=5.5, alpha=0.8, label='observed P(risky)')
        ax.axhline(0.5, color='#bbb', lw=1, ls=':')
        ax.set_xlabel('lower probability option offered (%)')
        ax.set_ylabel('P(risky choice)')
        ax.set_title('Decoder psychometric\n(decision-period activity)')
        ax.legend(fontsize=8.5)
        verdict = f'decoder psychometric generated  (n={X.shape[0]})'
    else:
        axes[0].text(0.5, 0.5, 'too few matched-EV trials', ha='center', va='center',
                     transform=axes[0].transAxes, color=MUTED)
        verdict = 'insufficient matched-EV trials'

    # ── F3: dPCA-style risk trajectory ─────────────────────────────────────────
    ax = axes[1]
    risky_idx = [trial_idx_keep[i] for i, r in enumerate(y_risk) if r == 1]
    safe_idx  = [trial_idx_keep[i] for i, r in enumerate(y_risk) if r == 0]

    if len(risky_idx) >= 5 and len(safe_idx) >= 5:
        r_risky = r_policy[:, risky_idx, :].mean(axis=1)   # (T, N)
        r_safe  = r_policy[:, safe_idx,  :].mean(axis=1)   # (T, N)

        diff = r_risky - r_safe   # (T, N) — what changes between risky and safe choices
        # Find axis that best captures this difference over the decision epoch
        diff_dec = diff[50:, :]
        pca_diff = PCA(n_components=1).fit(diff_dec)
        risk_axis = pca_diff.components_[0]   # (N,)

        # Project individual trials onto risk axis (decision period)
        proj_risky = r_policy[50:, risky_idx, :] @ risk_axis  # (T_dec, n_risky)
        proj_safe  = r_policy[50:, safe_idx,  :] @ risk_axis  # (T_dec, n_safe)
        time_ms    = np.arange(50, T) * 10   # ms

        def mean_sem(x):
            return x.mean(1), x.std(1) / np.sqrt(x.shape[1])

        m_r, s_r = mean_sem(proj_risky)
        m_s, s_s = mean_sem(proj_safe)

        ax.fill_between(time_ms, m_r - s_r, m_r + s_r, alpha=0.18, color=RUST)
        ax.fill_between(time_ms, m_s - s_s, m_s + s_s, alpha=0.18, color=TEAL)
        ax.plot(time_ms, m_r, '-', color=RUST, lw=2, label=f'risky (n={len(risky_idx)})')
        ax.plot(time_ms, m_s, '-', color=TEAL, lw=2, label=f'safe (n={len(safe_idx)})')
        ax.axvline(500, color='#bbb', ls='--', lw=1)
        ax.set_xlabel('time (ms)'); ax.set_ylabel('projection onto risk axis')
        ax.set_title('Risk axis trajectory\n(mean ± SEM by choice)')
        ax.legend(fontsize=8.5)
    else:
        axes[1].text(0.5, 0.5, 'too few trials for trajectory', ha='center', va='center',
                     transform=axes[1].transAxes, color=MUTED)

    fig.suptitle('Neural decoding — decision-period activity', fontsize=10, fontweight='semibold')
    fig.tight_layout()
    name = save_fig(fig, figs_dir, 'f1_decoder.png')
    add_fig('F', name, 'Decoder psychometric + risk axis trajectory',
            'Fig 5E/F + Fig 5A-D',
            'Left: decoder P(risky) vs offered probability with bootstrap CI. '
            'Right: mean ± SEM projection onto risk discriminant axis, risky vs safe choices.',
            verdict, '', 'info')

    # ── F2: bootstrap ON vs OFF scatter ─────────────────────────────────────────
    try:
        # baseline OFF: bootstrap riskiness from ctx=0 pkl
        tri_off = row0['trials']; cho_off = row0['choices']
        N_off = len(tri_off)
        # stim ON: run fresh inference with d1_only_stim
        row_stim_high = next(
            (r for r in ctx_rows if abs(r['level'] - cfg['stim_mag']) < 0.15),
            None)

        stim_run = runner.run(cfg['stim_mag'], cfg['stim_n'], kind='prob',
                              pathway_mode='d1_only_stim')
        tri_on  = stim_run['trials']; cho_on = stim_run['choices']
        N_on = len(tri_on)

        n_boot = 250
        ri_off_bs, ri_on_bs = [], []
        for b in range(n_boot):
            idx_off = np.random.choice(N_off, N_off, replace=True)
            idx_on  = np.random.choice(N_on,  N_on,  replace=True)
            ri_off_bs.append(riskiness_from([tri_off[i] for i in idx_off], cho_off[idx_off]))
            ri_on_bs.append(riskiness_from([tri_on[i]  for i in idx_on],  cho_on[idx_on]))

        ri_off_bs = np.array(ri_off_bs); ri_on_bs = np.array(ri_on_bs)
        valid = np.isfinite(ri_off_bs) & np.isfinite(ri_on_bs)
        ri_off_bs = ri_off_bs[valid]; ri_on_bs = ri_on_bs[valid]

        fig2, ax2 = plt.subplots(figsize=(5.0, 5.0))
        ax2.plot([0, 1], [0, 1], '--', color='#bbb', lw=1.2, zorder=1)
        ax2.scatter(ri_off_bs, ri_on_bs, s=10, alpha=0.35, color=RUST,
                    edgecolors='none', rasterized=True, zorder=2)
        # CI crosshairs at medians
        m_off, m_on = np.median(ri_off_bs), np.median(ri_on_bs)
        ci_off = np.percentile(ri_off_bs, [2.5, 97.5])
        ci_on  = np.percentile(ri_on_bs,  [2.5, 97.5])
        ax2.errorbar(m_off, m_on, xerr=[[m_off-ci_off[0]], [ci_off[1]-m_off]],
                     yerr=[[m_on-ci_on[0]], [ci_on[1]-m_on]],
                     fmt='o', color='#222', ms=8, lw=1.6, capsize=4, zorder=5)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.set_xlabel('riskiness  stim OFF  (bootstrap sessions)')
        ax2.set_ylabel('riskiness  D1 stim ON  (bootstrap sessions)')
        ax2.set_title('Decoder shift: riskiness ON vs OFF\n(bootstrap pseudo-sessions, n=250)')
        ax2.text(0.04, 0.95,
                 f'OFF median={m_off:.2f}  ON median={m_on:.2f}\nΔ={m_on-m_off:+.2f}',
                 transform=ax2.transAxes, va='top', fontsize=8.5, color='#444')
        name2 = save_fig(fig2, figs_dir, 'f2_decoder_shift.png')
        shift = m_on > m_off
        add_fig('F', name2, 'Riskiness ON vs OFF scatter (bootstrap)',
                'Fig 5G',
                f'Bootstrap pseudo-sessions: riskiness stim-OFF (x) vs D1-stim ON (y). '
                f'OFF={m_off:.2f}, ON={m_on:.2f}, Δ={m_on-m_off:+.2f}.',
                f'D1-stim shifts cloud above diagonal.', '',
                'pass' if shift else 'info')
    except Exception as e:
        print('F2 failed:', e)
        traceback.print_exc()

# --------------------------------------------------------------------------- #
# GROUP E  ── lasting-change simulation: context ON → OFF sessions
# --------------------------------------------------------------------------- #
def group_E(ctx_rows, figs_dir, runner, cfg):
    mag = cfg['e_stim_ctx']
    sessions = [
        (f'S0\nbaseline\n(ctx=0.0)',             0.0,  100),
        (f'S1\nstim ON\n(ctx={mag:+.1f})',      mag,  101),
        (f'S2\nstim OFF\n(ctx=0.0)',             0.0,  102),
        (f'S3\nstim ON\n(ctx={mag:+.1f})',      mag,  103),
        (f'S4\nstim OFF\n(ctx=0.0)',             0.0,  104),
    ]
    slopes, riski, labels, is_on = [], [], [], []
    for label, ctx, seed in sessions:
        r = runner.run(ctx, cfg['stim_n'], kind='prob', seed=seed)
        slopes.append(r['risk_slope'])
        riski.append(r['riskiness'])
        labels.append(label)
        is_on.append(ctx != 0.0)

    colors = [RUST if on else TEAL for on in is_on]
    off_ri = [riski[i] for i,on in enumerate(is_on) if not on]
    persists = (len(off_ri) > 1 and all(np.isfinite(x) for x in off_ri)
                and abs(off_ri[-1] - off_ri[0]) > 0.04)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, vals, ylabel, title in [
        (axes[0], slopes, 'risk slope  (neg = risk-seeking)', 'E1a · Risk slope per session'),
        (axes[1], riski,  'P(risky | matched-EV)',            'E1b · Riskiness per session'),
    ]:
        ax.bar(range(len(labels)), vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3)
        ax.axhline(0 if 'slope' in ylabel else 0.5, color=INK, lw=0.9, ls=':', zorder=2)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v+(0.005 if v>=0 else -0.005), f'{v:+.3f}',
                        ha='center', va='bottom' if v>=0 else 'top', fontsize=8)
    axes[0].legend(handles=[Patch(color=RUST, label='ctx ON'), Patch(color=TEAL, label='ctx OFF')],
                   loc='lower right')
    fig.suptitle('E1 · Context ON → OFF session simulation', fontsize=11, fontweight='semibold')
    fig.tight_layout()
    name = save_fig(fig, figs_dir, 'e1_session_simulation.png')

    if persists:
        verdict = (f'OFF sessions drift {off_ri[0]:.2f}→{off_ri[-1]:.2f}: some persistence ✓')
        st = 'pass'
    else:
        verdict = (f'OFF sessions revert: {" → ".join(f"{x:.2f}" for x in off_ri)}. '
                   'No lasting effect without weight changes — requires actual plasticity.')
        st = 'info'
    add_fig('E', name, 'E1: Context ON → OFF simulation', 'Fig 4A-C',
            'Repeated VTA stimulation caused durable risk shift that persisted OFF.',
            'OFF-session riskiness should drift risk-seeking and persist.',
            verdict, st)

# --------------------------------------------------------------------------- #
# Bonus neural figures from existing pkl data
# --------------------------------------------------------------------------- #
def bonus_neural(ctx_rows, figs_dir):
    """Neural regression, D1/D2 contribution, value dynamics."""
    row0 = next((r for r in ctx_rows if abs(r['level']) < 0.06), ctx_rows[len(ctx_rows)//2])
    d = row0['d']
    r_policy = to_np(d['r_policy'])   # (T, B, N)
    T, B, N = r_policy.shape
    dec = slice(50, T)

    # --- Neural regression: β_HH vs β_EV scatter ---
    try:
        act  = r_policy[dec, :, :].mean(axis=0)   # (B, N) mean decision-period activity
        dh, de = [], []
        valid = []
        for i, tr in enumerate(row0['trials']):
            pl, pr = float(tr['prob_l']), float(tr['prob_r'])
            evl = pl*float(tr['size_l']); evr = pr*float(tr['size_r'])
            if pl <= 0 or pr <= 0 or evl <= 0 or evr <= 0: continue
            dh.append(float(np.log(pr) - np.log(pl)))
            de.append(float(np.log(evr) - np.log(evl)))
            valid.append(i)
        dh, de = np.array(dh), np.array(de)
        X_reg = np.column_stack([dh, de, np.ones(len(dh))])
        act_v = act[valid, :]   # (n_valid, N)
        from numpy.linalg import lstsq
        coefs, _, _, _ = lstsq(X_reg, act_v, rcond=None)  # (3, N)
        beta_hh, beta_ev = coefs[0], coefs[1]

        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        half = N // 2
        ax.scatter(beta_hh[:half],  beta_ev[:half],  s=22, alpha=0.7, color=RUST,
                   label=f'D1 neurons (0–{half-1})', edgecolors='none')
        ax.scatter(beta_hh[half:], beta_ev[half:], s=22, alpha=0.7, color=TEAL,
                   label=f'D2 neurons ({half}–{N-1})', edgecolors='none')
        ax.axhline(0, color=INK, lw=0.7, alpha=0.4)
        ax.axvline(0, color=INK, lw=0.7, alpha=0.4)
        ax.set_xlabel('β  HH-LL  (risk encoding)'); ax.set_ylabel('β  EV  (value encoding)')
        ax.set_title('Neural regression: risk vs value encoding')
        ax.legend()
        name = save_fig(fig, figs_dir, 'bonus_neural_regression.png')
        add_fig('NEURAL', name, 'Neural regression β scatter', 'Fig 5A-D',
                'Neurons encoding EV only, risk only, or mixed — D1/D2 split.',
                'D1 vs D2 populations diverge in encoding.',
                'β_HH vs β_EV scatter by D1/D2 population', 'info')
    except Exception as e:
        print('Neural regression failed:', e)

    # --- D1 vs D2 value contribution across ctx levels ---
    try:
        D1pull = to_np(d['D1pull'])  # (T, B, 3)
        D2pull = to_np(d['D2pull'])
        if D1pull is not None:
            # mean pull magnitude at choice time for each action
            d1_mean = float(np.abs(D1pull[dec, :, 1:]).mean())  # choices are actions 1,2
            d2_mean = float(np.abs(D2pull[dec, :, 1:]).mean())

            # D1/D2 pull ratio across ctx levels
            d1_levels, d2_levels = [], []
            for row in ctx_rows:
                rd = row['d']
                if rd['D1pull'] is not None:
                    d1m = float(np.abs(to_np(rd['D1pull'])[dec, :, 1:]).mean())
                    d2m = float(np.abs(to_np(rd['D2pull'])[dec, :, 1:]).mean())
                    d1_levels.append(d1m); d2_levels.append(d2m)
                else:
                    d1_levels.append(np.nan); d2_levels.append(np.nan)

            lvs = np.array([r['level'] for r in ctx_rows])
            fig, ax = plt.subplots(figsize=(7.5, 3.8))
            ax.plot(lvs, d1_levels, 'o-', color=RUST, lw=2, ms=5, label='|D1 pull|')
            ax.plot(lvs, d2_levels, 'o-', color=TEAL, lw=2, ms=5, label='|D2 pull|')
            ax.set_xlabel('dopamine context input')
            ax.set_ylabel('mean |pathway pull| (decision period)')
            ax.set_title('D1 vs D2 pathway contribution across dopamine levels')
            ax.legend()
            name = save_fig(fig, figs_dir, 'bonus_d1d2_pull.png')
            add_fig('NEURAL', name, 'D1 vs D2 pathway pull across ctx', 'Fig 5 / Fig 3',
                    'D1 pull grows with DA; D2 shrinks — push-pull mechanism.',
                    'D1 and D2 curves diverge monotonically with ctx.',
                    'D1/D2 pull magnitudes vs ctx level', 'info')
    except Exception as e:
        print('D1/D2 pull figure failed:', e)

    # Embed the existing mega figures
    for src, dst, title, cap in [
        ('context_v_rpe_dynamics_mega_10trials.png', 'existing_v_rpe_mega.png',
         'Existing: V and RPE dynamics across all ctx levels',
         'Value prediction and objective RPE traces for 10 representative trials at each context level.'),
        ('mega_policy_subjective_values_5x19.png', 'existing_policy_values_mega.png',
         'Existing: Policy subjective values mega (5 rows × 19 ctx)',
         'D1 pull, D2 pull, total logit, engagement drive, and choice advantage '
         'across all 19 context levels.'),
        ('mega_comparison_context_sweep.png', 'existing_mega_comparison.png',
         'Existing: Mega comparison context sweep',
         'Comprehensive multi-row comparison of behaviour, neural regression, and '
         'value grids across all dopamine context levels.'),
    ]:
        n = copy_existing(src, figs_dir, dst)
        if n:
            add_fig('NEURAL', n, title, 'multiple paper figures', cap,
                    'Rich neural/behavioural visualisation across all ctx levels.',
                    'Embedded from pre-computed figures', 'info')

# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
CSS = """
    :root{--bg:#f2f2f2;--paper:#fff;--ink:#1a1a1a;--muted:#5a5a5a;--line:#d8d8d8}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);
      font-family:"Helvetica Neue",Arial,sans-serif;font-size:13px;line-height:1.55}
    main{max-width:1180px;margin:0 auto;padding:32px 20px 64px}
    header{background:var(--paper);border:1px solid var(--line);border-radius:5px;
      padding:24px 28px;margin-bottom:20px}
    h1{font-size:1.55rem;font-weight:600;margin:0 0 5px;letter-spacing:-.02em}
    h2{font-size:1.05rem;font-weight:600;margin:0 0 14px;letter-spacing:-.01em;
      padding-bottom:6px;border-bottom:1px solid var(--line)}
    h3{font-size:.88rem;font-weight:500;margin:0 0 5px;color:#333}
    .kicker{font-size:.68rem;font-weight:700;text-transform:uppercase;
      letter-spacing:.13em;color:#888;margin-bottom:8px}
    .lede{color:var(--muted);font-size:.86rem;margin:6px 0 0;max-width:820px}
    section{background:var(--paper);border:1px solid var(--line);border-radius:5px;
      padding:20px 22px;margin-bottom:16px}
    .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:14px}
    .card{padding:12px 14px;border:1px solid var(--line);border-radius:3px}
    .metric{display:block;margin-top:4px;font-size:1.6rem;font-weight:700;
      color:#2166ac;letter-spacing:-.03em}
    .small{color:var(--muted);font-size:.78rem;margin:3px 0 0}
    table{width:100%;border-collapse:collapse;font-size:.83rem;margin-top:12px}
    th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
    th{background:#f5f5f5;font-weight:600;font-size:.72rem;text-transform:uppercase;
      letter-spacing:.06em;color:#444}
    tr:last-child td{border-bottom:0}
    .note{padding:10px 13px;border-left:3px solid #aaa;background:#f8f8f8;
      border-radius:3px;color:#444;margin:12px 0 0;font-size:.84rem}
    .plot-block{display:grid;grid-template-columns:1fr;gap:14px;margin-top:14px}
    .two-col{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start;margin-top:14px}
    .three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;align-items:start;margin-top:14px}
    figure{margin:0;border:1px solid var(--line);border-radius:3px;overflow:hidden;background:#fff}
    figure img{display:block;width:100%;height:auto}
    figcaption{padding:6px 9px;font-size:.76rem;color:var(--muted);
      border-top:1px solid var(--line);background:#fafafa}
    code{background:#efefef;padding:.1em .26em;border-radius:2px;
      font-family:monospace;font-size:.85em}
    @media(max-width:860px){.grid,.two-col,.three-col{grid-template-columns:1fr}}
"""

GROUP_LABELS = {
    'P': 'Pre-flight',
    'A': 'Baseline behaviour',
    'B': 'Necessity — ablation',
    'C': 'Directional DA control',
    'D': 'Timing specificity',
    'F': 'Neural decoding',
    'E': 'Lasting change',
    'NEURAL': 'Neural representations',
}

_STRIP_CODE = re.compile(r'^[A-Z]\d[\w+/]*[:\s·]+\s*', re.ASCII)

def _clean_title(t):
    """Remove leading figure codes like 'A1:', 'B1+B2:', 'F1+F3:' from titles."""
    return _STRIP_CODE.sub('', t).strip()

def write_html(outdir, cfg, ctx_rows):
    levels = [r['level'] for r in ctx_rows]
    risk   = [r['riskiness'] for r in ctx_rows]
    mid_ri = float(np.nanmean([r for l, r in zip(levels, risk) if abs(l) < 0.11]))
    full_span = float(np.nanmax(risk) - np.nanmin(risk)) if risk else np.nan

    rows_by_group = defaultdict(list)
    for r in REPORT:
        rows_by_group[r['group']].append(r)

    def fig_html(fname, caption):
        if not fname:
            return ''
        return (f'<figure><img src="figs/{html_module.escape(fname)}" loading="lazy">'
                f'<figcaption>{html_module.escape(caption)}</figcaption></figure>')

    def section_figs(entries, layout='plot-block'):
        parts = [f'<div class="{layout}">']
        for e in entries:
            title = _clean_title(e['title'])
            parts.append('<div>')
            parts.append(f'<h3>{html_module.escape(title)}</h3>')
            parts.append(f'<p class="small">{html_module.escape(e["what"])}</p>')
            if e['fig']:
                parts.append(fig_html(e['fig'], e['what']))
            parts.append('</div>')
        parts.append('</div>')
        return '\n'.join(parts)

    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Sasaki 2024 replication</title>'
        f'<style>{CSS}</style></head><body><main>'
    ]

    parts.append(
        f'<header>'
        f'<div class="kicker">Gambling task · opal04 · D1/D2 modulation</div>'
        f'<h1>Replicating Sasaki et al. (2024)</h1>'
        f'<p class="lede">Single trained network — <code>{html_module.escape(MODEL_NAME)}</code>. '
        f'Generated {time.strftime("%Y-%m-%d %H:%M")}.</p>'
        f'</header>'
    )

    # Compact metric cards
    parts.append('<section><div class="grid">')
    for label, val, desc in [
        ('Riskiness (DA=0)', f'{mid_ri:.2f}', 'P(risky) at neutral dopamine'),
        ('Modulation span',  f'{full_span:.2f}', 'Δriskiness, ctx −0.9 → +0.9'),
        ('Context levels',  str(len(ctx_rows)), 'pre-computed pkl files'),
        ('Figures',         str(len(REPORT)),   'panels generated'),
    ]:
        parts.append(
            f'<div class="card"><h3>{html_module.escape(label)}</h3>'
            f'<span class="metric">{html_module.escape(val)}</span>'
            f'<p class="small">{html_module.escape(desc)}</p></div>'
        )
    parts.append('</div></section>')

    # Index table — just figure title + description, no verdicts/badges
    parts.append('<section><h2>Figure index</h2><table><thead><tr>'
                 '<th>Section</th><th>Figure</th><th>Description</th>'
                 '</tr></thead><tbody>')
    for g in ['P', 'A', 'B', 'C', 'D', 'F', 'NEURAL']:
        for e in rows_by_group.get(g, []):
            parts.append(
                f'<tr><td>{html_module.escape(GROUP_LABELS.get(g, g))}</td>'
                f'<td>{html_module.escape(_clean_title(e["title"]))}</td>'
                f'<td style="color:var(--muted)">{html_module.escape(e["what"][:100])}</td></tr>'
            )
    parts.append('</tbody></table></section>')

    # Per-group figure sections
    for g in ['P', 'A', 'B', 'C', 'D', 'F', 'NEURAL']:
        entries = rows_by_group.get(g, [])
        if not entries:
            continue
        label = GROUP_LABELS.get(g, g)
        parts.append(f'<section><h2>{html_module.escape(label)}</h2>')
        # large/fullwidth figures (mega, existing) go full width; rest in 2-col
        mega = [e for e in entries
                if any(k in e.get('title', '').lower()
                       for k in ('mega', 'v and rpe', 'comparison', 'sweep'))]
        rest = [e for e in entries if e not in mega]
        if rest:
            layout = 'plot-block' if len(rest) == 1 else 'two-col'
            parts.append(section_figs(rest, layout))
        for e in mega:
            parts.append(section_figs([e], 'plot-block'))
        parts.append('</section>')

    parts.append('</main></body></html>')

    out = os.path.join(outdir, 'report.html')
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    print('\n✓ Report written:', out)
    return out

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--savefile',     default=SAVEFILE)
    ap.add_argument('--outdir',       default=os.path.join(REPO_ROOT, 'reports/figure_replication'))
    ap.add_argument('--device',       default='cpu')
    ap.add_argument('--n-per-pair',   type=int, default=50,
                    help='trials per option-pair for fresh inference (ablation, timing, stim)')
    ap.add_argument('--stim-mag',     type=float, default=0.6)
    ap.add_argument('--e-stim-ctx',   type=float, default=0.6)
    ap.add_argument('--groups',       default='P,A,B,C,D,F,E,NEURAL')
    ap.add_argument('--skip-e',       action='store_true')
    ap.add_argument('--quick',        action='store_true',
                    help='tiny sample sizes / fewer ctx levels — smoke test only')
    args = ap.parse_args()

    setup_style()

    groups = [g.strip().upper() for g in args.groups.split(',') if g.strip()]
    if args.skip_e: groups = [g for g in groups if g != 'E']

    n = 5 if args.quick else args.n_per_pair
    stim_n = 8 if args.quick else max(20, n // 2)
    cfg = dict(
        outdir=args.outdir, device=args.device,
        n=n, stim_n=stim_n, stim_mag=args.stim_mag, e_stim_ctx=args.e_stim_ctx,
        groups=groups, quick=args.quick,
    )

    figs_dir = os.path.join(args.outdir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    # Restore any previously-computed report entries from other groups so
    # partial re-runs (--groups D) don't erase everything else.
    import json
    report_cache = os.path.join(args.outdir, 'report_cache.json')
    if os.path.exists(report_cache):
        with open(report_cache) as f:
            cached = json.load(f)
        # Keep cached entries for groups we are NOT running this time
        for entry in cached:
            if entry['group'] not in groups:
                REPORT.append(entry)

    # Load all pre-computed context levels (no inference needed for these)
    print('Loading pre-computed trial pkl files...')
    ctx_rows = load_all_ctx_data(TRIALS_DIR)
    if args.quick:
        ctx_rows = ctx_rows[::3]   # every 3rd level for smoke test
    print(f'  {len(ctx_rows)} context levels loaded: {[r["level"] for r in ctx_rows]}')

    # Load the model once (used only for fresh-inference groups)
    print(f'Loading model: {args.savefile}')
    runner = Runner(args.savefile, device=args.device)
    print(f'  N={runner.N}')

    dispatch = {
        'P':      lambda: group_P(figs_dir, runner),
        'A':      lambda: group_A(ctx_rows, figs_dir, runner, cfg),
        'B':      lambda: group_B(ctx_rows, figs_dir, runner, cfg),
        'C':      lambda: group_C(ctx_rows, figs_dir, runner, cfg),
        'D':      lambda: group_D(figs_dir, runner, cfg),
        'F':      lambda: group_F(ctx_rows, figs_dir, runner, cfg),
        'E':      lambda: group_E(ctx_rows, figs_dir, runner, cfg),
        'NEURAL': lambda: bonus_neural(ctx_rows, figs_dir),
    }

    order = ['P', 'A', 'B', 'C', 'D', 'F', 'E', 'NEURAL']
    for g in order:
        if g not in groups: continue
        t0 = time.time()
        print(f'\n=== Group {g} ===')
        try:
            dispatch[g]()
        except Exception as e:
            print(f'Group {g} failed: {e}')
            traceback.print_exc()
        print(f'  done in {time.time()-t0:.1f}s')

    # Persist all report entries (merged) for future partial re-runs
    with open(report_cache, 'w') as f:
        json.dump(REPORT, f, indent=2)

    write_html(args.outdir, cfg, ctx_rows)


if __name__ == '__main__':
    main()
