#!/usr/bin/env python3
"""Plot delta HH-LL slope across fixed-context trial files."""

import argparse
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pyrl import utils


def load_pkl(path):
    data = utils.load(path)
    return {
        'trials': data[0],
        'A': data[4],
    }


def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def first_choices(A):
    a_np = to_np(A)
    acts = np.argmax(a_np, axis=2) if a_np.ndim == 3 else a_np
    choices = np.full(acts.shape[1], -1, dtype=int)
    for j in range(acts.shape[1]):
        idx = np.where((acts[:, j] == 1) | (acts[:, j] == 2))[0]
        if idx.size:
            choices[j] = acts[idx[0], j]
    return choices


def proportion_slope(trials, choices):
    offered = defaultdict(int)
    chosen = defaultdict(int)
    for i, tr in enumerate(trials):
        c = choices[i]
        if c not in (1, 2):
            continue
        pl, pr = float(tr['prob_l']), float(tr['prob_r'])
        evl = pl * float(tr['size_l'])
        evr = pr * float(tr['size_r'])
        if not (abs(evl - evr) < 0.015 and abs(pl - pr) > 1e-6):
            continue
        la, lb = round(pl, 4), round(pr, 4)
        offered[la] += 1
        offered[lb] += 1
        chosen[la if c == 1 else lb] += 1
    levels = np.array(sorted(offered))
    if levels.size < 2:
        return np.nan
    props = np.array([chosen[l] / offered[l] for l in levels])
    return float(np.polyfit(levels, props, 1)[0])


def parse_ctx_level(path):
    name = os.path.basename(path)
    match = re.search(r'ctx(neg|pos)(\d+p\d+)', name)
    if not match:
        raise ValueError(f'Could not parse context level from {name}')
    sign = -1.0 if match.group(1) == 'neg' else 1.0
    return round(sign * float(match.group(2).replace('p', '.')), 3)


def load_ctx_rows(trials_dir):
    rows = []
    for name in sorted(os.listdir(trials_dir)):
        if not name.endswith('.pkl') or 'ctx' not in name:
            continue
        path = os.path.join(trials_dir, name)
        level = parse_ctx_level(path)
        if abs(level) > 0.91:
            continue
        data = load_pkl(path)
        choices = first_choices(data['A'])
        trials = data['trials'][:len(choices)]
        rows.append({
            'level': level,
            'risk_slope': proportion_slope(trials, choices),
            'path': path,
        })
    rows.sort(key=lambda row: row['level'])
    return rows


def plot_delta_slope(rows, outfile, title):
    baseline = next((row for row in rows if abs(row['level']) < 1e-9), None)
    if baseline is None:
        raise ValueError('Missing ctx=0.0 baseline file')

    levels = np.array([row['level'] for row in rows], dtype=float)
    deltas = np.array([row['risk_slope'] - baseline['risk_slope'] for row in rows], dtype=float)
    colors = plt.cm.coolwarm((levels - levels.min()) / (levels.max() - levels.min()))

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.axhline(0.0, color='#333333', lw=1.2, ls='--', zorder=1)
    ax.scatter(levels, deltas, s=85, c=colors, edgecolors='white', linewidths=0.7, zorder=3)

    ax.set_xlabel('Context level', fontsize=13)
    ax.set_ylabel(r'$\Delta$ slope for HH-LL  (ctx - baseline)', fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(levels.min() - 0.06, levels.max() + 0.06)
    ax.set_xticks(levels)
    ax.set_xticklabels([f'{level:+.1f}' for level in levels], rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.35)

    fig.tight_layout()
    fig.savefig(outfile, dpi=250, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot delta HH-LL slope across 19 context conditions.')
    parser.add_argument('trials_dir', help='Directory containing trials_activity_ctx*.pkl files')
    parser.add_argument('--out', default=None, help='Output PNG path')
    parser.add_argument('--title', default='Delta HH-LL slope across context conditions')
    args = parser.parse_args()

    rows = load_ctx_rows(args.trials_dir)
    if not rows:
        raise SystemExit(f'No context trial files found in {args.trials_dir}')

    outfile = args.out or os.path.join(args.trials_dir, '..', '..', 'figures', 'ctx_delta_risk_slope.png')
    outfile = os.path.abspath(outfile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    plot_delta_slope(rows, outfile, args.title)
    print(f'Saved figure: {outfile}')


if __name__ == '__main__':
    main()
