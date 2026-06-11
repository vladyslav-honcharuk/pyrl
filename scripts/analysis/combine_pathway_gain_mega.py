#!/usr/bin/env python3
"""Combine pathway stim/suppress sweeps into one signed mega plot."""

import argparse
import os
import re
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.plotting.gambling import plot_context_mega_comparison  # noqa: E402


def parse_level(name):
    match = re.search(r'ctx(?:neg|pos)(\d+p\d+)', name)
    if not match:
        raise ValueError(f'Could not parse level from {name}')
    return float(match.group(1).replace('p', '.'))


def collect_levels(trials_dir, sign):
    levels = {}
    for name in sorted(os.listdir(trials_dir)):
        if not name.endswith('.pkl') or 'ctx' not in name:
            continue
        level = parse_level(name)
        if level == 0.0:
            signed_level = 0.0
        else:
            signed_level = sign * level
        levels[signed_level] = os.path.join(trials_dir, name)
    return levels


def main():
    parser = argparse.ArgumentParser(description='Combine suppress/stim sweeps into one signed mega plot.')
    parser.add_argument('--stim-trials-dir', required=True)
    parser.add_argument('--suppress-trials-dir', required=True)
    parser.add_argument('--model-file', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--title-prefix', required=True)
    parser.add_argument('--basename', default='mega_comparison_context_sweep_signed.png')
    args = parser.parse_args()

    stim = collect_levels(args.stim_trials_dir, sign=1.0)
    suppress = collect_levels(args.suppress_trials_dir, sign=-1.0)

    trialsfiles = {}
    trialsfiles.update(suppress)
    trialsfiles.update(stim)
    levels = sorted(trialsfiles)

    os.makedirs(args.outdir, exist_ok=True)
    plot_context_mega_comparison(
        levels,
        trialsfiles,
        args.model_file,
        args.outdir,
        title_prefix=args.title_prefix,
        output_basename=args.basename,
    )
    print(os.path.join(args.outdir, args.basename))


if __name__ == '__main__':
    main()
