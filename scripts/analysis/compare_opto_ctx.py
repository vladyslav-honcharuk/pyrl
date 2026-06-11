#!/usr/bin/env python3
"""
Compare optogenetic RPE-based modulation to directly applied context inputs.

Generates a small psychometric trial set, runs the model with optostimulation
to get the trial-averaged RPE (the implicit "context"), then runs the same
trials with that averaged RPE provided explicitly as `context_input` and
compares per-unit D1/D2 contributions.
"""
import argparse
import os
import numpy as np
import torch

from pyrl.model import Model


def main(args):
    model = Model('tasks/gambling.py')
    pg = model.get_pg(args.savefile, seed=args.seed, load='best', device='cpu')

    # Deterministic RNG to match plotting utilities
    pg.rng = np.random.RandomState(args.seed)

    # Generate psychometric trials (small set)
    from tasks.gambling import generate_psychometric_trial_set, get_condition
    specs = generate_psychometric_trial_set(trials_per_comparison=args.trials_per_comparison)
    pg.rng.shuffle(specs)
    trials = [get_condition(pg.rng, pg.dt, spec) for spec in specs]

    # --- 1) Opto condition: enable RPE modulation and set opto offset ---
    pg.use_rpe_modulation = True
    pg.opto_stim_offset = args.opto_offset
    pg.opto_stim_gain = 1.0
    pg.opto_stim_phase = 'all'

    print(f"Running opto condition (offset={args.opto_offset}) with {len(trials)} trials")
    results_opto = pg.run_trials(trials, return_states=True, collect_policy_diagnostics=True)

    # Extract RPE timeseries and mask
    if 'RPE_continuous' not in results_opto:
        raise RuntimeError('Model did not return RPE_continuous. Ensure use_rpe_modulation=True.')

    RPE = results_opto['RPE_continuous']  # (T, B)
    M = results_opto['M']                 # (T, B)

    valid_steps = M.sum(dim=0).clamp(min=1.0)  # (B,)
    trial_avg_rpe = (RPE * M).sum(dim=0) / valid_steps  # (B,)

    # Convert to numpy for passing as context_input
    trial_ctx = trial_avg_rpe.cpu().numpy()

    # --- 2) Context condition: disable RPE modulation and provide trial_ctx ---
    pg.use_rpe_modulation = False
    pg.opto_stim_offset = 0.0

    print("Running direct-context condition using trial-averaged RPE as context_input")
    results_ctx = pg.run_trials(
        trials,
        return_states=True,
        context_input=trial_ctx,
        collect_policy_diagnostics=True,
    )

    # --- 3) Compare per-unit D1/D2 contributions ---
    D1_opto = results_opto['Policy_D1_Pull'].cpu().numpy()  # (T, B, Nout)
    D2_opto = results_opto['Policy_D2_Pull'].cpu().numpy()
    D1_ctx  = results_ctx['Policy_D1_Pull'].cpu().numpy()
    D2_ctx  = results_ctx['Policy_D2_Pull'].cpu().numpy()
    M_np = M.cpu().numpy()  # (T, B)

    T, B, Nout = D1_opto.shape

    # Average over time using mask M
    mask_exp = M_np[..., None]
    mean_d1_opto = (D1_opto * mask_exp).sum(axis=0) / (mask_exp.sum(axis=0).clip(min=1.0))  # (B, Nout)
    mean_d1_ctx  = (D1_ctx  * mask_exp).sum(axis=0) / (mask_exp.sum(axis=0).clip(min=1.0))

    # Flatten across trials to compute per-unit correlations
    mean_d1_opto_flat = mean_d1_opto.reshape(-1, Nout)  # (B, Nout)
    mean_d1_ctx_flat  = mean_d1_ctx.reshape(-1, Nout)

    # Compute Pearson correlation per output dimension (unit)
    corrs = []
    for unit in range(Nout):
        x = mean_d1_opto_flat[:, unit]
        y = mean_d1_ctx_flat[:, unit]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(x, y)[0, 1])

    corrs = np.array(corrs)

    out = {
        'opto_offset': args.opto_offset,
        'n_trials': len(trials),
        'mean_corr_mean': float(np.nanmean(corrs)),
        'mean_corr_median': float(np.nanmedian(corrs)),
        'per_unit_corrs': corrs,
        'trial_ctx': trial_ctx,
    }

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, f'compare_opto_ctx_off{str(args.opto_offset).replace(".","p")}.npz')
    np.savez(outpath, **out)
    print(f"Saved comparison results to {outpath}")
    print(f"Mean per-unit correlation (D1): {out['mean_corr_mean']:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('savefile', help='Path to saved model file (.pkl)')
    p.add_argument('--seed', type=int, default=999)
    p.add_argument('--opto-offset', type=float, default=0.1)
    p.add_argument('--trials-per-comparison', type=int, default=2)
    p.add_argument('--outdir', default='data/opto_ctx_compare')
    args = p.parse_args()
    main(args)
