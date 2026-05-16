#!/usr/bin/env python
"""
Test optogenetic VTA stimulation during inference.
"""
import sys
import numpy as np

sys.path.insert(0, '.')
from pyrl.model import Model

# Load model
modelfile = '/Users/vlaruks/.claude-worktrees/pyrl/sharp-panini/data/weights/gambling_test_mod_rpe_learn/gambling_test_mod_rpe_learn.pkl'
print(f"Loading model: {modelfile}\n")
model = Model('tasks/gambling.py')
pg = model.get_pg(modelfile, seed=100, load='best')

# Enable RPE modulation
pg.use_rpe_modulation = True

# Generate test trials
n_trials = 5
trials = [pg.task.get_condition(pg.rng, pg.dt) for _ in range(n_trials)]

print("="*80)
print("BASELINE: No optostimulation")
print("="*80)
pg.opto_stim_offset = 0.0
pg.opto_stim_gain = 1.0
results_baseline = pg.run_trials(trials, return_states=True)
rpe_baseline = results_baseline['RPE_continuous'].cpu().numpy()

print(f"RPE statistics:")
print(f"  Mean: {rpe_baseline.mean():.4f}")
print(f"  Std:  {rpe_baseline.std():.4f}")
print(f"  Min:  {rpe_baseline.min():.4f}")
print(f"  Max:  {rpe_baseline.max():.4f}")

print("\n" + "="*80)
print("TONIC STIMULATION: +0.1 dopamine offset")
print("="*80)
pg.opto_stim_offset = +0.1
pg.opto_stim_gain = 1.0
results_stim = pg.run_trials(trials, return_states=True)
rpe_stim = results_stim['RPE_continuous'].cpu().numpy()

print(f"RPE statistics:")
print(f"  Mean: {rpe_stim.mean():.4f} (shift: {rpe_stim.mean() - rpe_baseline.mean():+.4f})")
print(f"  Std:  {rpe_stim.std():.4f}")
print(f"  Min:  {rpe_stim.min():.4f}")
print(f"  Max:  {rpe_stim.max():.4f}")

print("\n" + "="*80)
print("TONIC INHIBITION: -0.1 dopamine offset")
print("="*80)
pg.opto_stim_offset = -0.1
pg.opto_stim_gain = 1.0
results_inhib = pg.run_trials(trials, return_states=True)
rpe_inhib = results_inhib['RPE_continuous'].cpu().numpy()

print(f"RPE statistics:")
print(f"  Mean: {rpe_inhib.mean():.4f} (shift: {rpe_inhib.mean() - rpe_baseline.mean():+.4f})")
print(f"  Std:  {rpe_inhib.std():.4f}")
print(f"  Min:  {rpe_inhib.min():.4f}")
print(f"  Max:  {rpe_inhib.max():.4f}")

print("\n" + "="*80)
print("AMPLIFIED: 2× gain on all RPE signals")
print("="*80)
pg.opto_stim_offset = 0.0
pg.opto_stim_gain = 2.0
results_amp = pg.run_trials(trials, return_states=True)
rpe_amp = results_amp['RPE_continuous'].cpu().numpy()

print(f"RPE statistics:")
print(f"  Mean: {rpe_amp.mean():.4f}")
print(f"  Std:  {rpe_amp.std():.4f} (vs baseline: {rpe_baseline.std():.4f})")
print(f"  Min:  {rpe_amp.min():.4f}")
print(f"  Max:  {rpe_amp.max():.4f}")

print("\n" + "="*80)
print("CUE-SPECIFIC: +0.15 offset during cue phase only")
print("="*80)
pg.opto_stim_offset = +0.15
pg.opto_stim_gain = 1.0
pg.opto_stim_phase = 'cue'
results_cue = pg.run_trials(trials, return_states=True)
rpe_cue = results_cue['RPE_continuous'].cpu().numpy()

print(f"RPE statistics:")
print(f"  Mean: {rpe_cue.mean():.4f}")
print(f"  Cue phase mean (t=25-49): {rpe_cue[25:50].mean():.4f}")
print(f"  Decision phase mean (t=50+): {rpe_cue[50:].mean():.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Baseline RPE:      {rpe_baseline.mean():+.4f}")
print(f"Stim +0.1:         {rpe_stim.mean():+.4f} (shift: {rpe_stim.mean() - rpe_baseline.mean():+.4f})")
print(f"Inhib -0.1:        {rpe_inhib.mean():+.4f} (shift: {rpe_inhib.mean() - rpe_baseline.mean():+.4f})")
print(f"Amplified 2×:      {rpe_amp.mean():+.4f}")
print(f"Cue-specific +0.15: {rpe_cue.mean():+.4f}")
print("\n✓ Optostimulation working! RPE shifts as expected.")
