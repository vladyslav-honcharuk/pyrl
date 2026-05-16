#!/usr/bin/env python
"""
Generate mega-comparison plot for contextual kappa values using Level 2 model.

This generates trial data for context levels from -1 to 1, then uses the
existing gambling.py plotting functions to create the exact same 8×11 plot
with behavioral maps, value predictions, regression coefficients, etc.

Usage:
    python generate_contextual_kappa_comparison.py [output_path]

Examples:
    python generate_contextual_kappa_comparison.py data/figures/gambling/mega_comparison_all_kappas.png
    python generate_contextual_kappa_comparison.py
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np
from pyrl.model import Model
from pyrl import utils
from pyrl.distributional_utils import interpolate_quantiles, context_to_quantile_idx

# Import plotting functions
sys.path.insert(0, 'scripts/plotting')
from gambling import (
    plot_mega_comparison, _load_comparison_data, _compute_weight_limits,
    _plot_row_behavior, _plot_row_values, _plot_row_regression,
    _plot_row_weights, _plot_row_beta_vs_weights, load_model_weights,
    convert_actions, compute_deltas, extract_choices, compute_value_grid,
    regress_neurons, to_numpy, load_trial_data
)
import matplotlib.pyplot as plt

def find_level_model(level=3, base_dir='data/weights'):
    """Find Level 2 or Level 3 model."""
    if level == 3:
        possible_names = ['gambling_level3_dist_full']
    else:
        possible_names = ['gambling_level2_dist_context_q']
    
    for name in possible_names:
        paths = [
            os.path.join(base_dir, name, f'{name}.pkl'),
            os.path.join(base_dir, 'gambling', f'{name}.pkl'),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return name, path
    
    return None, None

def generate_trials_for_context(pg, context_level, n_trials=500, seed_base=42):
    """
    Generate trial data for a specific context level.
    
    Returns data in same format as load_trial_data():
    (trials, U, Z, Z_b, A, R, M, perf, r_policy, r_value, RPE_obj, RPE_subj)
    
    For distributional models, extract the context-selected quantile from Z_b.
    """
    
    # Use different seed based on context level to get different trials
    # This ensures firing rates vary across contexts
    seed = seed_base + int((context_level + 1.5) * 100)  # Convert context to unique seed
    
    rng = np.random.RandomState(seed)
    
    # Generate trials
    trial_list = [pg.task.get_condition(rng, pg.dt) for _ in range(n_trials)]
    
    # Run with state tracking and context input for value biasing
    # Context is passed to modulate policy via opponent mechanism
    results = pg.run_trials(trial_list, return_states=True, context_input=context_level)
    
    # Extract baseline and handle distributional vs single-value
    Z_b = results['Z_b']  # (T, B) or (T, B, n_quantiles)
    if Z_b.ndim == 3:
        # Distributional: use context-based quantile selection with interpolation
        # (same as training code)
        n_quantiles = Z_b.shape[2]
        
        # Map context to quantile index using same function as training
        context_tensor = torch.tensor(context_level, dtype=torch.float32, device='cpu')
        quantile_idx = context_to_quantile_idx(context_tensor, n_quantiles)
        
        # Interpolate between quantiles (same as training)
        Z_b_interpolated = interpolate_quantiles(Z_b, quantile_idx)
        Z_b = Z_b_interpolated
    
    # Extract what we need
    data = (
        trial_list,
        results['U'],           # Inputs
        results['Z'],           # Policy outputs
        Z_b,                    # Baseline outputs (now single-valued, context-selected)
        results['A'],           # Actions
        results['R'],           # Rewards
        results['M'],           # Masking
        results['perf'],        # Performance
        results['r_policy'],    # Policy firing rates
        results['r_value'],     # Value firing rates
        results['RPE_objective'],  # RPE
        results['RPE_subjective']   # RPE subjective
    )
    
    return data

def main():
    # Find and load Level 3 model
    model_name, model_path = find_level_model(level=3)
    
    if not model_path:
        print("❌ Level 3 model not found!")
        print("   Expected: gambling_level3_dist_full")
        sys.exit(1)
    
    print(f"Loading Level 3 model: {model_path}")
    print()
    
    model = Model('tasks/gambling.py')
    pg = model.get_pg(model_path, seed=1, device='cpu')
    
    # Generate trials for context levels
    context_levels = np.linspace(-1.0, 1.0, 11)  # 11 levels
    
    print(f"Generating {len(context_levels)} trial datasets...")
    print()
    
    trialsfiles = {}
    modelfiles = {}
    
    for context_level in context_levels:
        print(f"  Context {context_level:+.1f}...", end=' ')
        try:
            # Generate trial data
            trial_data = generate_trials_for_context(pg, context_level, n_trials=500)
            
            # Save to temp location
            trials_dir = 'data/trials/gambling_temp'
            os.makedirs(trials_dir, exist_ok=True)
            
            # Format context level for filename (replace + sign)
            context_str = f'{context_level:+.1f}'.replace('+', 'pos').replace('-', 'neg').replace('.', 'p')
            trialsfile = os.path.join(trials_dir, f'trials_context_{context_str}.pkl')
            
            import pickle
            with open(trialsfile, 'wb') as f:
                pickle.dump(trial_data, f)
            
            trialsfiles[context_level] = trialsfile
            modelfiles[context_level] = model_path
            
            print(f"✓")
        except Exception as e:
            print(f"✗ {e}")
    
    print()
    print(f"✓ Generated {len(trialsfiles)} trial datasets")
    print()
    
    # Now use the existing mega_comparison plotting function
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'data/figures/gambling/mega_comparison_all_kappas.png'
    figspath = os.path.dirname(output_path)
    os.makedirs(figspath, exist_ok=True)
    
    print(f"Creating mega-comparison plot...")
    print(f"Output: {output_path}")
    print()
    
    # Call the existing plotting function with ALL context levels (not just default kappas)
    # Override the expected_kappas in the function by modifying it locally
    kappa_values = sorted(context_levels)
    
    print(f"\nUsing {len(kappa_values)} context levels: {[f'{k:+.1f}' for k in kappa_values]}")
    print()
    
    # Manually create the plot instead of calling plot_mega_comparison
    # since it hardcodes expected_kappas
    print(f"Creating 8×{len(kappa_values)} mega-plot...")
    
    from gambling import _load_comparison_data, _compute_weight_limits, _plot_row_behavior, _plot_row_values
    from gambling import _plot_row_regression, _plot_row_weights, _plot_row_beta_vs_weights
    
    all_data, baseline_data = _load_comparison_data(kappa_values, trialsfiles, modelfiles)
    
    if baseline_data is None:
        print("Error: No baseline (κ=0) data found!")
        # Try with 0.0 if it exists
        if 0.0 in all_data:
            baseline_data = all_data[0.0]
        else:
            print("Cannot find baseline data!")
            return
    
    policy_lim, value_lim = _compute_weight_limits(all_data, baseline_data, kappa_values)
    
    fig = plt.figure(figsize=(4*len(kappa_values), 24))
    gs = fig.add_gridspec(8, len(kappa_values), hspace=0.35, wspace=0.25)
    
    def get_title(kappa):
        return f'κ={kappa:+.2f}'
    
    # Row 0: Behavioral heatmaps
    _plot_row_behavior(fig, gs, 0, kappa_values, all_data, get_title)
    # Row 1: Predicted values
    _plot_row_values(fig, gs, 1, kappa_values, all_data)
    # Row 2: Policy regression
    _plot_row_regression(fig, gs, 2, kappa_values, all_data, 'policy', 'Policy\nβEV')
    # Row 3: Value regression
    _plot_row_regression(fig, gs, 3, kappa_values, all_data, 'value', 'Value\nβEV')
    # Row 4: Policy output weights
    _plot_row_weights(fig, gs, 4, kappa_values, all_data, baseline_data,
                      'policy', 'red', policy_lim, 'Policy\nOutput Weight',
                      'Policy Output Weight (κ=0)')
    # Row 5: Value output weights
    _plot_row_weights(fig, gs, 5, kappa_values, all_data, baseline_data,
                      'value', 'blue', value_lim, 'Value\nOutput Weight',
                      'Value Output Weight (κ=0)')
    # Row 6: Policy β vs weights
    _plot_row_beta_vs_weights(fig, gs, 6, kappa_values, all_data, 'policy', 'red', policy_lim)
    # Row 7: Value β vs weights
    _plot_row_beta_vs_weights(fig, gs, 7, kappa_values, all_data, 'value', 'blue', value_lim)
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    
    savefile = output_path
    plt.savefig(savefile, dpi=300, bbox_inches='tight')
    print(f"\nSaved mega-plot to {savefile}\n")
    plt.close()
    
    print()
    print("✓ Mega-comparison plot created successfully!")
    
    # Cleanup temp files
    print()
    print("Cleaning up temporary trial files...")
    for trialsfile in trialsfiles.values():
        try:
            os.remove(trialsfile)
        except:
            pass

if __name__ == '__main__':
    main()
