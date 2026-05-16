#!/usr/bin/env python
"""
Create a mega-comparison plot for contextual kappa values from -1 to 1.

This is similar to the kappa sweep plots but for distributional models
with context-based quantile selection showing different "effective" kappa
values based on dopamine context modulation.

Usage:
    python plot_contextual_kappa_comparison.py [output_path]

Examples:
    python plot_contextual_kappa_comparison.py data/figures/gambling/contextual_kappa_comparison.png
    python plot_contextual_kappa_comparison.py                    # Use default path
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import numpy as np
from pyrl.model import Model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress

def find_level2_model(base_dir='data/weights'):
    """Find Level 2 (context quantile selection) model."""
    possible_names = ['gambling_level2_dist_context_q', 'gambling_level2_contextual']
    
    for name in possible_names:
        paths = [
            os.path.join(base_dir, name, f'{name}.pkl'),
            os.path.join(base_dir, 'gambling', f'{name}.pkl'),
            os.path.join(base_dir, f'{name}.pkl'),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return name, path
    
    return None, None

def run_with_context_kappa(pg, context_level, n_trials=100, seed=42):
    """
    Run trials with Level 2 model at a specific context level.
    
    Returns behavior statistics simulating different "effective" kappa values
    based on context-selected quantiles.
    """
    
    rng = np.random.RandomState(seed)
    trials = [pg.task.get_condition(rng, pg.dt) for _ in range(n_trials)]
    results = pg.run_trials(trials, return_states=False)
    
    # Get baseline outputs
    Z_b = results['Z_b']  # (T, B, n_quantiles)
    A = results['A']      # (T, B, n_actions)
    R = results['R']      # (T, B)
    
    # Compute quantile index from context
    n_quantiles = Z_b.shape[2]
    quantile_idx = int((context_level + 1) / 2 * n_quantiles)
    quantile_idx = np.clip(quantile_idx, 0, n_quantiles - 1)
    
    # Extract selected quantile
    selected_Q = Z_b[:, :, quantile_idx].cpu().numpy()  # (T, B)
    
    # Compute statistics
    action_counts = A.sum(dim=0).cpu().numpy()  # (B, n_actions)
    total_actions = A.shape[0] * A.shape[1]
    
    # Action probabilities
    action_probs = action_counts.sum(axis=0) / total_actions
    
    # Trial outcomes
    trial_rewards = R.sum(dim=0).cpu().numpy()  # Cumulative reward per trial
    
    # Compute choice direction bias
    choice_left = action_counts[:, 1].sum()  # CHOOSE-LEFT
    choice_right = action_counts[:, 2].sum()  # CHOOSE-RIGHT
    total_choices = choice_left + choice_right
    if total_choices > 0:
        choice_bias = (choice_right - choice_left) / total_choices
    else:
        choice_bias = 0
    
    # Compute value sensitivity (correlation between selected value and choice)
    mean_value_by_t = np.mean(selected_Q, axis=1)
    
    # Get first choice timesteps
    first_choices = []
    for b in range(A.shape[1]):
        for t in range(A.shape[0]):
            if A[t, b, 1].item() + A[t, b, 2].item() > 0:  # CHOOSE-LEFT or CHOOSE-RIGHT
                first_choices.append((t, A[t, b, 2].item()))  # (timestep, is_right)
                break
    
    return {
        'context_level': context_level,
        'action_probs': action_probs,
        'choice_bias': choice_bias,
        'mean_value': np.mean(selected_Q),
        'value_std': np.std(selected_Q),
        'trial_rewards': trial_rewards,
        'quantile_idx': quantile_idx,
        'selected_Q': selected_Q,
        'mean_reward': np.mean(trial_rewards),
        'success_rate': np.mean(trial_rewards > 0)
    }

def main():
    # Find Level 2 model
    model_name, model_path = find_level2_model()
    
    if not model_path:
        print("❌ Level 2 (context quantile selection) model not found!")
        print("   Expected: gambling_level2_dist_context_q")
        sys.exit(1)
    
    print(f"Loading Level 2 model: {model_path}")
    print()
    
    model = Model('tasks/gambling.py')
    pg = model.get_pg(model_path, seed=1, device='cpu')
    
    # Run trials for range of contextual kappa values
    context_kappas = np.linspace(-1.0, 1.0, 11)  # -1.0 to 1.0 in 0.2 steps
    
    print("Running trials with contextual dopamine levels...")
    all_results = []
    
    for context_level in context_kappas:
        print(f"  Context {context_level:+.1f}...", end=' ')
        try:
            result = run_with_context_kappa(pg, context_level, n_trials=100, seed=42)
            all_results.append(result)
            print(f"✓ (τ_idx={result['quantile_idx']}, reward={result['mean_reward']:.3f})")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not all_results:
        print("❌ No results generated!")
        sys.exit(1)
    
    print()
    print(f"✓ Generated {len(all_results)} context-kappa datasets")
    print()
    
    # Create mega-style comparison plot
    print("Creating mega-comparison plot...")
    
    output_path = sys.argv[1] if len(sys.argv) > 1 else 'data/figures/gambling/contextual_kappa_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create 4x11 figure (4 rows of analysis, 11 columns for context levels)
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(4, len(all_results), hspace=0.4, wspace=0.3)
    
    # Row 0: Selected quantile value distribution
    print("  Row 0: Selected quantile distributions")
    for i, result in enumerate(all_results):
        ax = fig.add_subplot(gs[0, i])
        
        Q_data = result['selected_Q']
        ax.hist(Q_data.flatten(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_title(f"κ={result['context_level']:+.1f}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Q value', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Add mean line
        mean_val = result['mean_value']
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'μ={mean_val:.2f}')
        ax.legend(fontsize=7)
    
    # Row 1: Action selection probabilities
    print("  Row 1: Action distributions")
    for i, result in enumerate(all_results):
        ax = fig.add_subplot(gs[1, i])
        
        action_names = ['Fixate', 'Left', 'Right']
        colors = ['gray', 'steelblue', 'coral']
        
        bars = ax.bar(action_names, result['action_probs'], color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_title(f"κ={result['context_level']:+.1f}", fontsize=10, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for bar, prob in zip(bars, result['action_probs']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{prob:.1%}',
                   ha='center', va='bottom', fontsize=8)
    
    # Row 2: Trial rewards distribution
    print("  Row 2: Reward distributions")
    for i, result in enumerate(all_results):
        ax = fig.add_subplot(gs[2, i])
        
        rewards = result['trial_rewards']
        ax.hist(rewards, bins=20, color='mediumseagreen', alpha=0.7, edgecolor='black')
        
        ax.set_title(f"κ={result['context_level']:+.1f}\nμ={result['mean_reward']:.3f}", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Cumulative reward', fontsize=8)
        ax.set_ylabel('# trials', fontsize=8)
        ax.grid(alpha=0.3)
    
    # Row 3: Summary statistics heatmap
    print("  Row 3: Summary statistics")
    
    # Collect all summary stats
    contexts = np.array([r['context_level'] for r in all_results])
    mean_values = np.array([r['mean_value'] for r in all_results])
    mean_rewards = np.array([r['mean_reward'] for r in all_results])
    success_rates = np.array([r['success_rate'] for r in all_results])
    choice_bias = np.array([r['choice_bias'] for r in all_results])
    
    # Left/Right choice probabilities
    left_probs = np.array([r['action_probs'][1] for r in all_results])
    right_probs = np.array([r['action_probs'][2] for r in all_results])
    
    # Summary data
    summary_data = np.array([
        mean_values / np.max(np.abs(mean_values)) if np.max(np.abs(mean_values)) > 0 else mean_values,
        mean_rewards / np.max(np.abs(mean_rewards)) if np.max(np.abs(mean_rewards)) > 0 else mean_rewards,
        success_rates,
        choice_bias / np.max(np.abs(choice_bias) + 1e-10) if np.max(np.abs(choice_bias)) > 0 else choice_bias,
    ])
    
    ax = fig.add_subplot(gs[3, :])
    
    im = ax.imshow(summary_data, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels([f'{c:+.1f}' for c in contexts], fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(['Mean V(s)', 'Mean Reward', 'Success Rate', 'Choice Bias'], fontsize=10)
    ax.set_xlabel('Context Dopamine Level (κ)', fontsize=11)
    ax.set_title('Summary Statistics across Context Levels', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Normalized value', fontsize=10)
    
    # Add text annotations
    for i in range(summary_data.shape[0]):
        for j in range(summary_data.shape[1]):
            value = summary_data[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontsize=8, fontweight='bold')
    
    # Main title
    fig.suptitle('Level 2: Context-Quantile Selection Across Dopamine Levels\nEffective κ Values Modulated by Contextual Dopamine Signal',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved mega-comparison plot to {output_path}")
    plt.close()
    
    print()
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    print("Context Dopamine Level Effects:")
    print(f"  Low dopamine (κ=-1.0):  uses pessimistic quantile → mean V = {all_results[0]['mean_value']:.3f}")
    print(f"  Neutral (κ=0.0):        uses median quantile     → mean V = {all_results[5]['mean_value']:.3f}")
    print(f"  High dopamine (κ=+1.0): uses optimistic quantile → mean V = {all_results[-1]['mean_value']:.3f}")
    print()
    print("✓ Contextual kappa comparison complete!")

if __name__ == '__main__':
    main()
