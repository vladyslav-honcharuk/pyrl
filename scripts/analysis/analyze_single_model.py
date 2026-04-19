#!/usr/bin/env python3
"""
Run comprehensive analysis on a single model.

This script analyzes a single trained model with optional suffix:
- Generates trial data (trials-a and trials-b)
- Creates behavioral heatmaps
- Analyzes value network neurons
- Plots temporal activity
- Regression scatter plots
- Neural analysis

Usage:
    python3 scripts/analysis/analyze_single_model.py models/gambling.py
    python3 scripts/analysis/analyze_single_model.py models/gambling.py --suffix _gaussian_kappa
    python3 scripts/analysis/analyze_single_model.py models/gambling.py --suffix pos0p5 --kappa 0.5
"""

import os
import sys
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, 'scripts', 'training', 'train.py')

def run_single_analysis(modelfile, suffix, analysis_action, args_list):
    """Run a single analysis command for a model."""
    # Build command with suffix if needed
    cmd = ['python3', TRAIN_SCRIPT, modelfile]
    if suffix:
        cmd.extend(['--suffix', suffix])
    cmd.extend(['run', 'analysis/gambling.py', analysis_action] + args_list)

    print(f"\n  [{analysis_action}] Running...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"  [{analysis_action}] ✗ FAILED (exit code {result.returncode})")
        return False

    print(f"  [{analysis_action}] ✓ Complete")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive analysis on a single trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze base model (no suffix)
  python3 scripts/analysis/analyze_single_model.py models/gambling.py

  # Analyze model with Gaussian per-neuron kappa
  python3 scripts/analysis/analyze_single_model.py models/gambling.py --suffix _gaussian_kappa

  # Analyze single-kappa model at kappa=0.5
  python3 scripts/analysis/analyze_single_model.py models/gambling.py --suffix pos0p5 --kappa 0.5

  # Analyze with specific kappa and custom number of trials
  python3 scripts/analysis/analyze_single_model.py models/gambling.py --suffix _uniform_kappa --kappa 0.0 --trials 10

  # Skip certain analysis steps
  python3 scripts/analysis/analyze_single_model.py models/gambling.py --skip-trials --skip-temporal

"""
    )
    parser.add_argument('model_file', help='Model specification file')
    parser.add_argument('--suffix', type=str, default='',
                       help='Model suffix (e.g., _gaussian_kappa, pos0p5)')
    parser.add_argument('--kappa', type=float, default=0.0,
                       help='Kappa value for analysis plots (default: 0.0)')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of trials per condition to generate (default: 5)')
    parser.add_argument('--temporal-samples', type=int, default=3,
                       help='Number of sample trials for temporal plots (default: 3)')

    # Flags to skip certain analyses
    parser.add_argument('--skip-trials', action='store_true',
                       help='Skip trial generation (trials-a and trials-b)')
    parser.add_argument('--skip-behavior', action='store_true',
                       help='Skip behavioral heatmap')
    parser.add_argument('--skip-value-neurons', action='store_true',
                       help='Skip value neuron analysis')
    parser.add_argument('--skip-temporal', action='store_true',
                       help='Skip temporal activity plots')
    parser.add_argument('--skip-regression', action='store_true',
                       help='Skip regression scatter plots')
    parser.add_argument('--skip-neural', action='store_true',
                       help='Skip neural analysis')

    args = parser.parse_args()

    modelfile = args.model_file

    # Validate model file
    if not modelfile:
        print("Error: model_file is required")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(modelfile):
        print(f"Error: Model file not found: {modelfile}")
        print("\nMake sure the model file path is correct.")
        print("Example: python3 scripts/analysis/analyze_single_model.py models/gambling.py")
        sys.exit(1)

    suffix = args.suffix
    kappa = args.kappa
    num_trials = args.trials
    temporal_samples = args.temporal_samples

    base_name = os.path.splitext(os.path.basename(modelfile))[0]

    # Handle suffix formatting - ensure it doesn't duplicate
    # If suffix doesn't start with underscore and base_name isn't already in it, add separator
    if suffix:
        # Remove base_name prefix if present (e.g., "gambling_gaussian_kappa" -> "_gaussian_kappa")
        if suffix.startswith(base_name):
            # Extract the actual suffix part
            suffix_part = suffix[len(base_name):]
            if suffix_part and not suffix_part.startswith('_'):
                suffix = '_' + suffix_part
            else:
                suffix = suffix_part
        elif not suffix.startswith('_'):
            suffix = '_' + suffix

    model_name = f"{base_name}{suffix}" if suffix else base_name
    model_path = os.path.join(REPO_ROOT, "data", "weights", model_name, f"{model_name}.pkl")

    print("=" * 80)
    print("SINGLE MODEL ANALYSIS")
    print("=" * 80)
    print(f"Model file: {modelfile}")
    print(f"Model name: {model_name}")
    print(f"Suffix: {suffix if suffix else '(none)'}")
    print(f"Kappa (for plots): {kappa:+.2f}")
    print(f"Model path: {model_path}")
    print("=" * 80)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n✗ Error: Model not found at {model_path}")
        print("\nMake sure you've trained the model first:")
        if suffix:
            print(f"  python3 scripts/training/train.py {modelfile} train --suffix {suffix}")
        else:
            print(f"  python3 scripts/training/train.py {modelfile} train")
        sys.exit(1)

    print(f"\n✓ Model found: {model_path}")

    # List of analysis steps
    analysis_steps = []

    if not args.skip_trials:
        analysis_steps.append({
            'name': 'trials-a',
            'action': 'trials-a',
            'args': [str(num_trials)],
            'description': f'Generate {num_trials} trials per condition (type A)'
        })
        analysis_steps.append({
            'name': 'trials-b',
            'action': 'trials-b',
            'args': [str(num_trials)],
            'description': f'Generate {num_trials} trials per condition (type B)'
        })

    if not args.skip_behavior:
        analysis_steps.append({
            'name': 'behavior',
            'action': 'behavior',
            'args': [],
            'description': 'Generate behavioral heatmap'
        })

    if not args.skip_value_neurons:
        analysis_steps.append({
            'name': 'value-neurons',
            'action': 'value-neurons',
            'args': [str(kappa)],
            'description': 'Analyze value network neurons'
        })

    if not args.skip_temporal:
        analysis_steps.append({
            'name': 'temporal-activity',
            'action': 'temporal-activity',
            'args': ['value', str(temporal_samples), str(kappa)],
            'description': f'Plot temporal activity ({temporal_samples} sample trials)'
        })

    if not args.skip_regression:
        analysis_steps.append({
            'name': 'regression-scatter',
            'action': 'regression-scatter',
            'args': ['policy', str(kappa)],
            'description': 'Generate regression scatter plots'
        })

    if not args.skip_neural:
        analysis_steps.append({
            'name': 'neural-analysis',
            'action': 'neural-analysis',
            'args': [str(kappa)],
            'description': 'Run neural analysis'
        })

    print(f"\nAnalysis steps to run ({len(analysis_steps)} total):")
    for i, step in enumerate(analysis_steps, 1):
        print(f"  {i}. {step['name']}: {step['description']}")

    print("\n" + "=" * 80)
    print("RUNNING ANALYSES")
    print("=" * 80)

    # Run each analysis step
    results = []
    for i, step in enumerate(analysis_steps, 1):
        print(f"\n[{i}/{len(analysis_steps)}] {step['name'].upper()}")
        print("-" * 80)

        success = run_single_analysis(
            modelfile,
            suffix,
            step['action'],
            step['args']
        )

        results.append((step['name'], success))

        if not success:
            print(f"\n⚠️  Analysis step '{step['name']}' failed. Continuing with remaining steps...")

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\nResults:")
    for name, success in results:
        status = "✓" if success else "✗ FAILED"
        print(f"  {status}  {name}")

    print(f"\n{'=' * 80}")
    print(f"Total successful: {len(successful)}/{len(results)}")
    print(f"Total failed: {len(failed)}")

    if failed:
        print(f"\n⚠️  Some analyses failed:")
        for name, _ in failed:
            print(f"  - {name}")
    else:
        print(f"\n🎉 All {len(results)} analyses completed successfully!")

    print(f"\n💾 Results saved in:")
    print(f"  Figures: data/figures/{model_name}/")
    print(f"  Data: data/trials/{model_name}/")
    print(f"{'=' * 80}\n")

    # Exit with error code if any failed
    sys.exit(0 if len(failed) == 0 else 1)

if __name__ == '__main__':
    main()
