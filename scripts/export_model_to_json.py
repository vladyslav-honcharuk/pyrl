#!/usr/bin/env python3
"""
Export trained PyRL model to JSON format for web visualization.

Usage:
    python scripts/export_model_to_json.py <model_path.pkl> <output.json>

Example:
    python scripts/export_model_to_json.py data/weights/gamblingc_mod_policy/gamblingc_mod_policy.pkl model.json
"""
import sys
import json
import numpy as np
from pyrl import utils


def numpy_to_list(obj):
    """Recursively convert numpy arrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def export_model_to_json(model_path, output_path):
    """
    Export trained model to JSON format.

    Parameters
    ----------
    model_path : str
        Path to the .pkl model file
    output_path : str
        Path to save JSON file
    """
    # Load model
    print(f"Loading model from {model_path}...")
    saved_data = utils.load(model_path)

    # Extract best policy parameters
    policy_params = saved_data.get('best_policy_params', saved_data.get('current_policy_params'))

    if policy_params is None:
        print("Error: Could not find policy parameters in model file")
        print(f"Available keys: {list(saved_data.keys())}")
        sys.exit(1)

    # Extract configuration
    config = saved_data['config']

    # Build export data
    export_data = {
        'model_info': {
            'network_type': config.get('network_type', 'gru'),
            'N': int(config['N']),
            'Nin': int(config['Nin']),
            'Nout': int(config['Nout']),
            'dt': float(config['dt']),
            'tau': float(config['tau']),
            'alpha': float(config['dt'] / config['tau']),
            'kappa': float(config.get('kappa', 0.0)),
            'best_iter': int(saved_data.get('best_iter', 0)),
            'best_reward': float(saved_data.get('best_reward', 0)),
        },
        'weights': {
            'Win': numpy_to_list(policy_params['Win']),
            'bin': numpy_to_list(policy_params['bin']),
            'Wrec_gates': numpy_to_list(policy_params['Wrec_gates']),
            'Wrec': numpy_to_list(policy_params['Wrec']),
            'Wout': numpy_to_list(policy_params['Wout']),
            'bout': numpy_to_list(policy_params['bout']),
            'x0': numpy_to_list(policy_params['x0']),
        },
        'task_info': {
            'fixation_ms': 250,
            'stimulus_ms': 250,
            'decision_ms': 260,
            'dt': float(config['dt']),
        }
    }

    # Save to JSON
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print("Export complete!")
    print(f"  Network: {export_data['model_info']['network_type']}")
    print(f"  Hidden units: {export_data['model_info']['N']}")
    print(f"  Kappa: {export_data['model_info']['kappa']}")
    print(f"  Best reward: {export_data['model_info']['best_reward']:.3f}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    export_model_to_json(model_path, output_path)
