#!/usr/bin/env python
"""
Main script for training and running cognitive task models.

Usage:
    python do.py <model_file> <action> [options]

Actions:
    info     - Display model information
    train    - Train the model
    finetune - Fine-tune a pre-trained model with new kappa value
    run      - Run analysis on trained model

Examples:
    python do.py models/rdm_fixed.py info
    python do.py models/rdm_fixed.py train --seed 1
    python do.py models/rdm_fixed.py train --gpu
    python do.py models/rdm_fixed.py finetune --kappa 0.5 --suffix _kappa0p5
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from pyrl import utils
from pyrl.model import Model


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and run cognitive task models')
    parser.add_argument('model_file', help='Model specification file')
    parser.add_argument('action', nargs='?', type=str, default='info',
                       help='Action to perform (info/train/finetune/run)')
    parser.add_argument('args', nargs='*', help='Additional arguments')
    parser.add_argument('--dt', type=float, default=0,
                       help='Time step (ms). Default: use config value')
    parser.add_argument('--dt-save', type=float, default=0,
                       help='Time step for saving trial data (ms). Default: use dt value')
    parser.add_argument('--seed', type=int, default=100,
                       help='Random seed')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix for output files')
    parser.add_argument('--gpu', action='store_true', default=False,
                       help='Use GPU if available (auto-detects CUDA or MPS)')
    parser.add_argument('--device', type=str, default=None,
                       help='Specific device (e.g., cuda, cuda:0, mps, cpu)')
    parser.add_argument('--kappa', type=float, default=None,
                       help='Risk-sensitivity parameter: -1 (risk-averse) to +1 (risk-seeking)')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pre-trained model weights (for finetune action). '
                            'If not specified, automatically uses base model name without suffix.')
    parser.add_argument('--finetune-iter', type=int, default=None,
                       help='Number of iterations for fine-tuning (default: use model config)')
    parser.add_argument('--finetune-lr', type=float, default=None,
                       help='Learning rate for fine-tuning (default: use pretrained model lr)')
    parser.add_argument('--grad-clip', type=float, default=None,
                       help='Gradient clipping threshold for policy network (default: no clipping)')
    parser.add_argument('--baseline-grad-clip', type=float, default=None,
                       help='Gradient clipping threshold for baseline network (default: no clipping)')

    args = parser.parse_args()

    # Process arguments
    modelfile = os.path.abspath(args.model_file)
    if not modelfile.endswith('.py'):
        modelfile += '.py'

    action = args.action
    action_args = args.args
    dt = args.dt if args.dt > 0 else None
    dt_save = args.dt_save if args.dt_save > 0 else None
    seed = args.seed
    suffix = args.suffix

    # Determine device
    if args.device:
        device = args.device
    elif args.gpu:
        # Auto-detect best available GPU
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            print("Warning: --gpu specified but no GPU available, using CPU")
            device = 'cpu'
    else:
        device = 'cpu'

    print("=" * 80)
    print(f"MODELFILE: {modelfile}")
    print(f"ACTION:    {action}")
    print(f"ARGS:      {action_args}")
    print(f"SEED:      {seed}")
    print(f"SUFFIX:    {suffix}")
    print(f"DEVICE:    {device}")
    print("=" * 80)

    # Setup paths
    here = utils.get_here(__file__)
    prefix = os.path.basename(here)

    # Name to use
    name = os.path.splitext(os.path.basename(modelfile))[0] + suffix

    # Scratch directory for trials (can be set via SCRATCH env variable)
    scratchpath = os.environ.get('SCRATCH')
    if scratchpath is None:
        scratchpath = os.path.join(os.environ['HOME'], 'scratch')
    trialspath = os.path.join(scratchpath, 'work', 'pyrl', prefix, name)

    # Create work directories
    workpath = os.path.join(here, 'work')
    datapath = os.path.join(workpath, 'data', name)
    figspath = os.path.join(workpath, 'figs', name)

    for path in [datapath, figspath, trialspath]:
        utils.mkdir_p(path)

    # Savefile
    savefile = os.path.join(datapath, name + '.pkl')

    # Execute action
    if action == 'info':
        # Display model information
        model = Model(modelfile)
        # Use config if savefile doesn't exist, otherwise load from file
        if os.path.exists(savefile):
            pg = model.get_pg(savefile, seed, dt=dt, device=device)
        else:
            pg = model.get_pg(model.config, seed, dt=dt, device=device)

        print("\n" + "=" * 80)
        print("MODEL INFORMATION")
        print("=" * 80)
        print(f"\nPolicy network: {pg.config['network_type']}")
        print(f"Baseline network: {pg.config.get('baseline_network_type', pg.config['network_type'])}")
        print(f"\nPolicy network size: {pg.config['N']} units")
        print(f"Baseline network size: {pg.config['baseline_N']} units")
        print(f"\nInputs: {len(pg.config['inputs'])}")
        print(f"Actions: {len(pg.config['actions'])}")
        print(f"\nTime step: {pg.dt} ms")
        print(f"Max time: {pg.config['tmax']} ms")
        print("=" * 80)

    elif action == 'train':
        # Train model
        model = Model(modelfile)
        recover = 'recover' in action_args
        model.train(savefile, seed, recover=recover, device=device, kappa=args.kappa)

    elif action == 'finetune':
        # Fine-tune model with new kappa value
        if args.kappa is None:
            print("Error: --kappa is required for finetune action")
            sys.exit(1)

        # Determine pretrained file
        if args.pretrained:
            pretrained_file = args.pretrained
            print(f"Using specified pre-trained weights: {pretrained_file}")
        else:
            # Default: use the base model name (without any suffix) for pretrained file
            base_name = os.path.splitext(os.path.basename(modelfile))[0]
            # Look in the directory without suffix (the original pre-trained model)
            pretrained_datapath = os.path.join(workpath, 'data', base_name)
            pretrained_file = os.path.join(pretrained_datapath, base_name + '.pkl')
            print(f"Auto-detecting pre-trained weights: {pretrained_file}")

        if not os.path.exists(pretrained_file):
            print(f"\nError: Pre-trained file not found: {pretrained_file}")
            print("\nOptions:")
            print("  1. Train a base model first (without --kappa or with --kappa 0)")
            print("  2. Specify custom path with --pretrained /path/to/model.pkl")
            sys.exit(1)

        model = Model(modelfile)
        model.finetune(pretrained_file, savefile, args.kappa, seed=seed,
                      max_iter=args.finetune_iter, lr=args.finetune_lr,
                      grad_clip=args.grad_clip, baseline_grad_clip=args.baseline_grad_clip,
                      device=device)

    elif action == 'run':
        # Get analysis script
        try:
            runfile = action_args[0]
        except IndexError:
            print("Please specify the analysis script.")
            sys.exit(1)
        if not runfile.endswith('.py'):
            runfile += '.py'

        # Load analysis module
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("analysis", runfile)
            analysis_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(analysis_module)
        except IOError:
            print(f"Couldn't load analysis module from {runfile}")
            sys.exit(1)

        # Load model
        model = Model(modelfile)

        # Reset args
        action_args = action_args[1:]
        if len(action_args) > 0:
            run_action = action_args[0]
            run_args = action_args[1:]
        else:
            run_action = None
            run_args = []

        # Copy the savefile for safe access
        if os.path.isfile(savefile):
            import shutil
            base, ext = os.path.splitext(savefile)
            savefile_copy = base + '_copy.pkl'
            while True:
                shutil.copy(savefile, savefile_copy)
                try:
                    utils.load(savefile_copy)
                    break
                except EOFError:
                    continue
        else:
            print(f"File {savefile} doesn't exist.")
            sys.exit(1)

        # Pass everything on to the analysis module
        config = {
            'seed': 1,
            'suffix': suffix,
            'model': model,
            'savefile': savefile_copy,
            'datapath': datapath,
            'figspath': figspath,
            'trialspath': trialspath
        }

        if dt is not None and dt > 0:
            config['dt'] = dt
        else:
            config['dt'] = None

        if dt_save is not None and dt_save > 0:
            config['dt-save'] = dt_save
        else:
            config['dt-save'] = None

        try:
            analysis_module.do(run_action, run_args, config)
        except SystemExit as e:
            print(f"Error: {e.code}")
            raise

    else:
        print(f"Unrecognized action '{action}'.")
        print("Valid actions: info, train, finetune, run")
        sys.exit(1)


if __name__ == '__main__':
    main()
