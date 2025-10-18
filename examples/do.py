#!/usr/bin/env python
"""
Main script for training and running cognitive task models.

Usage:
    python do.py <model_file> <action> [options]

Actions:
    info  - Display model information
    train - Train the model
    run   - Run analysis on trained model

Examples:
    python do.py models/rdm_fixed.py info
    python do.py models/rdm_fixed.py train --seed 1
    python do.py models/rdm_fixed.py train --gpu
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pyrl import utils
from pyrl.model import Model


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and run cognitive task models')
    parser.add_argument('model_file', help='Model specification file')
    parser.add_argument('action', nargs='?', type=str, default='info',
                       help='Action to perform (info/train/run)')
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
                       help='Use GPU if available')
    parser.add_argument('--device', type=str, default=None,
                       help='Specific device (e.g., cuda:0)')

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
        device = 'mps'
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
        model.train(savefile, seed, recover=recover, device=device)

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
        print("Valid actions: info, train, run")
        sys.exit(1)


if __name__ == '__main__':
    main()
