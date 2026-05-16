#!/usr/bin/env python
"""
Main script for training and running cognitive task models.

Usage:
    python scripts/training/train.py <task_file> <action> [options]

Actions:
    info     - Display model information
    train    - Train the model
    finetune - Fine-tune a pre-trained model with new kappa value
    run      - Run analysis on trained model

Examples:
    python scripts/training/train.py tasks/gambling.py info
    python scripts/training/train.py tasks/gambling.py train --seed 1
    python scripts/training/train.py tasks/gambling.py train --gpu
    python scripts/training/train.py tasks/gambling.py finetune --kappa 0.5 --suffix _kappa0p5
"""
import argparse
import os
import sys

# Add repository root to path so scripts import the local checkout.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from pyrl import utils
from pyrl.model import Model


def apply_context_sampling_args(model, args):
    """Apply command-line context sampling overrides to a model config."""
    if args.no_rpe_modulation:
        model.config['use_rpe_modulation'] = False
    if args.rpe_modulation:
        model.config['use_rpe_modulation'] = True
    if args.rpe_modulation_gain is not None:
        model.config['rpe_modulation_gain'] = args.rpe_modulation_gain
    if args.rpe_modulation_clamp is not None:
        model.config['rpe_modulation_clamp'] = args.rpe_modulation_clamp
    if args.context_distribution is not None:
        model.config['context_distribution'] = args.context_distribution
    if args.context_uniform_low is not None:
        model.config['context_uniform_low'] = args.context_uniform_low
    if args.context_uniform_high is not None:
        model.config['context_uniform_high'] = args.context_uniform_high
    if args.context_gaussian_mean is not None:
        model.config['context_gaussian_mean'] = args.context_gaussian_mean
    if args.context_gaussian_std is not None:
        model.config['context_gaussian_std'] = args.context_gaussian_std
    if args.vta_training_context:
        model.config['vta_training_context'] = True
        if not args.keep_training_context_input:
            model.config['training_context_input'] = False
    if args.vta_context_distribution is not None:
        model.config['vta_context_distribution'] = args.vta_context_distribution
    if args.vta_context_low is not None:
        model.config['vta_context_low'] = args.vta_context_low
    if args.vta_context_high is not None:
        model.config['vta_context_high'] = args.vta_context_high
    if args.vta_context_mean is not None:
        model.config['vta_context_mean'] = args.vta_context_mean
    if args.vta_context_std is not None:
        model.config['vta_context_std'] = args.vta_context_std
    if args.vta_context_weight is not None:
        model.config['vta_context_weight'] = args.vta_context_weight
    if args.dopamine_homogeneous_sensitivity:
        model.config['dopamine_heterogeneous_sensitivity'] = False
    if args.dopamine_sensitivity_min is not None:
        model.config['dopamine_sensitivity_min'] = args.dopamine_sensitivity_min
    if args.dopamine_sensitivity_max is not None:
        model.config['dopamine_sensitivity_max'] = args.dopamine_sensitivity_max
    if args.dopamine_sensitivity_learned:
        model.config['dopamine_sensitivity_learned'] = True
    if args.dopamine_bias:
        model.config['dopamine_bias_enabled'] = True
    if args.dopamine_bias_max_abs is not None:
        model.config['dopamine_bias_max_abs'] = args.dopamine_bias_max_abs
    if args.dopamine_modulation_mode is not None:
        model.config['dopamine_modulation_mode'] = args.dopamine_modulation_mode
    if args.dopamine_hill_base_da is not None:
        model.config['dopamine_hill_base_da'] = args.dopamine_hill_base_da
    if args.dopamine_hill_da_range is not None:
        model.config['dopamine_hill_da_range'] = args.dopamine_hill_da_range
    if args.dopamine_hill_ec50_d1 is not None:
        model.config['dopamine_hill_ec50_d1'] = args.dopamine_hill_ec50_d1
    if args.dopamine_hill_ec50_d2 is not None:
        model.config['dopamine_hill_ec50_d2'] = args.dopamine_hill_ec50_d2
    if args.dopamine_hill_coefficient is not None:
        model.config['dopamine_hill_coefficient'] = args.dopamine_hill_coefficient
    if args.dopamine_hill_gain_scale is not None:
        model.config['dopamine_hill_gain_scale'] = args.dopamine_hill_gain_scale
    if args.dopamine_learning_modulation_mode is not None:
        model.config['dopamine_learning_modulation_mode'] = args.dopamine_learning_modulation_mode
    if args.dopamine_learning_eta_min is not None:
        model.config['dopamine_learning_eta_min'] = args.dopamine_learning_eta_min
    if args.dopamine_learning_eta_max is not None:
        model.config['dopamine_learning_eta_max'] = args.dopamine_learning_eta_max
    if args.baseline_activity_balance is not None:
        model.config['baseline_activity_balance'] = args.baseline_activity_balance


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
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory for weights/figures/trials output (default: data)')
    parser.add_argument('--gpu', action='store_true', default=False,
                       help='Use GPU if available (auto-detects CUDA or MPS)')
    parser.add_argument('--device', type=str, default=None,
                       help='Specific device (e.g., cuda, cuda:0, mps, cpu)')
    parser.add_argument('--kappa', type=float, default=None,
                       help='Risk-sensitivity parameter: -1 (risk-averse) to +1 (risk-seeking)')
    parser.add_argument('--kappa-dist', type=str, default=None, choices=['gaussian', 'uniform'],
                       help='Distribution for per-neuron kappa values (gaussian or uniform)')
    parser.add_argument('--kappa-dist-mean', type=float, default=0.0,
                       help='Mean for Gaussian kappa distribution (default: 0.0)')
    parser.add_argument('--kappa-dist-std', type=float, default=0.1,
                       help='Standard deviation for Gaussian kappa distribution (default: 0.1)')
    parser.add_argument('--kappa-dist-low', type=float, default=-0.5,
                       help='Lower bound for uniform kappa distribution (default: -0.5)')
    parser.add_argument('--kappa-dist-high', type=float, default=0.5,
                       help='Upper bound for uniform kappa distribution (default: 0.5)')
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

    # Distributional RL level presets
    parser.add_argument('--level', type=int, default=None, choices=[0, 1, 2, 3, 4],
                       help='Distributional RL level preset (overrides individual flags): '
                            '0=Original (no distributional), '
                            '1=Distributional critic only, '
                            '2=+Context quantile selection, '
                            '3=+Context temperature (full), '
                            '4=+Opponent modulation (D1/D2)')

    # Individual distributional feature flags (use if --level not specified)
    parser.add_argument('--distributional', action='store_true', default=False,
                       help='Enable distributional critic (5-quantile value function)')
    parser.add_argument('--context-quantile', action='store_true', default=False,
                       help='Enable context-based quantile selection (requires --distributional)')
    parser.add_argument('--context-temperature', action='store_true', default=False,
                       help='Enable context-based temperature modulation (requires --distributional)')
    parser.add_argument('--n-quantiles', type=int, default=5,
                       help='Number of quantiles for distributional critic (default: 5)')
    parser.add_argument('--quantile-huber-kappa', type=float, default=1.0,
                       help='Huber loss threshold for quantile regression (default: 1.0)')
    parser.add_argument('--temperature-base', type=float, default=1.0,
                       help='Base softmax temperature (default: 1.0)')
    parser.add_argument('--temperature-scale', type=float, default=0.5,
                       help='Context scale for temperature modulation (default: 0.5)')
    parser.add_argument('--opponent-modulation', action='store_true', default=False,
                       help='Enable D1/D2 opponent modulation of policy activations')
    parser.add_argument('--context-decision-only', action='store_true', default=False,
                       help='Apply context input only during the decision period')
    parser.add_argument('--context-distribution', type=str, default=None,
                       choices=['uniform', 'gaussian'],
                       help='Training-time distribution for context c (default: model config)')
    parser.add_argument('--context-uniform-low', type=float, default=None,
                       help='Lower bound for uniform context c sampling')
    parser.add_argument('--context-uniform-high', type=float, default=None,
                       help='Upper bound for uniform context c sampling')
    parser.add_argument('--context-gaussian-mean', type=float, default=None,
                       help='Mean for Gaussian context c sampling')
    parser.add_argument('--context-gaussian-std', type=float, default=None,
                       help='Standard deviation for Gaussian context c sampling')
    parser.add_argument('--no-rpe-modulation', action='store_true', default=False,
                       help='Disable RPE/dopamine modulation for this run')
    parser.add_argument('--rpe-modulation', action='store_true', default=False,
                       help='Enable RPE/dopamine modulation for this run')
    parser.add_argument('--rpe-modulation-gain', type=float, default=None,
                       help='Gain applied to natural RPE before dopamine modulation')
    parser.add_argument('--rpe-modulation-clamp', type=float, default=None,
                       help='Clamp for dopamine/RPE signal before D1/D2 modulation')
    parser.add_argument('--vta-training-context', action='store_true', default=False,
                       help='Train with trial-constant VTA dopamine context added to natural RPE dopamine')
    parser.add_argument('--keep-training-context-input', action='store_true', default=False,
                       help='When using --vta-training-context, also keep the old direct context input sampling')
    parser.add_argument('--vta-context-distribution', type=str, default=None,
                       choices=['uniform', 'gaussian'],
                       help='Training-time VTA context distribution (default: model config)')
    parser.add_argument('--vta-context-low', type=float, default=None,
                       help='Lower bound for VTA context sampling')
    parser.add_argument('--vta-context-high', type=float, default=None,
                       help='Upper bound for VTA context sampling')
    parser.add_argument('--vta-context-mean', type=float, default=None,
                       help='Mean for Gaussian VTA context sampling')
    parser.add_argument('--vta-context-std', type=float, default=None,
                       help='Standard deviation for Gaussian VTA context sampling')
    parser.add_argument('--vta-context-weight', type=float, default=None,
                       help='Multiplier applied to sampled VTA context')
    parser.add_argument('--dopamine-homogeneous-sensitivity', action='store_true', default=False,
                       help='Disable per-neuron dopamine sensitivity heterogeneity')
    parser.add_argument('--dopamine-sensitivity-min', type=float, default=None,
                       help='Minimum per-neuron dopamine sensitivity')
    parser.add_argument('--dopamine-sensitivity-max', type=float, default=None,
                       help='Maximum per-neuron dopamine sensitivity')
    parser.add_argument('--dopamine-sensitivity-learned', action='store_true', default=False,
                       help='Make per-neuron dopamine sensitivities trainable')
    parser.add_argument('--dopamine-bias', action='store_true', default=False,
                       help='Enable learned per-neuron dopamine-dependent current/bias')
    parser.add_argument('--dopamine-bias-max-abs', type=float, default=None,
                       help='Absolute clamp for dopamine bias weights')
    parser.add_argument('--dopamine-modulation-mode', type=str, default=None,
                       choices=['linear', 'hill'],
                       help='Dopamine gain model: linear push-pull or Hill dose-occupancy')
    parser.add_argument('--dopamine-hill-base-da', type=float, default=None,
                       help='Baseline dopamine concentration for Hill modulation')
    parser.add_argument('--dopamine-hill-da-range', type=float, default=None,
                       help='Signed signal-to-concentration range for Hill modulation')
    parser.add_argument('--dopamine-hill-ec50-d1', type=float, default=None,
                       help='D1 EC50 for Hill receptor occupancy')
    parser.add_argument('--dopamine-hill-ec50-d2', type=float, default=None,
                       help='D2 EC50 for Hill receptor occupancy')
    parser.add_argument('--dopamine-hill-coefficient', type=float, default=None,
                       help='Hill coefficient for receptor occupancy')
    parser.add_argument('--dopamine-hill-gain-scale', type=float, default=None,
                       help='Scale from occupancy change to D1/D2 firing-rate gain')
    parser.add_argument('--dopamine-learning-modulation-mode', type=str, default=None,
                       choices=['linear', 'hill'],
                       help='Learning eta model: linear context scaling or Hill dose-occupancy')
    parser.add_argument('--dopamine-learning-eta-min', type=float, default=None,
                       help='Minimum eta_plus/eta_minus after learning modulation')
    parser.add_argument('--dopamine-learning-eta-max', type=float, default=None,
                       help='Maximum eta_plus/eta_minus after learning modulation')
    parser.add_argument('--baseline-activity-balance', type=float, default=None,
                       help='Regularizer strength to spread baseline/value activity across neurons')

    # Fixed/frozen network options
    parser.add_argument('--fix-policy', action='store_true', default=False,
                       help='Freeze policy network weights (train only baseline)')
    parser.add_argument('--fix-baseline', action='store_true', default=False,
                       help='Freeze baseline network weights (train only policy)')
    parser.add_argument('--fix-context-projection', action='store_true', default=False,
                       help='Freeze context projection weights (requires learned projection)')
    parser.add_argument('--context-projection-learned', action='store_true', default=False,
                       help='Use learned context projection instead of simple sum')
    parser.add_argument('--context-projection-lr', type=float, default=0.001,
                       help='Learning rate for context projection (default: 0.001)')

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

    # Process distributional RL level presets
    # If --level is specified, it overrides individual flags
    if args.level is not None:
        if args.level == 0:
            # Level 0: Original (no distributional features)
            distributional = False
            context_quantile = False
            context_temperature = False
            opponent_modulation = False
            print(f"Using Level {args.level}: Original (no distributional features)")
        elif args.level == 1:
            # Level 1: Distributional critic only
            distributional = True
            context_quantile = False
            context_temperature = False
            opponent_modulation = False
            print(f"Using Level {args.level}: Distributional critic only")
        elif args.level == 2:
            # Level 2: Distributional + context quantile selection
            distributional = False
            context_quantile = False
            context_temperature = False
            opponent_modulation = False
            print(f"Using Level {args.level}: Distributional + context quantile selection")
        elif args.level == 3:
            # Level 3: Full features (distributional + context quantile + temperature)
            distributional = True
            context_quantile = True
            context_temperature = True
            opponent_modulation = False
            print(f"Using Level {args.level}: Full distributional features (all enabled)")
        elif args.level == 4:
            # Level 4: Full features + opponent modulation
            distributional = False
            context_quantile = False
            context_temperature = False
            opponent_modulation = True
            print(f"Using Level {args.level}: Full features + opponent modulation")
    else:
        # Use individual flags
        distributional = args.distributional
        context_quantile = args.context_quantile
        context_temperature = args.context_temperature
        opponent_modulation = args.opponent_modulation
    context_decision_only = args.context_decision_only

    # Process kappa distribution parameters
    kappa_dist = args.kappa_dist
    kappa_dist_params = None
    if kappa_dist == 'gaussian':
        kappa_dist_params = {
            'mean': args.kappa_dist_mean,
            'std': args.kappa_dist_std
        }
    elif kappa_dist == 'uniform':
        kappa_dist_params = {
            'low': args.kappa_dist_low,
            'high': args.kappa_dist_high
        }

    print("=" * 80)
    print(f"MODELFILE: {modelfile}")
    print(f"ACTION:    {action}")
    print(f"ARGS:      {action_args}")
    print(f"SEED:      {seed}")
    print(f"SUFFIX:    {suffix}")
    print(f"DEVICE:    {device}")
    if kappa_dist:
        print(f"KAPPA DIST: {kappa_dist}")
        print(f"KAPPA PARAMS: {kappa_dist_params}")
    if args.context_distribution is not None:
        print(f"CONTEXT DIST: {args.context_distribution}")
        if args.context_distribution == 'gaussian':
            mean = args.context_gaussian_mean if args.context_gaussian_mean is not None else 'config'
            std = args.context_gaussian_std if args.context_gaussian_std is not None else 'config'
            print(f"CONTEXT PARAMS: mean={mean}, std={std}")
        else:
            low = args.context_uniform_low if args.context_uniform_low is not None else 'config'
            high = args.context_uniform_high if args.context_uniform_high is not None else 'config'
            print(f"CONTEXT PARAMS: low={low}, high={high}")
    print("=" * 80)

    # Setup paths
    here = utils.get_here(__file__)
    # We're in scripts/training/, go up two levels to get repository root
    repo_root = os.path.dirname(os.path.dirname(here))

    # Name to use
    name = os.path.splitext(os.path.basename(modelfile))[0] + suffix

    # Data directory structure in repository root
    datadir = args.data_root
    if datadir is None:
        datadir = os.path.join(repo_root, 'data')
    elif not os.path.isabs(datadir):
        datadir = os.path.join(repo_root, datadir)
    weightsdir = os.path.join(datadir, 'weights')
    figuresdir = os.path.join(datadir, 'figures')
    trialsdir = os.path.join(datadir, 'trials')

    # Paths for this specific model
    datapath = os.path.join(weightsdir, name)
    figspath = os.path.join(figuresdir, name)
    trialspath = os.path.join(trialsdir, name)

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
        apply_context_sampling_args(model, args)

        recover = 'recover' in action_args
        model.train(savefile, seed, recover=recover, device=device, kappa=args.kappa,
                   kappa_dist=kappa_dist, kappa_dist_params=kappa_dist_params,
                   distributional=distributional,  # Use processed variable (from --level or --distributional)
                   context_quantile=context_quantile,  # Use processed variable
                   context_temperature=context_temperature,  # Use processed variable
                   use_opponent_modulation=opponent_modulation,
                   context_decision_only=context_decision_only,
                   n_quantiles=args.n_quantiles,
                   quantile_huber_kappa=args.quantile_huber_kappa,
                   temperature_base=args.temperature_base,
                   temperature_scale=args.temperature_scale)

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
            # Look in the weights directory (the original pre-trained model)
            pretrained_datapath = os.path.join(weightsdir, base_name)
            pretrained_file = os.path.join(pretrained_datapath, base_name + '.pkl')
            print(f"Auto-detecting pre-trained weights: {pretrained_file}")

        if not os.path.exists(pretrained_file):
            print(f"\nError: Pre-trained file not found: {pretrained_file}")
            print("\nOptions:")
            print("  1. Train a base model first (without --kappa or with --kappa 0)")
            print("  2. Specify custom path with --pretrained /path/to/model.pkl")
            sys.exit(1)

        model = Model(modelfile)
        if context_decision_only:
            model.config['context_decision_only'] = True
        apply_context_sampling_args(model, args)
        model.finetune(pretrained_file, savefile, args.kappa, seed=seed,
                      max_iter=args.finetune_iter, lr=args.finetune_lr,
                      grad_clip=args.grad_clip, baseline_grad_clip=args.baseline_grad_clip,
                      device=device, kappa_dist=kappa_dist, kappa_dist_params=kappa_dist_params)

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
        if context_decision_only:
            model.config['context_decision_only'] = True
        apply_context_sampling_args(model, args)

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
