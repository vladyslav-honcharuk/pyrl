# CLAUDE.md - Project Context for AI Assistants

## Project Overview

This is a **risk-sensitive reinforcement learning** project that trains recurrent neural networks (RNNs) using policy gradients to perform a gambling task. The core focus is on understanding how neural networks develop risk-averse or risk-seeking behaviors controlled by the **kappa parameter**.

## Key Concepts

### Risk-Sensitivity Parameter (Kappa)
- **kappa < 0**: Risk-averse behavior (prefer high-probability, low-reward options)
- **kappa = 0**: Risk-neutral (expected value maximization)
- **kappa > 0**: Risk-seeking behavior (prefer low-probability, high-reward options)

### Gambling Task
- Agent chooses between two targets with different probability-reward profiles
- 25 gambling options arranged in 5×5 matrix by probability (0.1, 0.3, 0.5, 0.7, 0.9) and reward magnitude
- Each option represented by RGB colors for visual discrimination
- Agent must learn to discriminate and choose based on risk preference

## Project Structure

```
pyrl/                          # Core RL library
├── model.py                   # Model wrapper for training
├── policygradient.py          # Policy gradient algorithm implementation
├── gru.py                     # GRU recurrent network
├── simple.py                  # Simple RNN network
├── networks.py                # Network registry
├── configs.py                 # Default configuration parameters
├── tasktools.py               # Task utilities
└── utils.py                   # General utilities

tasks/
└── gambling.py                # Gambling task definition and parameters

scripts/
├── training/
│   ├── train.py                          # Main training script
│   ├── train_kappa_sweep_parallel.py     # Parallel kappa sweep training
│   └── train_kappa_all_parallel.py       # Train all kappa values
└── analysis/
    ├── analyze_kappa_sweep.py            # Analyze kappa sweep results
    └── analyze_single_model.py           # Single model analysis

data/                          # Output directory (gitignored)
├── weights/                   # Trained model weights (.pkl)
├── figures/                   # Generated analysis figures
└── trials/                    # Trial-by-trial data
```

## Important Configuration Parameters

From [pyrl/configs.py](pyrl/configs.py):

- `N`: Number of hidden units (default: 100)
- `lr`: Learning rate (default: 0.0005)
- `max_iter`: Maximum training iterations (default: 3000)
- `kappa`: Risk-sensitivity parameter (default: 0)
- `network_type`: 'gru' or 'simple' (default: 'gru')
- `var_rec`: Recurrent noise variance (default: 0.01)
- `grad_clip`: Gradient clipping threshold (default: None)
- `tau`: Time constant in ms (default: 100)
- `dt`: Time step in ms (default: 10)

## Common Operations

### Training
```bash
# Train with default parameters
python scripts/training/train.py tasks/gambling.py train --seed 1

# Train with specific kappa
python scripts/training/train.py tasks/gambling.py train --kappa 0.5 --suffix _kappa0p5

# Train with GPU
python scripts/training/train.py tasks/gambling.py train --gpu

# Fine-tune existing model with different kappa
python scripts/training/train.py tasks/gambling.py finetune --kappa 0.5 --suffix _kappa0p5
```

### Analysis
```bash
# Get model info
python scripts/training/train.py tasks/gambling.py info

# Analyze single model (note: suffix is used as-is, no underscore added)
python scripts/analysis/analyze_single_model.py tasks/gambling.py --suffix neg0p5 --kappa -0.5

# Analyze kappa sweep (all models)
python scripts/analysis/analyze_kappa_sweep.py tasks/gambling.py [--parallel]

# Generate mega-comparison plot (requires trials for all kappa values)
python scripts/training/train.py tasks/gambling.py run scripts/plotting/gambling.py mega-comparison
```

## Recent Work

Based on recent commits:
1. **Per-neuron kappa distributions**: Support for heterogeneous risk preferences across neurons
2. **Training improvements**: Fixed bug where model wasn't training due to low probability of exploration
3. **Value network**: Switched from TD to Monte Carlo approach
4. **Kappa sweep training**: Parallel training infrastructure for exploring different risk preferences
5. **Enhanced analysis**: Regression scatter analysis and trial-by-trial tracking

## Active Development Areas

Current modified files (uncommitted changes):
- [pyrl/configs.py](pyrl/configs.py) - Configuration updates
- [pyrl/model.py](pyrl/model.py) - Model architecture changes
- [pyrl/policygradient.py](pyrl/policygradient.py) - Policy gradient algorithm updates
- [pyrl/tasktools.py](pyrl/tasktools.py) - Task utility improvements
- [pyrl/utils.py](pyrl/utils.py) - General utility updates
- [scripts/plotting/gambling.py](scripts/plotting/gambling.py) - Visualization updates
- [tasks/gambling.py](tasks/gambling.py) - Task definition modifications

## Key Files to Understand

1. **[tasks/gambling.py](tasks/gambling.py)**: Defines the gambling task, including the 25 gambling options with their probability-reward profiles and RGB color encodings

2. **[pyrl/policygradient.py](pyrl/policygradient.py)**: Core RL algorithm implementing risk-sensitive policy gradients with kappa parameter

3. **[pyrl/gru.py](pyrl/gru.py)**: GRU network implementation used as the agent's "brain"

4. **[pyrl/configs.py](pyrl/configs.py)**: All default hyperparameters and configuration options

5. **[pyrl/model.py](pyrl/model.py)**: High-level wrapper that ties everything together for training

## Technical Notes

### Policy Gradient Training
- Uses episodic mode with Monte Carlo returns
- Supports baseline subtraction for variance reduction
- Implements risk-sensitive updates via kappa-transformed rewards
- Gradient clipping available for training stability

### Network Architecture
- Default: GRU with 100 hidden units
- Alternative: Simple RNN available via `network_type='simple'`
- Recurrent noise injection for exploration (`var_rec`)
- L1/L2 regularization on recurrent weights

### Task Design
- Fixation period: 250ms (25 timesteps)
- Stimulus presentation: 250ms (25 timesteps)
- Decision period: 260ms (26 timesteps)
- RGB color encoding for each gambling option
- Input noise (sigma = 0.1) for robustness

## Development Tips

- Models are saved in `data/weights/{model_name}/{model_name}.pkl`
- Use `--suffix` flag to differentiate model variants
- Checkpoint frequency controlled by `checkfreq` parameter (default: 50 iterations)
- Training stops when `target_reward` is reached or `max_iter` iterations complete
- GPU support available via `--gpu` flag (uses PyTorch CUDA)

## Dependencies

- Python 3.7+
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- SciPy >= 1.5.0

## Git Workflow

- Main branch: `master`
- Current branch: `sharp-panini`
- Naming convention: Model names often include kappa values (e.g., `model_kappa0p5`)
