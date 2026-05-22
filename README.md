# Gambling Task - Risk-Sensitive Reinforcement Learning

This repository contains a recurrent neural network trained with policy gradients to perform a gambling task with risk-sensitive behavior controlled by the kappa parameter.

## Project Structure

```
.
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── .gitignore                   # Git ignore rules
│
├── pyrl/                        # Core RL library
│   ├── __init__.py
│   ├── model.py                 # Model wrapper
│   ├── actor_critic.py          # Actor-critic trainer
│   ├── gru.py                   # GRU network
│   ├── simple.py                # Simple RNN network
│   ├── networks.py              # Network registry
│   ├── networks_base.py         # Base network class
│   ├── tasktools.py             # Task utilities
│   ├── utils.py                 # General utilities
│   ├── configs.py               # Default configurations
│   ├── performance.py           # Performance tracking
│   ├── nptools.py               # NumPy utilities
│   └── matrixtools.py           # Matrix operations
│
├── tasks/                       # Task definitions
│   ├── __init__.py
│   └── gambling.py              # Gambling task model
│
├── scripts/                     # Executable scripts
│   ├── training/
│   │   ├── train.py                 # Main training script
│   │   ├── train_kappa_sweep_parallel.py   # Parallel kappa sweep
│   │   ├── train_kappa_all_parallel.py     # Train all kappas
│   │   └── example_perneuron_kappa_workflow.sh  # Example workflow
│   └── analysis/
│       ├── analyze_kappa_sweep.py          # Analyze kappa sweep
│       └── analyze_single_model.py         # Analyze single model
│
└── data/                        # Data directory (gitignored)
    ├── weights/                 # Trained model weights
    │   └── {model_name}/
    │       └── {model_name}.pkl
    ├── figures/                 # Generated figures
    │   └── {model_name}/
    └── trials/                  # Trial-by-trial data
        └── {model_name}/
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd gambling-task

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Usage

### Training a Model

```bash
# Train the gambling task with default parameters
python scripts/training/train.py tasks/gambling.py train --seed 1

# Train with GPU acceleration
python scripts/training/train.py tasks/gambling.py train --gpu

# Train with specific kappa (risk-sensitivity parameter)
python scripts/training/train.py tasks/gambling.py train --kappa 0.5 --suffix _kappa0p5
```

### Fine-tuning with Different Kappa

```bash
# Fine-tune a pre-trained model with new kappa value
python scripts/training/train.py tasks/gambling.py finetune --kappa 0.5 --suffix _kappa0p5
```

### Getting Model Info

```bash
# Display model architecture and configuration
python scripts/training/train.py tasks/gambling.py info
```

## Gambling Task

The gambling task presents agents with two targets, each with different probability-reward profiles. The agent must choose between:
- High-probability, low-reward options (risk-averse)
- Low-probability, high-reward options (risk-seeking)

The task uses 25 gambling options arranged in a 5×5 matrix by probability (0.1, 0.3, 0.5, 0.7, 0.9) and reward magnitude.

## Risk-Sensitivity Parameter (Kappa)

- **kappa < 0**: Risk-averse behavior
- **kappa = 0**: Risk-neutral (expected value maximization)
- **kappa > 0**: Risk-seeking behavior

## Output Directories

All outputs are saved in the `data/` directory:
- `data/weights/{model_name}/` - Trained model weights (.pkl files)
- `data/figures/{model_name}/` - Generated analysis figures
- `data/trials/{model_name}/` - Trial-by-trial data for detailed analysis

## Requirements

- Python 3.7+
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0 (for analysis)
- SciPy >= 1.5.0

See `requirements.txt` for complete list.

## Citation

Based on the work:
> Computational mechanisms of risk preference generated in recurrent neural networks
> Nakazawa, Isa, & Sasaki, 2023 (poster)
