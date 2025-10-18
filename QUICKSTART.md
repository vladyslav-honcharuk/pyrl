# PyRL-Torch Quick Start Guide

## Installation

1. **Install the package in development mode:**

```bash
cd /path/to/pyrl_torch
python3 -m pip install -e .
```

This will install `pyrl_torch` and its dependencies (numpy, torch, matplotlib, scipy).

## Basic Usage

### 1. View Model Information

```bash
python3 examples/do.py examples/models/rdm_fixed.py info
```

This will display:
- Network architecture details
- Number of inputs and actions
- Time step and max time
- Network sizes

### 2. Train a Model

```bash
python3 examples/do.py examples/models/rdm_fixed.py train --seed 1
```

Options:
- `--seed N` - Set random seed (default: 100)
- `--gpu` - Use GPU/MPS if available (Mac: uses MPS, others: CUDA)
- `--device DEVICE` - Specify exact device (e.g., 'cuda:0', 'mps', 'cpu')
- `--suffix STR` - Add suffix to save file name

### 3. Run Analysis

```bash
python3 examples/do.py examples/models/rdm_fixed.py run
```

## Available Models

All models are in `examples/models/`:

| Model | Description |
|-------|-------------|
| `rdm_fixed.py` | Random dot motion (fixed duration) |
| `rdm_rt.py` | Random dot motion (reaction time) |
| `mante.py` | Context-dependent integration |
| `multisensory.py` | Multisensory integration |
| `romo.py` | Parametric working memory |
| `postdecisionwager.py` | Post-decision wagering |
| `padoaschioppa2006.py` | Economic choice task |
| `cartpole.py` | CartPole RL task |

## Paper Figures

Generate paper figures using:

```bash
cd /path/to/pyrl_torch
PYTHONPATH=. python3 paper/all.py MODEL_NAME
```

Examples:
```bash
PYTHONPATH=. python3 paper/all.py rdm_fixed
PYTHONPATH=. python3 paper/all.py mante
PYTHONPATH=. python3 paper/all.py multisensory
```

## Directory Structure

```
pyrl_torch/
├── pyrl_torch/          # Core library
├── examples/
│   ├── do.py           # Main entry point
│   ├── models/         # Task definitions
│   └── analysis/       # Analysis scripts
└── paper/              # Paper figure generation
```

## Troubleshooting

### Import Errors

If you get import errors, make sure the package is installed:
```bash
python3 -m pip install -e .
```

### Python Version

Requires Python 3.7 or later. Check your version:
```bash
python3 --version
```

### Device Selection

- **CPU**: Use `--device cpu` (default)
- **Mac with Apple Silicon**: Use `--gpu` (uses MPS)
- **NVIDIA GPU**: Use `--gpu` or `--device cuda:0`

## Next Steps

1. **Read the README**: See [README.md](README.md) for detailed documentation
2. **Check structure**: See [STRUCTURE.md](STRUCTURE.md) for migration details
3. **Explore examples**: Browse `examples/models/` for different task implementations
4. **Modify tasks**: Create your own cognitive task by copying and modifying an existing model

## Example: Complete Workflow

```bash
# 1. Install package
python3 -m pip install -e .

# 2. View model info
python3 examples/do.py examples/models/rdm_fixed.py info

# 3. Train the model
python3 examples/do.py examples/models/rdm_fixed.py train --seed 1

# 4. Train with GPU (if available)
python3 examples/do.py examples/models/rdm_fixed.py train --seed 1 --gpu

# 5. Generate paper figures (after training)
PYTHONPATH=. python3 paper/all.py rdm_fixed
```

## Configuration

Default parameters are in `pyrl_torch/configs.py`:
- Learning rate: 0.004
- Network size (N): 100
- Time step (dt): 10ms
- Max updates: 100,000

These can be overridden in individual task files.
