# PyRL-Torch

PyTorch implementation of the PyRL framework for reward-based training of recurrent neural networks on cognitive and value-based tasks.

This is a modernized version of the original [pyrl](https://github.com/frsong/pyrl) framework, migrated from Python 2.7/Theano to Python 3/PyTorch.

## Directory Structure

```
pyrl_torch/
├── pyrl_torch/          # Core library (equivalent to pyrl/pyrl/)
│   ├── model.py         # Model class wrapping PolicyGradient
│   ├── policygradient.py # Policy gradient algorithm
│   ├── networks.py      # Network architectures (GRU, Linear)
│   ├── gru.py          # GRU implementation
│   ├── linear.py       # Linear network
│   ├── recurrent.py    # Base recurrent class
│   ├── tasktools.py    # Task definition utilities
│   ├── torchtools.py   # PyTorch utility functions
│   └── ...
├── examples/            # Examples (equivalent to pyrl/examples/)
│   ├── do.py           # Main entry point for training/running
│   ├── models/         # Task model definitions
│   │   ├── rdm_fixed.py
│   │   ├── mante.py
│   │   └── ...
│   └── analysis/       # Analysis scripts
│       ├── rdm.py
│       └── ...
└── paper/              # Scripts for generating paper figures
    ├── fig1_rdm.py
    └── ...
```

## Key Differences from Original PyRL

### Framework Changes
- **Python 3** instead of Python 2.7
- **PyTorch** instead of Theano
- **torchtools.py** replaces theanotools.py
- Modern Python syntax (print functions, division, etc.)

### Architecture
The PyTorch implementation maintains the same high-level architecture:
- `Model` class wraps `PolicyGradient` for task-specific training
- `PolicyGradient` implements the core RL algorithm
- Network classes (GRU, Linear, Recurrent) define architectures
- Task modules define trial structure, inputs, actions, and rewards

## Usage

### Training a Model

```bash
cd examples
python do.py models/rdm_fixed.py train --seed 1
```

### Model Information

```bash
python do.py models/rdm_fixed.py info
```

### Running with GPU (MPS on Mac)

```bash
python do.py models/rdm_fixed.py train --gpu
```

## Configuration

Default parameters are in `pyrl_torch/configs.py`:
- Learning rate: 0.004
- Network size (N): 100
- Time step (dt): 10ms
- Baseline network size: 100

## Task Structure

Tasks are defined as Python modules with:
- `inputs` - Input channels (e.g., FIXATION, LEFT, RIGHT)
- `actions` - Available actions (e.g., FIXATE, CHOOSE-LEFT)
- `get_condition()` - Defines trial conditions
- `get_step()` - Implements task dynamics per timestep
- `terminate()` - Defines training termination criteria

## Development Notes

- Default timestep is 10ms (can result in hundreds of steps per trial)
- Use larger `dt` values during development to reduce training time
- Model files in `examples/models/` provide templates for different tasks
- The framework uses policy gradient with baseline for training

## References

Original paper:
- Song, H. F., Yang, G. R., & Wang, X. J. (2016). Training excitatory-inhibitory recurrent neural networks for cognitive tasks: a simple and flexible framework. *PLoS computational biology*, 12(2), e1004792.

Original repository:
- https://github.com/frsong/pyrl
