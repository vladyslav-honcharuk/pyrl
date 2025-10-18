"""
Wrapper class for PolicyGradient to simplify model creation and training.
"""
import importlib.util
import os
import sys

from . import configs
from .performance import Performance2AFC
from .policygradient import PolicyGradient


class Struct:
    """Treat a dictionary like a module."""
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Model:
    """
    Model wrapper for cognitive task training.

    This class loads task specifications and configures the PolicyGradient
    algorithm for training recurrent neural networks.
    """

    def __init__(self, modelfile=None, **kwargs):
        """
        Initialize model.

        Parameters
        ----------
        modelfile : str, optional
            Path to model specification file.
        **kwargs
            Alternative to modelfile - directly specify model parameters.
        """
        # Load model specification
        if modelfile is not None:
            try:
                spec = importlib.util.spec_from_file_location("model", modelfile)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.spec = module
            except IOError:
                print(f"Couldn't load model file {modelfile}")
                sys.exit(1)
        else:
            self.spec = Struct(**kwargs)

        # Task definition
        if hasattr(self.spec, 'Task'):
            self.Task = self.spec.Task
            self.task = self.spec.Task()
        else:
            class Task:
                def __init__(_self):
                    if hasattr(self.spec, 'get_condition'):
                        setattr(_self, 'get_condition', self.spec.get_condition)
                    if hasattr(self.spec, 'get_step'):
                        setattr(_self, 'get_step', self.spec.get_step)
                    if hasattr(self.spec, 'terminate'):
                        setattr(_self, 'terminate', self.spec.terminate)

            self.Task = Task
            self.task = Task()

        # Build configuration
        self.config = {}

        # Check required fields
        for k in configs.required:
            if not hasattr(self.spec, k):
                print(f"[ Model ] Error: {k} is required.")
                sys.exit()
            self.config[k] = getattr(self.spec, k)

        # Fill in defaults
        for k in configs.default:
            self.config[k] = getattr(self.spec, k, configs.default[k])

        # Input/output dimensions
        self.config['Nin'] = len(self.config['inputs'])

        if 'Nout' not in self.config:
            self.config['Nout'] = len(self.config['actions'])

        # Ensure integer types
        self.config['n_gradient'] = int(self.config['n_gradient'])
        self.config['n_validation'] = int(self.config['n_validation'])

        # Performance measure
        if self.config['Performance'] is None:
            self.config['Performance'] = Performance2AFC

        # For trial-by-trial learning
        if self.config['n_gradient'] == 1:
            self.config['checkfreq'] = 1

    def get_pg(self, config_or_savefile, seed=1, dt=None, load='best', device=None):
        """
        Get PolicyGradient instance.

        Parameters
        ----------
        config_or_savefile : dict or str
            Configuration dictionary or path to saved model.
        seed : int
            Random seed.
        dt : float, optional
            Time step (ms). If None, uses config default.
        load : str
            Which parameters to load ('best' or 'current').
        device : str, optional
            Device to use ('cpu', 'cuda', or specific cuda device).

        Returns
        -------
        pg : PolicyGradient
            Configured PolicyGradient instance.
        """
        return PolicyGradient(self.Task, config_or_savefile, seed=seed,
                            dt=dt, load=load, device=device)

    def train(self, savefile='savefile.pkl', seed=1, recover=False, device='mps'):
        """
        Train the network.

        Parameters
        ----------
        savefile : str
            Path to save trained model.
        seed : int
            Random seed.
        recover : bool
            Whether to recover from existing savefile.
        device : str, optional
            Device to use ('cpu', 'cuda', or specific cuda device).
        """
        if recover and os.path.isfile(savefile):
            pg = self.get_pg(savefile, load='current', device=device)
        else:
            self.config['seed'] = 3 * seed
            self.config['policy_seed'] = 3 * seed + 1
            self.config['baseline_seed'] = 3 * seed + 2
            pg = self.get_pg(self.config, self.config['seed'], device=device)

        # Train
        pg.train(savefile, recover=recover)
