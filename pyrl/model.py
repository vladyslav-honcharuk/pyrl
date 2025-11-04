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

    def get_pg(self, config_or_savefile, seed=1, dt=None, load='best', device=None, kappa=0.0):
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
        kappa : float
            Risk-sensitivity parameter (-1 to +1, default 0.0).

        Returns
        -------
        pg : PolicyGradient
            Configured PolicyGradient instance.
        """
        # If config_or_savefile is a dict, check if it has kappa
        if isinstance(config_or_savefile, dict) and 'kappa' in config_or_savefile:
            kappa = config_or_savefile['kappa']
        
        return PolicyGradient(self.Task, config_or_savefile, seed=seed,
                            dt=dt, load=load, device=device, kappa=kappa)

    def train(self, savefile='savefile.pkl', seed=1, recover=False, device='mps', kappa=None):
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
        kappa : float, optional
            Risk-sensitivity parameter (-1 to +1). If None, uses default 0.0.
        """
        # Default kappa to 0.0 if not specified
        if kappa is None:
            kappa = 0.0
        
        if recover and os.path.isfile(savefile):
            pg = self.get_pg(savefile, load='current', device=device, kappa=kappa)
        else:
            self.config['seed'] = 3 * seed
            self.config['policy_seed'] = 3 * seed + 1
            self.config['baseline_seed'] = 3 * seed + 2
            # Store kappa in config for reference
            self.config['kappa'] = kappa
            pg = self.get_pg(self.config, self.config['seed'], device=device, kappa=kappa)

        # Train
        pg.train(savefile, recover=recover)

    def retrain(self, pretrained_file, savefile, kappa, max_iter=None, device='cpu'):
        """
        Retrain a pre-trained network with a new kappa value.

        This implements the retraining procedure from Nakazawa et al. (2023):
        1. Load a network pre-trained with kappa=0
        2. Update only the kappa parameter (keep all weights)
        3. Continue training with the new kappa value

        Parameters
        ----------
        pretrained_file : str
            Path to pre-trained model (trained with kappa=0).
        savefile : str
            Path to save retrained model.
        kappa : float
            New risk-sensitivity parameter (-1 to +1).
        max_iter : int, optional
            Maximum iterations for retraining. If None, uses config default.
        device : str, optional
            Device to use ('cpu', 'cuda', or specific cuda device).
        """
        # Load pre-trained model
        pg = self.get_pg(pretrained_file, load='best', device=device, kappa=kappa)

        # Update kappa (this changes eta_plus and eta_minus)
        pg.kappa = kappa
        pg.eta_plus = 1.0 + kappa
        pg.eta_minus = 1.0 - kappa

        print(f"\n{'='*80}")
        print(f"RETRAINING WITH NEW KAPPA")
        print(f"{'='*80}")
        print(f"Loaded pre-trained model from: {pretrained_file}")
        print(f"New kappa (κ): {kappa}")
        print(f"  η⁺ (positive RPE scaling): {pg.eta_plus}")
        print(f"  η⁻ (negative RPE scaling): {pg.eta_minus}")
        if kappa > 0:
            print(f"  Bias: RISK-SEEKING (optimistic)")
        elif kappa < 0:
            print(f"  Bias: RISK-AVERSE (pessimistic)")
        else:
            print(f"  Bias: BALANCED (neutral)")
        print(f"{'='*80}\n")

        # Override max_iter if specified
        if max_iter is not None:
            original_max_iter = pg.config['max_iter']
            pg.config['max_iter'] = max_iter
            print(f"Retraining iterations: {max_iter} (original: {original_max_iter})")

        # Train with new kappa
        pg.train(savefile, recover=False)