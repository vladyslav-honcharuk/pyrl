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

    def get_pg(self, config_or_savefile, seed=1, dt=None, load='best', device=None, kappa=0.0,
               kappa_dist=None, kappa_dist_params=None):
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
        kappa_dist : str, optional
            Distribution for per-neuron kappa ('gaussian', 'uniform', or None).
        kappa_dist_params : dict, optional
            Parameters for kappa distribution.

        Returns
        -------
        pg : PolicyGradient
            Configured PolicyGradient instance.
        """
        # If config_or_savefile is a dict, check if it has kappa/distribution params
        if isinstance(config_or_savefile, dict):
            if 'kappa' in config_or_savefile:
                kappa = config_or_savefile['kappa']
            if 'kappa_dist' in config_or_savefile:
                kappa_dist = config_or_savefile['kappa_dist']
            if 'kappa_dist_params' in config_or_savefile:
                kappa_dist_params = config_or_savefile['kappa_dist_params']

        return PolicyGradient(self.Task, config_or_savefile, seed=seed,
                            dt=dt, load=load, device=device, kappa=kappa,
                            kappa_dist=kappa_dist, kappa_dist_params=kappa_dist_params)

    def train(self, savefile='savefile.pkl', seed=1, recover=False, device='mps', kappa=None,
              kappa_dist=None, kappa_dist_params=None, distributional=False, context_quantile=False,
              context_temperature=False, use_opponent_modulation=False, context_decision_only=False,
              n_quantiles=5, quantile_huber_kappa=1.0, temperature_base=1.0,
              temperature_scale=0.5):
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
        kappa_dist : str, optional
            Distribution for per-neuron kappa ('gaussian', 'uniform', or None).
        kappa_dist_params : dict, optional
            Parameters for kappa distribution.
        distributional : bool
            Enable distributional critic (5-quantile value function).
        context_quantile : bool
            Enable context-based quantile selection.
        context_temperature : bool
            Enable context-based temperature modulation.
        use_opponent_modulation : bool
            Enable D1/D2 opponent modulation of policy activations.
        context_decision_only : bool
            Apply context input only during the decision period.
        n_quantiles : int
            Number of quantiles for distributional critic.
        quantile_huber_kappa : float
            Huber loss threshold for quantile regression.
        temperature_base : float
            Base softmax temperature.
        temperature_scale : float
            Context scale for temperature modulation.
        """
        # Default kappa to 0.0 if not specified
        if kappa is None:
            kappa = 0.0
        
        # Add distributional flags to config
        if distributional:
            self.config['use_distributional_critic'] = True
            self.config['n_quantiles'] = n_quantiles
            self.config['quantile_huber_kappa'] = quantile_huber_kappa
            if context_quantile:
                self.config['use_context_quantile_selection'] = True
            if context_temperature:
                self.config['use_context_temperature'] = True
                self.config['temperature_base'] = temperature_base
                self.config['temperature_context_scale'] = temperature_scale
        if use_opponent_modulation:
            self.config['use_opponent_modulation'] = True
        if context_decision_only:
            self.config['context_decision_only'] = True

        if recover and os.path.isfile(savefile):
            pg = self.get_pg(savefile, load='current', device=device, kappa=kappa,
                           kappa_dist=kappa_dist, kappa_dist_params=kappa_dist_params)
        else:
            self.config['seed'] = 3 * seed
            self.config['policy_seed'] = 3 * seed + 1
            self.config['baseline_seed'] = 3 * seed + 2
            # Store kappa configuration in config for reference
            self.config['kappa'] = kappa
            self.config['kappa_dist'] = kappa_dist
            self.config['kappa_dist_params'] = kappa_dist_params
            pg = self.get_pg(self.config, self.config['seed'], device=device, kappa=kappa,
                           kappa_dist=kappa_dist, kappa_dist_params=kappa_dist_params)

        # Train
        pg.train(savefile, recover=recover)

    def finetune(self, pretrained_file, savefile, kappa, seed=1, max_iter=None, lr=None,
                 grad_clip=None, baseline_grad_clip=None, device='cpu',
                 kappa_dist=None, kappa_dist_params=None):
        """
        Fine-tune a pre-trained network with a new kappa value.

        This implements the fine-tuning procedure from Nakazawa et al. (2023):
        1. Load weights from a network pre-trained with kappa=0
        2. Use the original training hyperparameters (learning rate, batch size, etc.) from the pretrained model
        3. Update only the kappa parameter (keep all weights)
        4. Continue training with the new kappa value

        Parameters
        ----------
        pretrained_file : str
            Path to pre-trained model (trained with kappa=0).
        savefile : str
            Path to save fine-tuned model.
        kappa : float
            New risk-sensitivity parameter (-1 to +1).
        seed : int
            Random seed for fine-tuning.
        max_iter : int, optional
            Maximum iterations for fine-tuning. If None, uses config default.
        lr : float, optional
            Learning rate for fine-tuning. If None, uses pretrained model's learning rate.
        grad_clip : float, optional
            Gradient clipping threshold for policy network. If None, no clipping.
        baseline_grad_clip : float, optional
            Gradient clipping threshold for baseline network. If None, no clipping.
        device : str, optional
            Device to use ('cpu', 'cuda', or specific cuda device).
        kappa_dist : str, optional
            Distribution for per-neuron kappa ('gaussian', 'uniform', or None).
        kappa_dist_params : dict, optional
            Parameters for kappa distribution.
        """
        # Load the pretrained model's config to get the training hyperparameters
        from . import utils
        saved_data = utils.load(pretrained_file)
        saved_config = saved_data['config']

        # Use saved hyperparameters (learning rates, batch size, etc.)
        # This ensures we fine-tune with the same settings as the original training
        finetune_config = self.config.copy()
        finetune_config['lr'] = saved_config['lr'] if lr is None else lr
        finetune_config['baseline_lr'] = saved_config['baseline_lr'] if lr is None else lr
        finetune_config['n_gradient'] = saved_config['n_gradient']
        finetune_config['n_validation'] = saved_config['n_validation']

        # Apply gradient clipping if specified
        if grad_clip is not None:
            finetune_config['grad_clip'] = grad_clip
        if baseline_grad_clip is not None:
            finetune_config['baseline_grad_clip'] = baseline_grad_clip

        # Set seeds
        finetune_config['seed'] = 3 * seed
        finetune_config['policy_seed'] = 3 * seed + 1
        finetune_config['baseline_seed'] = 3 * seed + 2
        finetune_config['kappa'] = kappa
        finetune_config['kappa_dist'] = kappa_dist
        finetune_config['kappa_dist_params'] = kappa_dist_params

        # Create a PolicyGradient instance using the saved training hyperparameters
        pg = self.get_pg(finetune_config, seed=finetune_config['seed'], device=device, kappa=kappa,
                        kappa_dist=kappa_dist, kappa_dist_params=kappa_dist_params)

        # Load the best weights from the pretrained model
        # The save format uses separate keys for policy and baseline params
        policy_params = saved_data.get('best_policy_params', saved_data.get('current_policy_params'))
        baseline_params = saved_data.get('best_baseline_params', saved_data.get('current_baseline_params'))

        if policy_params is None or baseline_params is None:
            print("Error: Could not find saved parameters in pretrained file")
            print(f"Available keys: {list(saved_data.keys())}")
            sys.exit(1)

        # Convert numpy arrays to PyTorch state dicts and load
        import torch
        policy_state_dict = {k: torch.from_numpy(v).to(device) for k, v in policy_params.items()}
        baseline_state_dict = {k: torch.from_numpy(v).to(device) for k, v in baseline_params.items()}

        pg.policy_net.load_state_dict(policy_state_dict)
        pg.baseline_net.load_state_dict(baseline_state_dict)

        # Override max_iter if specified
        if max_iter is not None:
            original_max_iter = pg.config['max_iter']
            pg.config['max_iter'] = max_iter
            print(f"Fine-tuning iterations: {max_iter} (original: {original_max_iter})")

        # Train with new kappa
        pg.train(savefile, recover=False)
