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

    def finetune(self, pretrained_file, savefile, kappa, seed=1, max_iter=None, lr=None,
                 grad_clip=None, baseline_grad_clip=None, device='cpu'):
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

        # Create a PolicyGradient instance using the saved training hyperparameters
        pg = self.get_pg(finetune_config, seed=finetune_config['seed'], device=device, kappa=kappa)

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

        # Verify weights loaded correctly by checking critical parameters
        print(f"\nWeight loading verification:")
        print(f"  Policy network:")
        for name, param in pg.policy_net.named_parameters():
            print(f"    {name}: shape={param.shape}, mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}, nonzero={torch.count_nonzero(param.data).item()}/{param.numel()}")
        print(f"  Baseline network:")
        for name, param in list(pg.baseline_net.named_parameters())[:3]:
            print(f"    {name}: shape={param.shape}, mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        print(f"  Total policy parameters: {sum(p.numel() for p in pg.policy_net.parameters())}")
        print(f"  Total baseline parameters: {sum(p.numel() for p in pg.baseline_net.parameters())}")

        print(f"\n{'='*80}")
        print(f"FINE-TUNING WITH NEW KAPPA")
        print(f"{'='*80}")
        print(f"Loaded pre-trained weights from: {pretrained_file}")
        print(f"Using hyperparameters from pretrained model:")
        print(f"  Learning rate: {pg.config['lr']}")
        print(f"  Baseline learning rate: {pg.config['baseline_lr']}")
        print(f"  Batch size (n_gradient): {pg.config['n_gradient']}")
        print(f"  Validation trials: {pg.config['n_validation']}")
        print(f"\nNew kappa (κ): {kappa}")
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
            print(f"Fine-tuning iterations: {max_iter} (original: {original_max_iter})")

        # Train with new kappa
        pg.train(savefile, recover=False)