"""Actor-critic trainer for recurrent cognitive-task agents."""

from .learning import LearningMixin
from .rollouts import RolloutMixin
from .training_setup import SetupMixin, resolve_device


class ActorCriticTrainer(SetupMixin, RolloutMixin, LearningMixin):
    """Train and evaluate recurrent neural networks with actor-critic updates."""

    def __init__(self, Task, config_or_savefile, seed, dt=None, load='best', device=None,
                 kappa=0.0, kappa_dist=None, kappa_dist_params=None):
        self.task = Task()
        self.device = resolve_device(device)
        self.save = None

        if isinstance(config_or_savefile, str):
            self._load_from_file(config_or_savefile, dt, load)
        else:
            self._create_new_model(config_or_savefile, dt, seed)

        self.policy_net.to(self.device)
        self.baseline_net.to(self.device)

        self._setup_training()
        self._setup_opto_stimulation()

        self.kappa_dist = kappa_dist
        self.kappa_dist_params = kappa_dist_params
        self._setup_kappa(kappa, kappa_dist, kappa_dist_params, seed)

    def _setup_opto_stimulation(self):
        self.opto_stim_offset = self.config.get('opto_stim_offset', 0.0)
        self.opto_stim_gain = self.config.get('opto_stim_gain', 1.0)
        self.opto_stim_phase = self.config.get('opto_stim_phase', 'all')

    def run_trials(self, *args, **kwargs):
        return self._run_trials(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def diagnose_critic(self, *args, **kwargs):
        return self._diagnose_critic(*args, **kwargs)
