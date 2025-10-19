"""
OpenAI Gym: CartPole-v0

  https://gym.openai.com/envs/CartPole-v0

Method: Policy network (ReLU-GRUs) + baseline network (ReLU-GRUs)
        + REINFORCE + Adam + gradient clipping.

Sorry this is a little opaque/overkill but we wanted to test some policy gradient RNN
code (actually intended for neuroscience tasks) on a classic RL task. To reproduce the
results, please download

  https://github.com/frsong/pyrl

and add it to your PYTHONPATH, then run the code below. Requires Theano 0.8.2.

"""
import numpy as np

import gym

from pyrl import tasktools

# Environment
env = gym.make('CartPole-v0')

# Inputs
inputs = tasktools.to_map(range(env.observation_space.high.size))

# Actions
actions = tasktools.to_map(range(env.action_space.n))

# Baseline bias
baseline_bout = 0

# Don't treat the last time point as special
abort_on_last_t = False

# Training
n_gradient   = 1
n_validation = 0

# Network structure
N  = 50
p0 = 0.1
baseline_N = 50

# Time
dt   = 1
tau  = 1
tmax = 200

# Recurrent noise
var_rec          = 0.002
baseline_var_rec = 0.002

class Performance(object):
    def __init__(self):
        self.rewards = []

    def update(self, trial, status):
        self.rewards.append(status['reward'])

    @property
    def n(self):
        return len(self.rewards)

    @property
    def mean(self):
        return np.mean(self.rewards[-100:])

    @property
    def sd(self):
        return np.std(self.rewards[-100:], ddof=1)

    def display(self, output=True):
        s = ''
        if self.rewards[-1] >= tmax:
            s = ' ***'
        print("Episode {}: {} ({:.2f} +- {:.2f}){}"
              .format(len(self.rewards), self.rewards[-1], self.mean, self.sd, s))

class Task(object):
    def start_trial(self):
        self.new_trial = True
        self.R         = 0

    def get_condition(self, rng, dt, context={}):
        return {}

    def get_step(self, rng, dt, trial, t, a):
        if self.new_trial:
            self.new_trial = False

            obs    = env.reset()
            reward = 0
            done   = False
        else:
            obs, reward, done, _ = env.step(a)
        self.R += reward

        status = {'continue': not done, 'reward': self.R}

        return obs, reward/tmax, status

    def terminate(self, perf):
        if perf.n >= 100 and perf.mean >= 195:
            return True
        return False

#/////////////////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    from pyrl.model import Model

    model = Model(inputs=inputs, actions=actions,
                  baseline_bout=baseline_bout,
                  abort_on_last_t=abort_on_last_t,
                  n_gradient=n_gradient, n_validation=n_validation,
                  N=N, p0=p0,
                  dt=dt, tau=tau, tmax=tmax,
                  var_rec=var_rec, baseline_var_rec=baseline_var_rec,
                  Performance=Performance, Task=Task)

    env.monitor.start('openai/cartpole', force=True)
    model.train()
    env.monitor.close()
    #gym.upload('openai/cartpole', api_key='sk_BbLks5hQIOpDviRfJCPFA')
