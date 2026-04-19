"""
Gambling/Risk-preference task, based on

  Computational mechanisms of risk preference generated in recurrent neural networks
  Nakazawa, Isa, & Sasaki, 2023 (poster)

This task presents two targets with different probability-reward profiles,
testing risk-seeking vs risk-averse behavior.
"""

import numpy as np
from pyrl import tasktools

# Inputs: Fixation + 3 RGB for left + 3 RGB for right
inputs = tasktools.to_map('FIXATION', 'LEFT_R', 'LEFT_G', 'LEFT_B',
                          'RIGHT_R', 'RIGHT_G', 'RIGHT_B')

# Actions
actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT', 'CHOOSE-RIGHT')

# 25 gambling options: [probability, reward_size]
# Arranged in 5x5 matrix by probability (rows) and reward magnitude (columns)
value_vector = np.array([
    [0.1, 10],   [0.1, 13.75], [0.1, 17.5],  [0.1, 21.25], [0.1, 25],
    [0.3, 3.33], [0.3, 4.58],  [0.3, 5.83],  [0.3, 7.08],  [0.3, 8.33],
    [0.5, 2],    [0.5, 2.75],  [0.5, 3.5],   [0.5, 4.25],  [0.5, 5.0],
    [0.7, 1.43], [0.7, 1.96],  [0.7, 2.5],   [0.7, 3.04],  [0.7, 3.57],
    [0.9, 1.11], [0.9, 1.53],  [0.9, 1.94],  [0.9, 2.36],  [0.9, 2.78]
])

# RGB colors for each of 25 targets (for visual discrimination)
color_vector = np.array([
    [1, 0, 1],      [0.75, 0, 1],      [0.5, 0.25, 1],   [0.25, 0.25, 1],  [0, 0.5, 1],
    [1, 0, 0.75],   [0.75, 0.25, 0.75],[0.5, 0.25, 0.75],[0.25, 0.5, 0.75],[0, 0.5, 0.75],
    [1, 0.25, 0.5], [0.75, 0.25, 0.5], [0.5, 0.5, 0.5],  [0.25, 0.5, 0.5], [0, 0.75, 0.5],
    [1, 0.25, 0.25],[0.75, 0.5, 0.25], [0.5, 0.5, 0.25], [0.25, 0.75, 0.25],[0, 0.75, 0.25],
    [1, 0.5, 0],    [0.75, 0.5, 0],    [0.5, 0.75, 0],   [0.25, 0.75, 0],  [0, 1, 0]
])

# Training
n_conditions = 25 * 25  # All possible pairs of left-right choices
n_gradient   = 64  # Batch size for gradient updates
n_validation = 500      # Increase to get more stable accuracy estimates

# Input noise
sigma = 0.1

# Durations (in ms)
fixation = 250   # 25 timesteps * 10ms
stimulus = 250   # 25 timesteps * 10ms
decision = 260   # 26 timesteps * 10ms
tmax = fixation + stimulus + decision

# Rewards
R_ABORTED = -0.5   # Penalty for breaking fixation (reduced to allow learning)
R_PENALTY = -0.5 # Penalty for not making a choice (reduced to allow learning)

# Initial action bias: start with a stronger preference to maintain fixation
# until the network has observed the trial state, instead of choosing randomly
# at t=0 and immediately aborting the trial.
bout = np.array([3.0, 0.0, 0.0])

# Reward scaling (from JAX code: size/2.5)
REWARD_SCALE = 2.5

def get_condition(rng, dt, context={}):
    """
    Generate a trial condition by randomly selecting two targets.
    """
    # Epochs
    durations = {
        'fixation': (0, fixation),
        'stimulus': (fixation, fixation + stimulus),
        'decision': (fixation + stimulus, fixation + stimulus + decision),
        'tmax': tmax
    }
    time, epochs = tasktools.get_epochs_idx(dt, durations)

    # Randomly select left and right targets from 25 options
    target_l = context.get('target_l')
    if target_l is None:
        target_l = rng.choice(25)

    target_r = context.get('target_r')
    if target_r is None:
        target_r = rng.choice(25)

    # Get probabilities and reward sizes
    prob_l = value_vector[target_l, 0]
    size_l = value_vector[target_l, 1] / REWARD_SCALE

    prob_r = value_vector[target_r, 0]
    size_r = value_vector[target_r, 1] / REWARD_SCALE

    # Get colors for visual representation
    color_l = color_vector[target_l]
    color_r = color_vector[target_r]

    return {
        'durations': durations,
        'time': time,
        'epochs': epochs,
        'target_l': target_l,
        'target_r': target_r,
        'prob_l': prob_l,
        'size_l': size_l,
        'prob_r': prob_r,
        'size_r': size_r,
        'color_l': color_l,
        'color_r': color_r
    }


def get_step(rng, dt, trial, t, a):
    """
    Execute one timestep of the task.
    """
    epochs = trial['epochs']
    status = {'continue': True}
    reward = 0

    # Initialize input
    u = np.zeros(len(inputs))

    # Fixation period (0-24 timesteps)
    if t in epochs['fixation']:
        u[inputs['FIXATION']] = 1
        # Penalize if not fixating - end trial immediately
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED

    # Stimulus period (25-49 timesteps)
    elif t in epochs['stimulus']:
        u[inputs['FIXATION']] = 1
        # Show colored targets with noise
        u[inputs['LEFT_R']] = trial['color_l'][0] + rng.normal(scale=sigma)
        u[inputs['LEFT_G']] = trial['color_l'][1] + rng.normal(scale=sigma)
        u[inputs['LEFT_B']] = trial['color_l'][2] + rng.normal(scale=sigma)

        u[inputs['RIGHT_R']] = trial['color_r'][0] + rng.normal(scale=sigma)
        u[inputs['RIGHT_G']] = trial['color_r'][1] + rng.normal(scale=sigma)
        u[inputs['RIGHT_B']] = trial['color_r'][2] + rng.normal(scale=sigma)

        # Penalize if not fixating - but end trial immediately
        if a != actions['FIXATE']:
            status['continue'] = False
            reward = R_ABORTED

    # Decision period (50-75 timesteps)
    elif t in epochs['decision']:
        u[inputs['FIXATION']] = 0  # Fixation off
        # Continue showing targets with noise
        u[inputs['LEFT_R']] = trial['color_l'][0] + rng.normal(scale=sigma)
        u[inputs['LEFT_G']] = trial['color_l'][1] + rng.normal(scale=sigma)
        u[inputs['LEFT_B']] = trial['color_l'][2] + rng.normal(scale=sigma)

        u[inputs['RIGHT_R']] = trial['color_r'][0] + rng.normal(scale=sigma)
        u[inputs['RIGHT_G']] = trial['color_r'][1] + rng.normal(scale=sigma)
        u[inputs['RIGHT_B']] = trial['color_r'][2] + rng.normal(scale=sigma)

        # Calculate expected values for correctness labeling
        ev_l = trial['prob_l'] * trial['size_l']
        ev_r = trial['prob_r'] * trial['size_r']

        # Check for choice
        if a == actions['CHOOSE-LEFT']:
            status['continue'] = False
            status['choice'] = 'L'
            status['t_choice'] = t
            # Correct if left has higher EV (symmetric tie-breaking)
            if abs(ev_l - ev_r) > 0.05:
                status['correct'] = ev_l > ev_r
            else:
                status['correct'] = True  # Don't penalize equal-EV choices
            # Probabilistic reward based on left target
            if rng.rand() < trial['prob_l']:
                reward = trial['size_l']
            else:
                reward = 0
        elif a == actions['CHOOSE-RIGHT']:
            status['continue'] = False
            status['choice'] = 'R'
            status['t_choice'] = t
            # Correct if right has higher EV (symmetric tie-breaking)
            if abs(ev_l - ev_r) > 0.05:
                status['correct'] = ev_r > ev_l
            else:
                status['correct'] = True  # Don't penalize equal-EV choices
            # Probabilistic reward based on right target
            if rng.rand() < trial['prob_r']:
                reward = trial['size_r']
            else:
                reward = 0
        elif a == actions['FIXATE']:
            # Still fixating during decision period - small penalty to encourage choice
            # Small enough to allow learning, but encourages exploration
            reward = 0  # Small incremental penalty (accumulates to ~-0.5 over 26 steps)

        # Force choice at end of decision period
        if t == epochs['decision'][-1]:
            status['continue'] = False
            if a == actions['FIXATE']:
                reward = R_PENALTY
                status['correct'] = False

    return u, reward, status


def terminate(perf):
    pass
    # """
    # Termination criterion: achieve good performance on gambling task.

    # The task is learned when the agent:
    # 1. Consistently makes decisions (doesn't fixate through decision period)
    # 2. Chooses higher expected value options at reasonable rate
    # """
    # p_decision, p_correct = tasktools.correct_2AFC(perf)

    # # Reasonable criterion: 95% decision rate + 75% accuracy
    # # 75% accuracy means network prefers higher EV options most of the time
    # return p_decision >= 0.95 and p_correct >= 0.95
