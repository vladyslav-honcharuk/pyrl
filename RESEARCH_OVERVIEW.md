# Research Overview: Risk-Sensitive RL in a Gambling Task

## Goal
This project studies how **risk preference** can emerge in recurrent neural networks trained with policy gradients. The central control parameter is **kappa (κ)**, which modulates how gains vs losses are weighted during learning.

## Task
The agent performs a two-choice gambling task:
- Each trial presents a left and right option with different probability-reward profiles.
- Options are drawn from a **5×5 grid (25 options)** spanning reward probability and magnitude.
- The network must maintain fixation, then choose left or right during the decision phase.

## Core Hypothesis
Changing κ should shift behavior systematically:
- **κ < 0** → risk-averse tendencies
- **κ = 0** → risk-neutral / expected-value style behavior
- **κ > 0** → risk-seeking tendencies

## Modeling Approach
- Architecture: recurrent policy network + recurrent baseline/value network.
- Training: policy-gradient framework with a value baseline.
- Current learning target in code: **Monte Carlo discounted returns** for baseline/policy updates.
- Additional analysis signal: **online TD error** is also computed and saved as objective/subjective RPE traces. <this is not necessary for the diagrams of how the network learns, but is useful for later analysis of how the network's internal representations relate to reward prediction errors.>

## What Is Produced
Training and analysis generate:
- `data/weights/<model>/` — trained model checkpoints
- `data/trials/<model>/` — trial-level behavior/activity data
- `data/figures/<model>/` — behavioral and neural analysis plots

## Typical Workflow
1. Train a model (or sweep κ values).
2. Generate trials (`trials-a` with neural activity, `trials-b` behavior-only).
3. Plot single-model behavior/neural summaries or multi-model κ comparisons.

## Research Relevance
This framework is designed to probe computational mechanisms behind individual differences in risk preference, inspired by the Nakazawa, Isa, & Sasaki (2023) gambling-task setting.
