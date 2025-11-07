# Gradual Kappa Training Guide

This guide explains how to train models with gradual kappa adjustment and how to address gradient collapse issues with negative kappa values.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Gradient Collapse Problem](#understanding-the-gradient-collapse-problem)
3. [Solution: Entropy Regularization](#solution-entropy-regularization)
4. [Gradual Kappa Training Script](#gradual-kappa-training-script)
5. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Basic Usage (Default Settings)

Train with gradual kappa adjustment from 0 to ±1.0 with checkpoints at 0.2, 0.4, 0.6, 0.8, 1.0:

```bash
cd examples
python3 train_kappa_gradual.py models/gambling.py
```

This will:
- Train base model with κ=0
- Gradually increase kappa in steps of 0.05 (default)
- Save checkpoints only at κ = 0.2, 0.4, 0.6, 0.8, 1.0 (both positive and negative)
- Run positive and negative chains in parallel for faster training

### Fix Gradient Collapse with Entropy Regularization

If you're experiencing gradient collapse (especially with negative kappa), add entropy regularization to your model config:

```python
# In your model file (e.g., models/gambling.py)
config = {
    # ... other config parameters ...
    'entropy_coef': 0.01,  # Add this line to prevent policy collapse
}
```

---

## Understanding the Gradient Collapse Problem

### Symptoms

When training with **negative kappa** (risk-seeking), you may observe:

```
POLICY UPDATE DEBUG - Step 600, κ=-0.200
Log-probabilities:
  Mean: -0.000000
  Std: 0.000000
  Min: -0.000000
  Max: 0.000000

Gradient Statistics (policy - before clipping):
  Total norm: 0.000000
  Max absolute: 0.000000
```

**What this means:**
- Log-probabilities are all ~0, which means the policy outputs actions with ~100% confidence
- The policy has collapsed to a deterministic policy (no exploration)
- Gradient = 0, so the policy network stops learning

### Why This Happens

1. **Early in training**: The policy might randomly find a seemingly good action
2. **Risk-seeking behavior** (negative κ): Amplifies positive rewards, reinforcing the action
3. **Policy becomes deterministic**: Network outputs one action with 100% probability
4. **Gradient vanishes**: When log(probability) = 0, there's no gradient signal
5. **Learning stops**: Policy is stuck and cannot improve

This is particularly problematic with **negative kappa** because:
- Risk-seeking amplifies early successes
- The policy prematurely commits to suboptimal actions
- No gradient signal to escape local minima

### Positive Kappa Works Better

With **positive kappa** (risk-averse):
- The policy is more cautious
- Maintains exploration longer
- Gradients remain healthy (as shown in your logs)

---

## Solution: Entropy Regularization

### What is Entropy Regularization?

Entropy measures the "randomness" of the policy:
- **High entropy**: Policy explores many actions (more random)
- **Low entropy**: Policy is deterministic (always same action)

**Entropy formula**: `H = -sum(p * log(p))`

For a 2-action policy:
- Maximum entropy: `log(2) ≈ 0.693` (50-50 split)
- Minimum entropy: `0.000` (100% one action)

### How It Helps

Adding an **entropy bonus** to the policy loss prevents collapse:

```
loss = -J + regularization - entropy_coef * entropy
```

- Encourages the policy to maintain exploration
- Prevents premature convergence to deterministic policies
- Keeps gradients flowing even when policy becomes confident

### Recommended Values

| `entropy_coef` | Effect |
|----------------|--------|
| `0` | No entropy regularization (default, may cause collapse with negative κ) |
| `0.001` | Mild entropy bonus (subtle encouragement to explore) |
| `0.01` | Moderate entropy bonus (**recommended for negative κ**) |
| `0.1` | Strong entropy bonus (may slow convergence) |

### How to Enable

Add to your model config file:

```python
# models/gambling.py (or your model file)
config = {
    # ... existing parameters ...
    'entropy_coef': 0.01,  # Recommended for negative kappa training
}
```

Or pass it as a command-line argument when using finetuning:

```bash
python3 do.py models/gambling.py finetune \
    --kappa -0.2 \
    --entropy-coef 0.01 \
    --pretrained work/data/gambling/gambling.pkl
```

### Monitoring Entropy

The updated code now displays entropy in the debug output:

```
Policy Entropy:
  Mean entropy: 0.452134
  Max possible entropy (log(2)): 0.693147

  ⚠️⚠️⚠️  POLICY COLLAPSE DETECTED! Entropy = 0.003421 < 0.01
  Policy is outputting near-deterministic actions (no exploration)
  This will cause gradient collapse (zero gradients)
```

---

## Gradual Kappa Training Script

### What's New

The new `train_kappa_gradual.py` script improves upon `train_kappa_sweep_parallel.py`:

**Old approach** (`train_kappa_sweep_parallel.py`):
- Fixed step size: 0.1
- Saves all intermediate models (21 total)
- κ values: 0, ±0.1, ±0.2, ..., ±1.0

**New approach** (`train_kappa_gradual.py`):
- **Configurable step size** (default: 0.05 for smoother learning)
- **Selective checkpoints** (only saves at milestones: 0.2, 0.4, 0.6, 0.8, 1.0)
- **Curriculum learning**: Gradually adjusts kappa in small steps
- More robust to gradient issues

### Command-Line Options

```bash
python3 train_kappa_gradual.py <model_file> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--step` | `0.05` | Kappa step size (smaller = more gradual) |
| `--checkpoints` | `0.2,0.4,0.6,0.8,1.0` | Kappa values to save (comma-separated) |
| `--direction` | `both` | Train `positive`, `negative`, or `both` |
| `--finetune-iter` | `1000` | Iterations per kappa step |
| `--finetune-lr` | `0.001` | Learning rate for finetuning |
| `--save-all` | `False` | Save all intermediate steps (not just checkpoints) |
| `--skip-base` | `False` | Skip training base model (if it already exists) |

### Examples

**1. Default: Gradual training with key checkpoints**

```bash
python3 train_kappa_gradual.py models/gambling.py
```

Trains in steps of 0.05:
- Positive: 0 → 0.05 → 0.10 → ... → 1.0 (saves at 0.2, 0.4, 0.6, 0.8, 1.0)
- Negative: 0 → -0.05 → -0.10 → ... → -1.0 (saves at -0.2, -0.4, -0.6, -0.8, -1.0)
- Total saved: 11 models (1 base + 10 checkpoints)

**2. Faster training with larger steps**

```bash
python3 train_kappa_gradual.py models/gambling.py --step 0.1
```

**3. Only train positive kappa**

```bash
python3 train_kappa_gradual.py models/gambling.py --direction positive
```

**4. Custom checkpoints**

```bash
python3 train_kappa_gradual.py models/gambling.py \
    --checkpoints "0.25,0.5,0.75,1.0"
```

**5. Save all intermediate steps (like old script)**

```bash
python3 train_kappa_gradual.py models/gambling.py \
    --step 0.1 \
    --save-all
```

**6. Resume from existing base model**

```bash
python3 train_kappa_gradual.py models/gambling.py --skip-base
```

**7. Very gradual training with many small steps**

```bash
python3 train_kappa_gradual.py models/gambling.py \
    --step 0.02 \
    --finetune-iter 500
```

Trains 50 steps per chain (0.02 increments from 0 to 1.0), but only saves at key milestones.

---

## Advanced Usage

### Recommended Workflow for Negative Kappa

If you're experiencing gradient collapse with negative kappa:

**Step 1**: Add entropy regularization to your model

```python
# models/gambling.py
config = {
    # ... other parameters ...
    'entropy_coef': 0.01,  # Prevent policy collapse
}
```

**Step 2**: Train with smaller steps and more iterations

```bash
python3 train_kappa_gradual.py models/gambling.py \
    --step 0.02 \
    --finetune-iter 2000 \
    --direction negative
```

**Step 3**: Monitor the debug output

Look for:
```
Policy Entropy:
  Mean entropy: 0.452134  # Should stay > 0.01
  Max possible entropy (log(2)): 0.693147

Gradient Statistics (policy - before clipping):
  Total norm: 7.770147  # Should be > 0
```

**Step 4**: Adjust entropy_coef if needed

- If gradients still collapse: increase `entropy_coef` to 0.05
- If learning is too slow: decrease `entropy_coef` to 0.005

### Comparing Old vs New Script

| Feature | `train_kappa_sweep_parallel.py` | `train_kappa_gradual.py` |
|---------|----------------------------------|---------------------------|
| Step size | Fixed (0.1) | Configurable (default 0.05) |
| Saved models | All 21 | Checkpoints only (default 11) |
| Flexibility | Low | High (many options) |
| Disk usage | Higher (saves all) | Lower (saves checkpoints) |
| Training smoothness | Good | Better (smaller steps) |
| Best for | Quick sweeps | Production training, research |

### Troubleshooting

**Problem**: Training is too slow

**Solution**: Use larger steps and fewer iterations
```bash
python3 train_kappa_gradual.py models/gambling.py \
    --step 0.1 \
    --finetune-iter 500
```

---

**Problem**: Gradient collapse persists even with entropy regularization

**Solutions**:
1. Increase `entropy_coef` to 0.05 or 0.1
2. Lower learning rate: `--finetune-lr 0.0005`
3. Use smaller kappa steps: `--step 0.01`
4. Train base model longer before finetuning

---

**Problem**: Model performance degrades with entropy regularization

**Solution**: Entropy is too high; reduce `entropy_coef`:
```python
'entropy_coef': 0.001,  # Smaller bonus
```

---

**Problem**: Want to train specific kappa values only

**Solution**: Use custom checkpoints and matching step size
```bash
python3 train_kappa_gradual.py models/gambling.py \
    --step 0.2 \
    --checkpoints "0.2,0.4,0.6,0.8,1.0"
```

---

## Summary

### Key Improvements

1. **New gradual training script** with flexible options
2. **Entropy regularization** to prevent policy collapse
3. **Better diagnostics** to detect gradient issues early
4. **Configurable checkpoints** to save only what you need

### Recommended Settings for Your Use Case

Based on your requirements (gradually go from 0 to 1, save at 0.2, 0.4, 0.6, 0.8):

```bash
# Add to your model config
config = {
    # ... existing parameters ...
    'entropy_coef': 0.01,  # Fix gradient collapse
}

# Run the training
python3 train_kappa_gradual.py models/gambling.py \
    --step 0.05 \
    --checkpoints "0.2,0.4,0.6,0.8,1.0"
```

This will:
- ✅ Gradually increase kappa from 0 to 1 in steps of 0.05
- ✅ Save checkpoints at exactly 0.2, 0.4, 0.6, 0.8, 1.0
- ✅ Prevent gradient collapse with entropy regularization
- ✅ Train both positive and negative kappa in parallel

### Quick Reference

```bash
# Most common use case
python3 train_kappa_gradual.py models/gambling.py

# With entropy regularization (recommended)
# Add 'entropy_coef': 0.01 to your model config first
python3 train_kappa_gradual.py models/gambling.py

# Only positive kappa with custom checkpoints
python3 train_kappa_gradual.py models/gambling.py \
    --direction positive \
    --checkpoints "0.2,0.4,0.6,0.8,1.0"
```

---

## References

- **Entropy regularization**: Williams & Peng (1991) "Function Optimization using Connectionist Reinforcement Learning Algorithms"
- **Risk-sensitive RL**: Mihatsch & Neuneier (2002) "Risk-sensitive reinforcement learning"
