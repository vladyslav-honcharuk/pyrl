# Gradient Collapse: Architecture Analysis

## Critical Findings

I found several architectural issues that could cause gradients to zero out, especially with negative kappa values.

---

## Issue 1: ReLU Activation in SimpleRNN (If Used)

**Location**: `pyrl/simple.py`, line 123

```python
def recurrent_step(self, u, q, x_tm1):
    # ...
    r_tm1 = torch.relu(x_tm1)  # ⚠️ POTENTIAL PROBLEM
    # ...
```

### What's Wrong

**ReLU "Dead Neurons" Problem**:
- If `x_tm1 < 0`, then `relu(x_tm1) = 0`
- Gradient of ReLU for negative inputs: **exactly 0**
- Once a neuron goes negative, it stays "dead" (no gradient flow)

### When This Becomes Critical

With **negative kappa** (risk-seeking):
1. Early random success → Large positive TD error
2. Risk-seeking amplifies: `δ' = 1.2 × δ` (for negative δ)
3. Network updates aggressively
4. Some neurons may become negative → ReLU kills them
5. Dead neurons have zero gradients → learning stops in those units
6. Eventually entire network could collapse

### Mismatch Issue

There's also an **inconsistency** in SimpleRNN:

```python
# In recurrent_step (line 123):
r_tm1 = torch.relu(x_tm1)  # Uses ReLU

# But the base class firing_rate (networks_base.py:46):
def firing_rate(self, x):
    return torch.tanh(x)  # Uses tanh!
```

The policy gradient calls `firing_rate()` which uses `tanh`, but the recurrent dynamics use `relu`. This mismatch could cause issues.

---

## Issue 2: Wout Initialization (GRU - Your Likely Network)

**Location**: `pyrl/gru.py`, lines 136-142

```python
# Output weights
# Initialize Wout with small random values. Avoid all-zero initialization
# because it produces identical logits and zero gradients for the policy.
std = self.config['Wout'] if self.config['Wout'] > 0 else 0.05
print(f"[ {self.network_name} ] Initialize Wout to random normal (std={std}).")
Wout = std * rng.normal(size=(self.N, self.Nout))
```

### The Problem

From `pyrl/configs.py`:
```python
# Default configs
defaults = {
    'Wout': 0,  # ⚠️ This means Wout is initialized with std=0.05 (very small!)
    # ...
}
```

### Why This Matters

**Small Wout → Near-uniform policy early in training**:
1. Small weights → similar logits for all actions
2. Softmax produces near-uniform probabilities (e.g., [0.50, 0.50])
3. Policy is random at first ✓ (this is good!)

**But with negative kappa**:
1. Random policy gets lucky → positive reward
2. Risk-seeking amplifies: `δ' = 1.2 × δ`
3. Policy shifts dramatically toward lucky action
4. After a few updates → deterministic policy: [0.999, 0.001]
5. **Log probabilities**: `log(0.999) ≈ -0.001` (nearly zero!)
6. **Gradient = log_prob × advantage ≈ 0 × advantage ≈ 0**

This is exactly what your logs show:
```
Log-probabilities:
  Mean: -0.000000  ← Nearly zero!
  Std: 0.000000

Gradient Statistics:
  Total norm: 0.000000  ← Collapsed!
```

---

## Issue 3: Log Softmax Numerical Issues

**Location**: `pyrl/simple.py` and `pyrl/gru.py`, `log_output` methods

```python
def log_output(self, r):
    logits = torch.matmul(r, self.Wout) + self.bout

    if self.f_out == 'softmax':
        return F.log_softmax(logits, dim=-1)  # This is numerically stable
```

### Potential Issue

When the policy becomes **extremely confident**:
```python
logits = [10.0, -10.0]  # Very confident about first action
softmax = [0.9999, 0.0001]
log_softmax = [-0.0001, -10.0001]
```

If action 0 is always chosen:
- `log_prob[action=0] = -0.0001` (nearly 0)
- Gradient ∝ log_prob × advantage ≈ 0

**This is NOT a numerical issue** (log_softmax is stable), but a **mathematical property**: when policy is deterministic, gradient signal vanishes.

---

## Issue 4: Policy Collapse Mechanism

Your logs show the **complete sequence of collapse**:

### Step 600 (Early Negative Kappa):
```
Advantage (delta_prime):
  Mean: -0.005360
  Positive: 36.6%
  Negative: 63.4%

Log-probabilities:
  Mean: -0.000000  ← Already collapsed!
  Std: 0.000000

Gradient norm: 0.000000  ← No learning
```

### What Happened

1. **Initialization**: Policy starts random (entropy ≈ 0.69 for 2 actions)
2. **Early lucky guess**: Random policy happens to make correct choice
3. **Risk-seeking amplification**: Negative kappa amplifies positive signals
4. **Premature convergence**: Policy becomes 100% confident in that action
5. **Entropy collapse**: Entropy → 0 (deterministic policy)
6. **Gradient collapse**: log(1.0) = 0 → no gradient → stuck forever

### Why Positive Kappa Works Better

From your logs (Step 1300, κ=+0.2):
```
Log-probabilities:
  Mean: -0.005719  ← Healthy non-zero value
  Std: 0.117366    ← Good variance

Gradient norm: 7.770147  ← Strong gradients!
```

**Risk-averse behavior** (positive κ):
- Dampens early successes (η⁺ = 0.8)
- Policy stays exploratory longer
- Gradients remain healthy
- Eventually converges to good solution

---

## Root Cause Summary

| Issue | Impact | Severity with Negative κ |
|-------|--------|-------------------------|
| **ReLU dead neurons** (SimpleRNN) | Neurons can permanently die | High (if using SimpleRNN) |
| **Small Wout init** | Policy starts with low confidence | Medium (exacerbated by risk-seeking) |
| **Deterministic policy** | log(p) ≈ 0 → gradient ≈ 0 | **CRITICAL** (main cause) |
| **No exploration mechanism** | Can't escape local minima | **CRITICAL** |

---

## Verification: Which Network Are You Using?

Check your model config file to confirm network type:

```python
# In your model file (e.g., models/gambling.py)
config = {
    'network_type': 'gru',  # or 'simple'?
    # ...
}
```

**Default** (from `pyrl/configs.py`): `'network_type': 'gru'`

### If Using GRU (Most Likely)

- ✅ No ReLU dead neuron problem (uses tanh)
- ✅ Better gradient flow
- ❌ Still susceptible to policy collapse with negative kappa

### If Using SimpleRNN

- ❌ ReLU dead neuron problem
- ❌ Mismatch between `recurrent_step` (relu) and `firing_rate` (tanh)
- ❌ Even more susceptible to gradient issues

---

## Solutions

### Solution 1: Entropy Regularization (Already Implemented ✓)

**What we added**:
```python
# In pyrl/policygradient.py
entropy_bonus = entropy_coef * mean_entropy
loss = -J + reg - entropy_bonus
```

**How it helps**:
- Prevents policy from becoming deterministic
- Maintains exploration (entropy > 0)
- Keeps gradients flowing
- **This is the primary fix for your issue!**

**Enable it**:
```python
config = {
    'entropy_coef': 0.01,  # Recommended for negative kappa
}
```

### Solution 2: Gradient Clipping (Already Enabled ✓)

From `pyrl/configs.py`:
```python
'grad_clip': 5,  # Prevents exploding gradients
```

This helps with exploding gradients but **doesn't fix vanishing gradients**.

### Solution 3: Better Wout Initialization (Optional)

Increase Wout initialization for faster early learning:

```python
# In your model config
config = {
    'Wout': 0.1,  # Instead of default 0 (which uses 0.05)
}
```

**Pros**: Stronger initial gradients
**Cons**: May need careful tuning

### Solution 4: Learning Rate Schedule (Advanced)

Use higher learning rate early, decay later:

```python
config = {
    'lr': 0.005,  # Higher than default 0.002
}
```

Then manually decrease after partial training.

---

## Recommended Action

Based on your logs showing **gradient collapse with negative kappa**, here's what to do:

### Immediate Fix (Use This)

**Add to your model config**:
```python
config = {
    # ... existing parameters ...
    'entropy_coef': 0.01,  # Prevents policy collapse
}
```

**Run training**:
```bash
python3 examples/train_kappa_gradual.py models/gambling.py
```

### Monitor These Metrics

Look for in the debug output:
```
Policy Entropy:
  Mean entropy: 0.452134  # Should stay > 0.1 with entropy_coef=0.01

Gradient Statistics:
  Total norm: 7.770147  # Should be > 0.1

Log-probabilities:
  Std: 0.117366  # Should be > 0.01 (indicates diversity)
```

### If Gradient Collapse Persists

Try stronger entropy regularization:
```python
'entropy_coef': 0.05,  # Or even 0.1
```

### If Learning Becomes Too Slow

Reduce entropy bonus:
```python
'entropy_coef': 0.005,  # Weaker regularization
```

---

## Why Your Current Code Shows Collapse

Looking at your logs:

**Negative kappa (κ=-0.2)**:
```
Step 600:
  Log-probabilities: Mean: -0.000000, Std: 0.000000
  Gradient norm: 0.000000
  → COLLAPSED
```

**Positive kappa (κ=+0.2)**:
```
Step 1300:
  Log-probabilities: Mean: -0.005719, Std: 0.117366
  Gradient norm: 7.770147
  → HEALTHY
```

**Root cause**: Risk-seeking (negative κ) causes premature convergence to deterministic policy → entropy = 0 → gradients = 0 → stuck.

**Solution**: Entropy regularization maintains exploration → entropy > 0 → gradients > 0 → continues learning.

---

## Additional Diagnostic: Check Network Type

Run this to see which network you're using:

```bash
cd examples
python3 -c "
import sys
sys.path.insert(0, '..')
from pyrl.model import Model

# Load your model config
# (You'll need to adapt this to your specific model file)
import models.gambling as gambling_model  # Replace with your model

config = gambling_model.config
print(f\"Network type: {config.get('network_type', 'gru')}\")
print(f\"Entropy coef: {config.get('entropy_coef', 0.0)}\")
"
```

---

## Summary

**Main culprit**: Policy becomes deterministic → log(probability) ≈ 0 → gradient ≈ 0

**Why it happens with negative kappa**: Risk-seeking amplifies early successes → premature convergence

**The fix**: Entropy regularization (already implemented in your code!)

**How to use**: Add `'entropy_coef': 0.01` to your model config

This should solve your gradient collapse issue! 🎯
