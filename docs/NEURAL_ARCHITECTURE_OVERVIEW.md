# Neural Architecture Overview: Risk-Sensitive Contextual RL Model

## Executive Summary

This document provides a comprehensive overview of a recurrent neural network (RNN) model designed to study **risk-sensitive decision-making** with **contextual modulation**. The architecture combines three key computational components:

1. **Risk-sensitive reinforcement learning** (kappa parameter)
2. **Two-pass value architecture** (critic-to-actor feedback)
3. **Contextual kappa modulation** (state-dependent risk preferences)

The model is designed as a computational analog of brain circuits involved in value-based decision-making, with potential mappings to orbitofrontal cortex (OFC), ventral tegmental area (VTA), and related structures.

---

## 1. Task Environment: Gambling Task

### Task Structure
The agent performs a **two-alternative forced choice gambling task** where it must:
1. Maintain fixation (250ms)
2. Observe two colored targets representing different gambles (250ms)
3. Choose left or right target (260ms)
4. Receive probabilistic reward based on choice

### Gambles
- **25 unique gambles** arranged in a 5×5 matrix:
  - **Rows (5 levels)**: Probability of reward (0.1, 0.3, 0.5, 0.7, 0.9)
  - **Columns (5 levels)**: Reward magnitude (scaled from 1.11 to 25)
  - All gambles have **equal expected value** (EV ≈ 1.0-2.5 after scaling)

### Sensory Encoding
- Each gamble is represented by a unique **RGB color triplet**
- Network receives 7 input channels:
  - 1 fixation cue
  - 3 RGB channels for left target
  - 3 RGB channels for right target
- Input noise: σ = 0.1 (Gaussian)

### Actions
- **FIXATE**: Maintain central fixation
- **CHOOSE-LEFT**: Select left target
- **CHOOSE-RIGHT**: Select right target

---

## 2. Neural Architecture

### 2.1 Policy Network (Actor)

**Network Type**: Gated Recurrent Unit (GRU)

**Dimensions**:
- **Input (Nin)**: 7 (fixation + 6 RGB channels)
  - In two-pass mode: 7 + N_value (extended with value features)
- **Hidden (N)**: 100 recurrent units
- **Output (Nout)**: 3 (action probabilities via softmax)

**Recurrent Dynamics**:
```
GRU gates:
  z_t = σ(Win_z @ u_t + Wrec_z @ r_{t-1} + b_z)  [update gate]
  r_t = σ(Win_r @ u_t + Wrec_r @ r_{t-1} + b_r)  [reset gate]

Candidate activation:
  h_t = tanh(Win_h @ u_t + Wrec_h @ (r_t ⊙ r_{t-1}))

State update:
  x_t = (1 - α) x_{t-1} + α [(1 - z_t) ⊙ h_t + z_t ⊙ r_{t-1}]

Firing rate:
  r_t = relu(x_t)

Action probabilities:
  π(a|s_t) = softmax(Wout @ r_t + bout)
```

**Key Parameters**:
- **α = dt/τ**: Leak rate (τ = 100ms, dt = 10ms → α = 0.1)
- **Recurrent noise**: σ_rec² = 0.01 (scaled by 2τ/dt)
- **Spectral radius (ρ)**: 2.0 (slightly unstable for rich dynamics)
- **Initial output bias (bout)**: [3.0, 0.0, 0.0] (favors fixation initially)

**Regularization**:
- **L2 on firing rates**: Encourages sparse activity
- **L1/L2 on Wrec**: Prevents runaway recurrent weights

---

### 2.2 Baseline Network (Critic)

**Network Type**: Gated Recurrent Unit (GRU)

**Dimensions**:
- **Input (Nin)**: 100 (policy firing rates) + 3 (one-hot action) + 1 (context, optional)
  - Total: 103 in vanilla mode, 104 in contextual mode
- **Hidden (N)**: 100 recurrent units
- **Output (Nout)**: 1 (state-value prediction)

**Purpose**:
- Predicts expected future return V(s_t) for variance reduction
- In contextual mode: learns to use context signal for risk-appropriate value estimation

**Training**:
- **Asymmetric Least Squares (Expectile Regression)**
- Loss weights determined by **kappa-modulated expectiles**:
  - Risk-averse (κ < 0): Overweight negative prediction errors
  - Risk-neutral (κ = 0): Standard MSE
  - Risk-seeking (κ > 0): Overweight positive prediction errors

---

### 2.3 Two-Pass Value Architecture

**Motivation**: Allow value network representations to directly influence policy decisions (analogous to OFC→motor cortex projections).

**Architecture Flow**:

```
FIRST PASS (Compute Value):
  u_t → Policy Network (with zero-padded value dims) → x_t^{policy}

  x_t^{policy} → Baseline Network → V(s_t)

  r_t^{baseline} = relu(x_t^{baseline})  [value network firing rates]


SECOND PASS (Recompute Policy with Value Input):
  u_t_extended = [u_t, prepare_value_features(r_t^{baseline}[:N_value])]

  u_t_extended → Policy Network → x_t^{policy,final}

  π(a|s_t) = softmax(Wout @ relu(x_t^{policy,final}))
```

**Value Feature Preprocessing**:
```python
def prepare_value_features(r_value):
    if standardize:
        r_value = (r_value - mean(r_value)) / (std(r_value) + 1e-8)
    r_value = r_value * gain  # Default gain = 3.0
    return r_value
```

**Key Parameters**:
- **value_influence**: Fraction of value neurons to use (0-1)
  - Default: 1.0 (all 100 value neurons)
- **value_feature_standardize**: Whether to z-score value features (default: True)
- **value_feature_gain**: Multiplicative scaling (default: 3.0)
- **value_grad_scale**: Gradient backprop through value feedback (default: 0.0 = detached)

**Critical Design Choice**:
- Value features are **detached** from policy gradients (value_grad_scale=0) to prevent instability
- This mimics **timescale separation**: value representations evolve slowly, policy adapts quickly

---

## 3. Risk-Sensitive Reinforcement Learning (Kappa)

### 3.1 Kappa Parameter (κ)

**Definition**: Risk-sensitivity parameter controlling preference for risky vs. safe options.

**Range**: κ ∈ [-1, +1]
- **κ < 0**: Risk-averse (prefer high-probability, low-reward)
- **κ = 0**: Risk-neutral (maximize expected value)
- **κ > 0**: Risk-seeking (prefer low-probability, high-reward)

**Computational Mechanism**: Kappa transforms the advantage function via **asymmetric expectile regression**.

---

### 3.2 Expectile-Based Advantage Estimation

The baseline network learns to predict state value using **kappa-modulated expectile regression**:

```
Advantage: δ_t = R_t - V(s_t)  [temporal difference error]

Expectile weights:
  η+ = 1 - κ  [weight for positive errors]
  η- = 1 + κ  [weight for negative errors]

Loss weights:
  w_t = η+ * I(δ_t > 0) + η- * I(δ_t < 0)

Baseline loss:
  L_baseline = Σ w_t * δ_t² * M_t
```

**Intuition**:
- **Risk-averse (κ=-0.5)**: η+=1.5, η-=0.5
  - Heavily penalize **underestimating** returns (positive errors)
  - Lightly penalize overestimating returns
  - Result: Baseline **overestimates** expected value → policy avoids risky choices

- **Risk-seeking (κ=+0.5)**: η+=0.5, η-=1.5
  - Lightly penalize underestimating returns
  - Heavily penalize **overestimating** returns (negative errors)
  - Result: Baseline **underestimates** expected value → policy seeks risky choices

---

### 3.3 Per-Neuron Kappa (Optional)

The model supports **heterogeneous risk preferences** across baseline neurons:

**Distributions**:
- **Gaussian**: κ_i ~ N(μ, σ²)
- **Uniform**: κ_i ~ U(low, high)

**Computation**:
```
For each baseline neuron i:
  η+_i = 1 - κ_i
  η-_i = 1 + κ_i

Expectile loss computed per-neuron, then averaged.
```

**Biological Motivation**: Different VTA/OFC neurons may have distinct risk sensitivities.

---

## 4. Contextual Kappa Modulation

### 4.1 Concept

**Goal**: Allow risk preferences to vary **dynamically** based on a context signal, analogous to:
- Internal states (hunger, stress, arousal)
- Task phases (early exploration vs. late exploitation)
- Environmental cues (threat level, resource abundance)

**Mechanism**: Add a **context input** to the baseline network that modulates kappa.

---

### 4.2 Architecture

**Context Signal (c)**:
- Scalar value representing environmental/internal context
- Training: sampled from uniform distribution c ~ U(-1, +1)
- Inference: can be fixed (tonic_c) to probe network behavior

**Context-Modulated Kappa**:
```
κ_trial = κ_base + κ_max * tanh(α_context * c)

where:
  κ_base: baseline kappa (e.g., 0)
  κ_max: maximum kappa deviation (default: 0.7)
  α_context: context_sensitivity parameter (0-1)
  tanh: ensures κ_trial ∈ [κ_base - κ_max, κ_base + κ_max]
```

**Baseline Network Input**:
```
Vanilla mode:    [r_policy (100), action (3)]
Contextual mode: [r_policy (100), action (3), context (1)]
```

**Training Objective**:
- Network must learn to **use context** to adjust value predictions appropriately
- E.g., when c > 0 (risk-seeking context), baseline should underestimate risky options less
- E.g., when c < 0 (risk-averse context), baseline should overestimate risky options more

---

### 4.3 Training Considerations

**Challenge**: Adding context dimension creates **covariate shift** if introduced mid-training.

**Solution (Progressive Training)**:
1. **Stage 1**: Train vanilla network (no context, no value feedback)
2. **Stage 2**: Fine-tune with two-pass value architecture
3. **Stage 3**: Fine-tune with context input added
   - **Keep value feature pipeline identical** to Stage 2
   - **Reduce learning rate** 10x to stabilize new context weights
   - **Optionally freeze policy network** for 200-500 iterations

---

## 5. Learning Algorithm: Policy Gradient

### 5.1 Monte Carlo Policy Gradient (REINFORCE)

**Objective**: Maximize expected return under policy π_θ.

```
J(θ) = E_π[R_total]

∇_θ J(θ) = E_π[∇_θ log π_θ(a_t|s_t) * A(s_t, a_t)]

where:
  A(s_t, a_t) = R_total - V(s_t)  [advantage with baseline for variance reduction]
  R_total = Σ r_t  (undiscounted episodic return)
```

**Policy Update**:
```python
# Compute log probabilities of chosen actions
logπ_t = log(π_θ(a_t | s_t))

# REINFORCE objective
J = Σ_t (logπ_t * A_t * M_t)  / batch_size

# Gradient ascent
loss = -J + regularization
optimizer.step()
```

**Baseline Update**:
```python
# Compute advantage
δ_t = R_total - V(s_t)

# Kappa-modulated weights
w_t = (1 - κ) * I(δ_t > 0) + (1 + κ) * I(δ_t < 0)

# Asymmetric MSE loss
L_baseline = Σ_t (w_t * δ_t²) / batch_size

optimizer_baseline.step()
```

---

### 5.2 Training Hyperparameters

**Learning Rates**:
- Policy network: lr = 0.0001
- Baseline network: lr = 0.0001

**Batch Sizes**:
- Training batch (n_gradient): 64 trials
- Validation batch (n_validation): 500 trials

**Optimization**:
- Optimizer: Adam
- Gradient clipping: Optional (default: None)
- Max iterations: 2000

**Recurrent Noise**:
- σ_rec² = 0.01 (for exploration during training)

---

## 6. Mapping to Brain Circuits

### 6.1 Anatomical Correspondences

| **Model Component** | **Putative Brain Region** | **Function** |
|---------------------|---------------------------|--------------|
| **Policy Network (GRU)** | **Dorsolateral Prefrontal Cortex (dlPFC)** / **Premotor Cortex** | Integrates sensory evidence over time, generates action plans |
| **Baseline Network (Critic)** | **Orbitofrontal Cortex (OFC)** / **Ventromedial PFC (vmPFC)** | Computes subjective value, encodes risk preferences |
| **Value → Policy Feedback** | **OFC → Premotor Projections** | Biases action selection based on value representations |
| **Kappa Parameter** | **VTA Dopamine Neurons** (heterogeneous risk coding) | Modulates learning signals based on risk sensitivity |
| **Context Signal** | **Anterior Cingulate Cortex (ACC)** / **Insula** | Encodes internal states (arousal, stress, hunger) |
| **Context → Baseline Input** | **ACC/Insula → OFC Projections** | Modulates value computation based on internal state |

---

### 6.2 Dopaminergic Risk-Sensitivity

**Hypothesis**: VTA dopamine neurons exhibit **heterogeneous risk coding**.

**Model Implementation**:
- **Per-neuron kappa**: Different baseline neurons have distinct κ_i values
- **Expectile regression**: Mimics dopamine neurons' asymmetric responses to positive/negative RPE

**Experimental Predictions**:
1. VTA neurons should show **heterogeneous responses** to identical RPEs
2. Risk-averse neurons: Stronger response to worse-than-expected outcomes
3. Risk-seeking neurons: Stronger response to better-than-expected outcomes

**Evidence**:
- Nakazawa, Isa, & Sasaki (2023): RNNs trained with kappa develop heterogeneous value representations
- Fiorillo et al. (2003): Dopamine neurons encode reward probability (uncertainty)
- Tobler et al. (2005): Some dopamine neurons prefer risky options

---

### 6.3 OFC Value Representations

**Role**: OFC encodes **state values** that guide decision-making.

**Model Implementation**:
- Baseline network receives **policy firing rates** (analogous to OFC receiving cortical input)
- In two-pass mode: OFC representations **feed back** to influence policy

**Computational Function**:
- **Risk-averse OFC**: Overestimates value of safe options
- **Risk-seeking OFC**: Underestimates value of safe options (making risky options relatively more attractive)

**Experimental Predictions**:
1. OFC lesions should impair risk-sensitive behavior
2. OFC neurons should show **asymmetric responses** to prediction errors based on individual risk preferences
3. Context signals (hunger, stress) should modulate OFC value coding

**Evidence**:
- Padoa-Schioppa & Assad (2006): OFC encodes subjective value
- Sul et al. (2010): OFC neurons encode risk-adjusted value
- Kennerley et al. (2011): OFC represents reward variability

---

### 6.4 Contextual Modulation of Risk Preferences

**Biological Phenomenon**: Risk preferences **vary with context**:
- Hunger increases risk-seeking for food
- Threat increases risk-aversion
- Social context modulates financial risk-taking

**Model Implementation**:
- **Context input to baseline network**: Modulates kappa dynamically
- **Learned mapping**: Network learns which contexts demand which risk preferences

**Brain Regions**:
- **ACC**: Monitors conflict, uncertainty, expected value of control
- **Insula**: Encodes interoceptive states (hunger, arousal, pain)
- **ACC/Insula → OFC**: Projects context signals to modulate value computation

**Experimental Predictions**:
1. Inactivating ACC/insula should eliminate context-dependent risk modulation
2. OFC neurons should show **context × value interactions**
3. Dopamine signals should reflect **context-modulated risk sensitivity**

---

## 7. Key Architectural Innovations

### 7.1 Two-Pass Value Architecture

**Innovation**: Value network computes representations, then **feeds back** to influence policy on the same timestep.

**Biological Plausibility**:
- OFC → premotor cortex projections are well-documented
- Timescale separation: OFC value representations update slowly, motor plans update rapidly

**Computational Advantage**:
- Richer action selection: Policy has access to **learned value features**, not just raw inputs
- Biologically realistic: Separates "what's valuable" (OFC) from "what to do" (motor cortex)

**Training Stability**:
- **Critical**: Value features must be **detached** from policy gradients
- **Critical**: Feature statistics must remain **consistent** across fine-tuning stages

---

### 7.2 Contextual Kappa Modulation

**Innovation**: Risk preferences are not fixed; they adapt to **context signals**.

**Biological Motivation**:
- Animals adjust risk preferences based on hunger, threat, social status
- Dopamine signals are modulated by internal states (Niv et al., 2007)

**Computational Challenge**:
- Adding context dimension mid-training creates **covariate shift**
- Solution: Progressive training with consistent value feature preprocessing

---

### 7.3 Expectile-Based Risk Sensitivity

**Innovation**: Use **asymmetric expectile regression** instead of traditional value estimation.

**Advantages over Alternatives**:
- **vs. Utility functions**: No need to specify U(r) functional form
- **vs. Risk-sensitive Bellman**: No exponential transformations (numerically unstable)
- **vs. Distributional RL**: Simpler, more biologically plausible

**Biological Plausibility**:
- Dopamine neurons show **asymmetric RPE coding** (Fiorillo et al., 2003)
- OFC neurons exhibit heterogeneous risk preferences (Sul et al., 2010)

---

## 8. Experimental Workflow

### 8.1 Training Pipeline

**Standard Training**:
```bash
# Train risk-neutral network
python scripts/training/train.py tasks/gambling.py train --seed 1

# Fine-tune with kappa
python scripts/training/train.py tasks/gambling.py finetune --kappa 0.5 --suffix _kappa0p5
```

**Progressive Contextual Training**:
```bash
# Train all 3 stages sequentially
python scripts/training/train_progressive_contextual.py --gpu

# With extra stability (freeze policy during Stage 3)
python scripts/training/train_progressive_contextual.py --gpu --freeze-policy-warmup 300
```

**Kappa Sweep**:
```bash
# Train models across kappa spectrum
python scripts/training/train_kappa_sweep_parallel.py --gpu
```

---

### 8.2 Analysis

**Single Model Analysis**:
```bash
python scripts/analysis/analyze_single_model.py tasks/gambling.py --suffix kappa0p5 --kappa 0.5
```

**Kappa Sweep Comparison**:
```bash
python scripts/analysis/analyze_kappa_sweep.py tasks/gambling.py
```

**Contextual Sweep**:
```bash
python scripts/analysis/run_contextual_kappa_sweep.py --gpu
```

---

## 9. Key Findings & Predictions

### 9.1 Behavioral Signatures

**Risk-Averse Networks (κ < 0)**:
- Prefer high-probability targets (p=0.9) over low-probability (p=0.1)
- Show **steep discount** of low-probability rewards
- Baseline network **overestimates** expected value

**Risk-Seeking Networks (κ > 0)**:
- Prefer low-probability, high-reward targets
- Show **preference reversals** when probabilities change
- Baseline network **underestimates** expected value

**Contextual Networks**:
- Dynamically adjust choice behavior based on context signal
- Context c > 0: More risk-seeking
- Context c < 0: More risk-averse

---

### 9.2 Neural-Level Predictions

**Value Network Representations**:
- Risk-averse: Neurons show **higher firing rates** for safe options
- Risk-seeking: Neurons show **higher firing rates** for risky options
- Contextual: Firing rates **co-vary with context signal**

**Policy Network Dynamics**:
- Two-pass architecture: Policy states should show **value-dependent modulation** during decision period
- Vanilla architecture: Policy states evolve independently of value predictions

---

### 9.3 Dopamine Predictions

**Heterogeneous Risk Coding**:
- Different dopamine neurons should have different κ_i values
- Per-neuron kappa predicts **individual RPE asymmetry**

**Context Modulation**:
- Dopamine signals should **interact with context**:
  - High arousal: Enhanced risk-seeking dopamine signals
  - High threat: Enhanced risk-averse dopamine signals

---

## 10. Open Questions & Future Directions

### 10.1 Biological Implementation

**Question**: How does the brain implement expectile-based risk sensitivity?
- **Candidate**: Opponent dopamine pathways (D1 vs. D2)
- **Test**: Optogenetically manipulate D1/D2 balance during risky decisions

**Question**: Where is the context signal computed?
- **Candidate**: ACC conflict monitoring + insula interoception
- **Test**: Record ACC/insula during context-dependent risk tasks

---

### 10.2 Learning Dynamics

**Question**: How does the brain discover appropriate context→kappa mappings?
- **Model prediction**: Slow learning in Stage 3 (requires co-adaptation)
- **Test**: Train animals on context-switching risk task, measure learning curves

**Question**: Can networks generalize context to novel situations?
- **Test**: Train on c ∈ [-1, +1], test on c ∈ [-2, +2]

---

### 10.3 Clinical Relevance

**Addiction**:
- Hypothesis: Dysregulated kappa (over-seeking risky rewards)
- Test: Fit kappa to addicted vs. control subjects in gambling tasks

**Anxiety**:
- Hypothesis: Pathologically negative kappa (excessive risk-aversion)
- Test: Context-dependent anxiety should modulate kappa

**Impulsivity**:
- Hypothesis: Deficient context→kappa learning (can't adjust risk to context)
- Test: ADHD patients show reduced context modulation

---

## 11. Technical Summary for Computational Neuroscientists

### Model Class
- **Architecture**: Recurrent Neural Network (GRU)
- **Learning**: Policy Gradient (REINFORCE with Monte Carlo returns)
- **Value Estimation**: Asymmetric Expectile Regression
- **Risk Mechanism**: Kappa-modulated expectile weights

### Key Equations

**GRU State Update**:
```
x_t = (1-α)x_{t-1} + α[(1-z_t)⊙h_t + z_t⊙r_{t-1}] + η_t
r_t = relu(x_t)
```

**Policy Gradient**:
```
∇_θ J = E[∇_θ log π_θ(a_t|s_t) · (R_total - V(s_t))]
```

**Expectile Loss**:
```
L_baseline = Σ_t w_t · (R_total - V(s_t))²
w_t = (1-κ)·I(δ_t>0) + (1+κ)·I(δ_t<0)
```

**Contextual Kappa**:
```
κ(c) = κ_base + κ_max · tanh(α_context · c)
```

---

## 12. References

### Primary Source
- Nakazawa, Isa, & Sasaki (2023): *Computational mechanisms of risk preference generated in recurrent neural networks* (Poster)

### Theoretical Foundations
- **Expectile Regression**: Newey & Powell (1987), *Asymmetric least squares estimation*
- **Risk-Sensitive RL**: Howard & Matheson (1972), *Risk-sensitive Markov decision processes*
- **Policy Gradient**: Williams (1992), *Simple statistical gradient-following algorithms*

### Neuroscience Evidence
- **Dopamine & Risk**: Fiorillo et al. (2003), *Discrete coding of reward probability and uncertainty*
- **OFC & Value**: Padoa-Schioppa & Assad (2006), *Neurons in OFC encode economic value*
- **OFC & Risk**: Sul et al. (2010), *Distinct functions of OFC subareas in reward-guided behavior*
- **Dopamine Heterogeneity**: Tobler et al. (2005), *Adaptive coding of reward value by dopamine neurons*
- **Context & Value**: Kennerley et al. (2011), *Neurons in OFC encode the value of chosen and unchosen actions*

---

## Document Metadata
- **Created**: 2026-04-28
- **Model**: Risk-Sensitive Contextual RNN
- **Codebase**: pyrl (sharp-panini branch)
- **Contact**: For questions about this architecture, refer to CLAUDE.md and code documentation
