#!/bin/bash
# Example workflow for training and analyzing models with per-neuron kappa distributions

# This script demonstrates how to:
# 1. Train a model with per-neuron Gaussian kappa
# 2. Train a model with per-neuron uniform kappa
# 3. Train a baseline model with single kappa
# 4. Analyze each model

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"
ANALYZE_SCRIPT="$SCRIPT_DIR/../analysis/analyze_single_model.py"

MODEL="models/gambling.py"

echo "================================================================================"
echo "PER-NEURON KAPPA EXAMPLE WORKFLOW"
echo "================================================================================"
echo ""
echo "This example trains 3 models with different kappa configurations:"
echo "  1. Gaussian per-neuron kappa (mean=0.0, std=0.2)"
echo "  2. Uniform per-neuron kappa (range=-0.4 to +0.4)"
echo "  3. Single kappa baseline (kappa=0.0)"
echo ""
echo "Then analyzes all three models for comparison."
echo "================================================================================"
echo ""

# ============================================================================
# 1. Train model with Gaussian per-neuron kappa
# ============================================================================
echo ""
echo "================================================================================"
echo "STEP 1: Training with Gaussian per-neuron kappa"
echo "================================================================================"
echo "Distribution: Gaussian(mean=0.0, std=0.2)"
echo "Each neuron gets kappa sampled from N(0, 0.2), clipped to [-1, 1]"
echo ""

python3 "$TRAIN_SCRIPT" "$MODEL" train \
    --kappa-dist gaussian \
    --kappa-dist-mean 0.0 \
    --kappa-dist-std 0.2 \
    --suffix _gaussian_kappa

echo ""
echo "✓ Gaussian per-neuron kappa model trained successfully!"
echo ""

# ============================================================================
# 2. Train model with uniform per-neuron kappa
# ============================================================================
echo ""
echo "================================================================================"
echo "STEP 2: Training with uniform per-neuron kappa"
echo "================================================================================"
echo "Distribution: Uniform(low=-0.4, high=0.4)"
echo "Each neuron gets kappa uniformly sampled from [-0.4, 0.4]"
echo ""

python3 "$TRAIN_SCRIPT" "$MODEL" train \
    --kappa-dist uniform \
    --kappa-dist-low -0.4 \
    --kappa-dist-high 0.4 \
    --suffix _uniform_kappa

echo ""
echo "✓ Uniform per-neuron kappa model trained successfully!"
echo ""

# ============================================================================
# 3. Train baseline model with single kappa
# ============================================================================
echo ""
echo "================================================================================"
echo "STEP 3: Training baseline with single kappa"
echo "================================================================================"
echo "Single kappa: 0.0 (all neurons use the same value)"
echo ""

python3 "$TRAIN_SCRIPT" "$MODEL" train \
    --kappa 0.0 \
    --suffix _single_kappa

echo ""
echo "✓ Single kappa baseline model trained successfully!"
echo ""

# ============================================================================
# 4. Analyze all models
# ============================================================================
echo ""
echo "================================================================================"
echo "STEP 4: Analyzing all models"
echo "================================================================================"
echo ""

echo "Analyzing Gaussian per-neuron kappa model..."
python3 "$ANALYZE_SCRIPT" "$MODEL" --suffix _gaussian_kappa --kappa 0.0 --trials 5

echo ""
echo "Analyzing Uniform per-neuron kappa model..."
python3 "$ANALYZE_SCRIPT" "$MODEL" --suffix _uniform_kappa --kappa 0.0 --trials 5

echo ""
echo "Analyzing single kappa baseline model..."
python3 "$ANALYZE_SCRIPT" "$MODEL" --suffix _single_kappa --kappa 0.0 --trials 5

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "Models trained:"
echo "  1. Gaussian per-neuron kappa: work/data/gambling_gaussian_kappa/"
echo "  2. Uniform per-neuron kappa:  work/data/gambling_uniform_kappa/"
echo "  3. Single kappa baseline:     work/data/gambling_single_kappa/"
echo ""
echo "Analysis results saved to:"
echo "  work/figs/gambling_gaussian_kappa/"
echo "  work/figs/gambling_uniform_kappa/"
echo "  work/figs/gambling_single_kappa/"
echo ""
echo "Next steps:"
echo "  - Compare behavioral heatmaps across models"
echo "  - Analyze value neuron diversity in per-neuron models"
echo "  - Check if per-neuron kappa improves performance"
echo ""
echo "To visualize the per-neuron kappa distributions:"
echo "  python3 -c \\"
echo "    from pyrl import utils; import matplotlib.pyplot as plt; import numpy as np;"
echo "    save = utils.load('work/data/gambling_gaussian_kappa/gambling_gaussian_kappa.pkl');"
echo "    kappa = save['kappa_neurons'];"
echo "    plt.figure(figsize=(10,4));"
echo "    plt.subplot(1,2,1); plt.hist(kappa, bins=30, edgecolor='black');"
echo "    plt.xlabel('Kappa value'); plt.ylabel('Number of neurons');"
echo "    plt.title('Gaussian Distribution');"
echo "    save2 = utils.load('work/data/gambling_uniform_kappa/gambling_uniform_kappa.pkl');"
echo "    kappa2 = save2['kappa_neurons'];"
echo "    plt.subplot(1,2,2); plt.hist(kappa2, bins=30, edgecolor='black');"
echo "    plt.xlabel('Kappa value'); plt.ylabel('Number of neurons');"
echo "    plt.title('Uniform Distribution');"
echo "    plt.tight_layout(); plt.savefig('work/figs/kappa_distributions.png'); print('Saved to work/figs/kappa_distributions.png')\\"
echo ""
echo "================================================================================"
