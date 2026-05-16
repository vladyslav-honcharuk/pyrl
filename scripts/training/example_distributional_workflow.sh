#!/bin/bash
# Example workflow for training and analyzing models with distributional RL features

# This script demonstrates how to train 4 models with different distributional configurations:
# 1. Original baseline (no distributional features)
# 2. Distributional critic only (5-quantile V(s))
# 3. Distributional + context quantile selection
# 4. Full distributional with all context modulation

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"

MODEL="tasks/gambling.py"

echo "================================================================================"
echo "DISTRIBUTIONAL RL EXAMPLE WORKFLOW"
echo "================================================================================"
echo ""
echo "This example trains 4 models with different distributional configurations:"
echo "  Level 0: Original (baseline, no distributional features)"
echo "  Level 1: Distributional critic only (5-quantile V(s))"
echo "  Level 2: + Context-based quantile selection"
echo "  Level 3: + Context-based temperature modulation (full features)"
echo ""
echo "================================================================================"
echo ""

# ============================================================================
# Level 0: Original (baseline)
# ============================================================================
echo ""
echo "================================================================================"
echo "LEVEL 0: Training original baseline (no distributional features)"
echo "================================================================================"
echo ""

python3 "$TRAIN_SCRIPT" "$MODEL" train \
    --seed 1 \
    --level 0 \
    --suffix _level0_original

echo ""
echo "✓ Level 0 training complete!"
echo ""

# ============================================================================
# Level 1: Distributional critic only
# ============================================================================
echo ""
echo "================================================================================"
echo "LEVEL 1: Training with distributional critic (5-quantile V(s))"
echo "================================================================================"
echo ""

python3 "$TRAIN_SCRIPT" "$MODEL" train \
    --seed 1 \
    --level 1 \
    --suffix _level1_distributional

echo ""
echo "✓ Level 1 training complete!"
echo ""

# ============================================================================
# Level 2: Distributional + Context quantile selection
# ============================================================================
echo ""
echo "================================================================================"
echo "LEVEL 2: Training with distributional + context quantile selection"
echo "================================================================================"
echo ""

python3 "$TRAIN_SCRIPT" "$MODEL" train \
    --seed 1 \
    --level 2 \
    --suffix _level2_dist_context_q

echo ""
echo "✓ Level 2 training complete!"
echo ""

# ============================================================================
# Level 3: Full features (distributional + context quantile + context temperature)
# ============================================================================
echo ""
echo "================================================================================"
echo "LEVEL 3: Training with all distributional features"
echo "================================================================================"
echo ""

python3 "$TRAIN_SCRIPT" "$MODEL" train \
    --seed 1 \
    --level 3 \
    --suffix _level3_dist_full

echo ""
echo "✓ Level 3 training complete!"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "Models trained:"
echo "  Level 0: data/weights/gambling/gambling_level0_original/"
echo "  Level 1: data/weights/gambling/gambling_level1_distributional/"
echo "  Level 2: data/weights/gambling/gambling_level2_dist_context_q/"
echo "  Level 3: data/weights/gambling/gambling_level3_dist_full/"
echo ""
echo "Next steps for analysis:"
echo "  1. Compare behavioral heatmaps across all 4 levels"
echo "  2. Analyze learned quantile distributions (Level 1-3)"
echo "  3. Examine context-to-quantile mappings (Level 2-3)"
echo "  4. Check temperature modulation effects (Level 3)"
echo ""
echo "================================================================================"
