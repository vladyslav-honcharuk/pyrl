#!/bin/bash
# Example workflow for training and analyzing models with distributional RL features

# This script demonstrates how to:
# 1. Train a baseline model (original non-distributional)
# 2. Train with distributional critic only (Level 1)
# 3. Train with distributional critic + context quantile selection (Level 2)
# 4. Train with all distributional features enabled (Level 3)
# 5. Analyze and compare all models

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"
MODEL="$REPO_ROOT/tasks/gambling.py"

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
echo "Then compares all four models for behavioral analysis."
echo "================================================================================"
echo ""

# ============================================================================
# Helper function to train with config
# ============================================================================
train_with_config() {
    local level=$1
    local name=$2
    local use_dist=$3
    local use_context_q=$4
    local use_context_temp=$5
    local description=$6

    echo ""
    echo "================================================================================"
    echo "LEVEL $level: $description"
    echo "================================================================================"
    echo "Configuration:"
    echo "  use_distributional_critic: $use_dist"
    echo "  use_context_quantile_selection: $use_context_q"
    echo "  use_context_temperature: $use_context_temp"
    echo ""

    # Create a temporary Python script with the config
    local temp_script=$(mktemp)
    cat > "$temp_script" << 'PYTHON_EOF'
import sys
sys.path.insert(0, '.')

from pyrl.model import Model
from tasks.gambling import inputs, actions

config = {
    'inputs': inputs,
    'actions': actions,
    'max_iter': 1000,
    'use_distributional_critic': USE_DIST_PLACEHOLDER,
    'use_context_quantile_selection': USE_CONTEXT_Q_PLACEHOLDER,
    'use_context_temperature': USE_CONTEXT_TEMP_PLACEHOLDER,
    'n_quantiles': 5,
    'quantile_huber_kappa': 1.0,
    'temperature_base': 1.0,
    'temperature_context_scale': 0.5,
}

model = Model(config=config)
savefile = 'data/weights/gambling/gambling_LEVEL_PLACEHOLDER.pkl'
print(f"Training Level LEVEL_PLACEHOLDER: DESCRIPTION_PLACEHOLDER")
model.train(savefile, seed=1)
PYTHON_EOF

    # Replace placeholders
    sed -i "s/USE_DIST_PLACEHOLDER/$use_dist/g" "$temp_script"
    sed -i "s/USE_CONTEXT_Q_PLACEHOLDER/$use_context_q/g" "$temp_script"
    sed -i "s/USE_CONTEXT_TEMP_PLACEHOLDER/$use_context_temp/g" "$temp_script"
    sed -i "s/LEVEL_PLACEHOLDER/$level/g" "$temp_script"
    sed -i "s|DESCRIPTION_PLACEHOLDER|$description|g" "$temp_script"

    # Run training
    python3 "$temp_script"
    
    rm "$temp_script"

    echo ""
    echo "✓ Level $level training complete!"
    echo ""
}

# ============================================================================
# Level 0: Original (baseline)
# ============================================================================
train_with_config 0 "original" "False" "False" "False" "Original baseline (no distributional features)"

# ============================================================================
# Level 1: Distributional critic only
# ============================================================================
train_with_config 1 "distributional" "True" "False" "False" "Distributional critic (5-quantile V(s))"

# ============================================================================
# Level 2: Distributional + Context quantile selection
# ============================================================================
train_with_config 2 "dist_context_q" "True" "True" "False" "Distributional + context quantile selection"

# ============================================================================
# Level 3: Full features (distributional + context quantile + context temperature)
# ============================================================================
train_with_config 3 "dist_full" "True" "True" "True" "Full distributional with all context modulation"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================================"
echo "WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "Models trained:"
echo "  Level 0: $REPO_ROOT/data/weights/gambling/gambling_level0_original/"
echo "  Level 1: $REPO_ROOT/data/weights/gambling/gambling_level1_distributional/"
echo "  Level 2: $REPO_ROOT/data/weights/gambling/gambling_level2_dist_context_q/"
echo "  Level 3: $REPO_ROOT/data/weights/gambling/gambling_level3_dist_full/"
echo ""
echo "Next steps for analysis:"
echo "  1. Compare behavioral heatmaps across all 4 levels"
echo "  2. Analyze learned quantile distributions (Level 1-3)"
echo "  3. Examine context-to-quantile mappings (Level 2-3)"
echo "  4. Check temperature modulation effects (Level 3)"
echo ""
echo "Example comparison:"
echo "  python3 scripts/analysis/compare_distributional_models.py \\"
echo "    --models level0_original level1_distributional level2_dist_context_q level3_dist_full"
echo ""
echo "================================================================================"
