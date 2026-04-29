#!/bin/bash
# Quick training script for distributional RL features

# This simpler version just trains one level at a time
# Usage: ./train_distributional_quick.sh [level] [seed]
# Example: ./train_distributional_quick.sh 0 1

set -e

LEVEL=${1:-0}
SEED=${2:-1}

if [ "$LEVEL" -lt 0 ] || [ "$LEVEL" -gt 3 ]; then
    echo "Error: level must be 0-3"
    echo "Usage: $0 <level> [seed]"
    echo "  level 0: Original (no distributional)"
    echo "  level 1: Distributional critic"
    echo "  level 2: + Context quantile selection"
    echo "  level 3: + Context temperature"
    exit 1
fi

# Create training script
cat > /tmp/train_dist_level.py << 'EOF'
import sys
sys.path.insert(0, '.')

from pyrl.model import Model
from tasks.gambling import inputs, actions

LEVEL = int(sys.argv[1])
SEED = int(sys.argv[2])

configs = [
    {
        'name': 'level0_original',
        'use_distributional_critic': False,
        'use_context_quantile_selection': False,
        'use_context_temperature': False,
    },
    {
        'name': 'level1_distributional',
        'use_distributional_critic': True,
        'use_context_quantile_selection': False,
        'use_context_temperature': False,
    },
    {
        'name': 'level2_dist_context_q',
        'use_distributional_critic': True,
        'use_context_quantile_selection': True,
        'use_context_temperature': False,
    },
    {
        'name': 'level3_dist_full',
        'use_distributional_critic': True,
        'use_context_quantile_selection': True,
        'use_context_temperature': True,
    },
]

cfg = configs[LEVEL]

config = {
    'inputs': inputs,
    'actions': actions,
    'max_iter': 1000,
    'use_distributional_critic': cfg['use_distributional_critic'],
    'use_context_quantile_selection': cfg['use_context_quantile_selection'],
    'use_context_temperature': cfg['use_context_temperature'],
    'n_quantiles': 5,
    'quantile_huber_kappa': 1.0,
    'temperature_base': 1.0,
    'temperature_context_scale': 0.5,
}

model = Model(config=config)
savefile = f'data/weights/gambling/gambling_{cfg["name"]}.pkl'
print(f"\n✓ Training Level {LEVEL} ({cfg['name']}) with seed {SEED}\n")
model.train(savefile, seed=SEED)
print(f"\n✓ Level {LEVEL} training complete!\n")
EOF

python3 /tmp/train_dist_level.py "$LEVEL" "$SEED"
rm /tmp/train_dist_level.py
