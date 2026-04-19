#!/usr/bin/env python3
"""
Train all kappa models IN PARALLEL from the base (κ=0) model.

This script:
1. Checks if base model (κ=0) exists, or trains it if not
2. Finetunes ALL kappa values in PARALLEL from the base model:
   - Each model starts from κ=0 checkpoint
   - All train simultaneously (maximum parallelization)

Much faster than gradual or sequential training since all models train at once.

Usage:
    python3 scripts/training/train_kappa_all_parallel.py models/gambling.py
"""

import os
import sys
import subprocess
import numpy as np
from multiprocessing import Pool
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'train.py')


def resolve_model_file(model_file):
    """Resolve model file path relative to repository root."""
    if os.path.isabs(model_file):
        return model_file
    return os.path.abspath(os.path.join(REPO_ROOT, model_file))

def run_command(cmd, description):
    """Run a shell command and print output."""
    msg = f"\n{'='*80}\n{description}\n{'='*80}\nRunning: {' '.join(cmd)}\n"
    print(msg)

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        error_msg = f"\nError: Command failed with return code {result.returncode}"
        print(error_msg)
        return False
    return True

def kappa_to_suffix(kappa):
    """Convert kappa value to filename suffix."""
    if kappa == 0:
        return ""
    elif kappa < 0:
        return f"neg{abs(kappa):.1f}".replace('.', 'p')
    else:
        return f"pos{kappa:.1f}".replace('.', 'p')

def train_single_kappa(args):
    """Train a single kappa model from base (for parallel execution)."""
    kappa, modelfile, base_name, base_path, finetune_iter = args
    
    suffix = kappa_to_suffix(kappa)
    
    print(f"\n🔄 [PID {os.getpid()}] Starting κ={kappa:+.1f} from base model...")
    
    # Finetune from base model (κ=0)
    finetune_cmd = [
        'python3', TRAIN_SCRIPT, modelfile, 'finetune',
        '--kappa', str(kappa),
        '--suffix', suffix,
        '--pretrained', base_path,
        '--finetune-iter', str(finetune_iter),
        '--finetune-lr', '0.0005'  # Lower LR for fine-tuning
    ]
    
    success = run_command(
        finetune_cmd,
        f"[PID {os.getpid()}] Fine-tuning κ=0.0 → κ={kappa:+.1f}"
    )
    
    if success:
        save_name = f"{base_name}{suffix}"
        save_path = os.path.join(REPO_ROOT, 'data', 'weights', save_name, f'{save_name}.pkl')
        print(f"✓ [PID {os.getpid()}] Successfully trained κ={kappa:+.1f}")
        return (kappa, suffix, save_path, "✓")
    else:
        print(f"✗ [PID {os.getpid()}] Failed to train κ={kappa:+.1f}")
        return (kappa, suffix, "", "✗ FAILED")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/training/train_kappa_all_parallel.py <model_file>")
        print("Example: python3 scripts/training/train_kappa_all_parallel.py models/gambling.py")
        sys.exit(1)

    modelfile = resolve_model_file(sys.argv[1])
    base_name = os.path.splitext(os.path.basename(modelfile))[0]

    # Parameters
    finetune_iter = 3000  # Shorter since we're going directly from base
    kappa_step = 0.1
    
    # Number of parallel processes (adjust based on your CPU cores)
    # Leave 1-2 cores free for system
    n_processes = min(10, os.cpu_count() - 2) if os.cpu_count() else 4

    # Generate all kappa values (excluding 0)
    positive_kappas = [round(k, 1) for k in np.arange(0.1, 1.0 + kappa_step/2, kappa_step)]
    negative_kappas = [round(k, 1) for k in np.arange(-0.1, -1.0 - kappa_step/2, -kappa_step)]
    all_kappas = sorted(negative_kappas + positive_kappas)

    print(f"{'='*80}")
    print(f"KAPPA SWEEP TRAINING (ALL FROM BASE - PARALLEL)")
    print(f"{'='*80}")
    print(f"Model: {modelfile}")
    print(f"Fine-tune iterations per κ: {finetune_iter}")
    print(f"Parallel processes: {n_processes}")
    print(f"\nTraining strategy:")
    print(f"  Step 1: Check/train base (κ=0.0)")
    print(f"  Step 2: Train ALL {len(all_kappas)} models in PARALLEL from base:")
    print(f"    • Each starts from κ=0 checkpoint")
    print(f"    • All train simultaneously")
    print(f"\nKappa values to train:")
    print(f"  Negative: {' '.join([f'{k:+.1f}' for k in negative_kappas])}")
    print(f"  Positive: {' '.join([f'{k:+.1f}' for k in positive_kappas])}")
    print(f"\nTotal models: {len(all_kappas) + 1} (including base)")
    print(f"Expected speedup: ~{len(all_kappas)}x vs sequential")
    print(f"{'='*80}\n")

    # Step 1: Check if base model exists, or train it
    base_path = os.path.join(REPO_ROOT, 'data', 'weights', base_name, f'{base_name}.pkl')
    
    print("\nSTEP 1: Checking for base model (κ=0.0)")
    print("-" * 80)
    
    if os.path.exists(base_path):
        print(f"✓ Base model already exists: {base_path}")
        print("  Skipping base model training.")
    else:
        print(f"✗ Base model not found at: {base_path}")
        print("  Training base model now...")
        
        base_cmd = [
            'python3', TRAIN_SCRIPT, modelfile, 'train'
        ]

        if not run_command(base_cmd, f"Training base model"):
            print("Failed to train base model. Exiting.")
            sys.exit(1)

        print(f"\n✓ Base model trained successfully")
        print(f"  Saved to: {base_path}")

    # Step 2: Train all kappas in parallel
    print("\n" + "="*80)
    print(f"STEP 2: Training ALL {len(all_kappas)} models IN PARALLEL from base")
    print("="*80)
    print(f"\nUsing {n_processes} parallel processes...")
    print("(Output from all processes will be interleaved)\n")

    start_time = time.time()

    # Prepare arguments for each kappa value (NO Queue!)
    train_args = [
        (kappa, modelfile, base_name, base_path, finetune_iter)
        for kappa in all_kappas
    ]

    # Train all kappas in parallel using Pool
    with Pool(processes=n_processes) as pool:
        results = pool.map(train_single_kappa, train_args)

    elapsed_time = time.time() - start_time

    # Combine results
    all_results = [("0.0", "", base_path, "✓ BASE")] + list(results)

    # Sort results by kappa value
    all_results.sort(key=lambda x: float(x[0]))

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nParallel training time: {elapsed_time/60:.1f} minutes")

    print("\nAll models (sorted by κ):")
    for kappa, suffix, path, status in all_results:
        if "✓" in status:
            display_path = path.replace(os.path.join(REPO_ROOT, 'data', 'weights', base_name, ''), "") if path else ""
            if status == "✓ BASE":
                print(f"  {status:12s} κ={kappa:>5s}  →  {display_path}")
            else:
                print(f"  {status:12s} κ={float(kappa):+.1f}  →  {display_path}")
        else:
            print(f"  {status:12s} κ={float(kappa):+.1f}")

    successful = [r for r in all_results if "✓" in r[3]]
    failed = [r for r in all_results if "✗" in r[3]]

    print(f"\n{'='*80}")
    print(f"Total successful: {len(successful)}/{len(all_results)}")
    print(f"Total failed: {len(failed)}")

    if failed:
        print(f"\n⚠️  Some models failed to train:")
        for kappa, suffix, _, _ in failed:
            print(f"  - κ={float(kappa):+.1f}")
    else:
        print(f"\n🎉 All {len(all_results)} models trained successfully!")

    print(f"\n⏱️  Total time: {elapsed_time/60:.1f} minutes")
    estimated_sequential = elapsed_time * len(all_kappas) / n_processes
    print(f"⚡ Speedup: ~{estimated_sequential/elapsed_time:.1f}x vs sequential")
    print(f"💾 All models saved in: {os.path.join(REPO_ROOT, 'data', 'weights', base_name)}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
