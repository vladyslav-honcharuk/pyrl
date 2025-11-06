#!/usr/bin/env python3
"""
Train a model with progressive kappa values IN PARALLEL.

This script:
1. Trains base model with kappa=0 for 1000 iterations
2. Runs TWO chains in PARALLEL:
   - Positive chain: 0 → 0.1 → 0.2 → ... → 1.0
   - Negative chain: 0 → -0.1 → -0.2 → ... → -1.0

Each step trains for 1000 iterations. The two chains run simultaneously
to save time (roughly 2x faster than sequential).

Usage:
    python3 train_kappa_sweep_parallel.py models/gambling.py
"""

import os
import sys
import subprocess
import numpy as np
from multiprocessing import Process, Queue
import time

def run_command(cmd, description, log_queue=None):
    """Run a shell command and print output."""
    msg = f"\n{'='*80}\n{description}\n{'='*80}\nRunning: {' '.join(cmd)}\n"
    if log_queue:
        log_queue.put(msg)
    else:
        print(msg)

    # Don't capture output if we have a log_queue - let subprocess print directly
    # This way we see real-time output from training
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))

    if result.returncode != 0:
        error_msg = f"\nError: Command failed with return code {result.returncode}"
        if log_queue:
            log_queue.put(error_msg)
        else:
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

def train_chain(chain_name, kappa_values, modelfile, base_name, finetune_iter, result_queue, log_queue):
    """Train a chain of kappa values (runs in separate process)."""
    log_queue.put(f"\n🚀 Starting {chain_name} chain in parallel process...")

    results = []
    prev_kappa = 0.0
    prev_suffix = ""

    for i, kappa in enumerate(kappa_values, 1):
        suffix = kappa_to_suffix(kappa)

        log_queue.put(f"\n[{chain_name} {i}/{len(kappa_values)}] κ={prev_kappa:.1f} → κ={kappa:.1f}")
        log_queue.put("-" * 80)

        # Construct pretrained path (note: no underscores between base_name and suffix!)
        if prev_suffix:
            prev_name = f"{base_name}{prev_suffix}"
            pretrained_path = f"work/data/{prev_name}/{prev_name}.pkl"
        else:
            pretrained_path = f"work/data/{base_name}/{base_name}.pkl"

        finetune_cmd = [
            'python3', 'do.py', modelfile, 'finetune',
            '--kappa', str(kappa),
            '--suffix', suffix,
            '--pretrained', pretrained_path,
            '--finetune-iter', str(finetune_iter),
            '--finetune-lr', '0.001'  # Lower LR for fine-tuning (0.001 vs 0.004 for base)
        ]

        success = run_command(
            finetune_cmd,
            f"{chain_name}: Fine-tuning κ={prev_kappa:.1f} → κ={kappa:.1f}",
            log_queue
        )

        if success:
            save_name = f"{base_name}{suffix}"
            save_path = f"work/data/{save_name}/{save_name}.pkl"
            results.append((f"{kappa:+.1f}", suffix, save_path, "✓"))
            log_queue.put(f"✓ {chain_name}: Successfully trained κ={kappa:.1f}")
            prev_kappa = kappa
            prev_suffix = suffix
        else:
            results.append((f"{kappa:+.1f}", suffix, "", "✗ FAILED"))
            log_queue.put(f"✗ {chain_name}: Failed at κ={kappa:.1f}. Stopping chain.")
            break

    # Send results back to main process
    result_queue.put((chain_name, results))
    log_queue.put(f"\n✅ {chain_name} chain completed!")

def log_printer(log_queue, stop_signal):
    """Process that prints logs from multiple processes in order."""
    while True:
        msg = log_queue.get()
        if msg == stop_signal:
            break
        print(msg, flush=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 train_kappa_sweep_parallel.py <model_file>")
        print("Example: python3 train_kappa_sweep_parallel.py models/gambling.py")
        sys.exit(1)

    modelfile = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(modelfile))[0]

    # Parameters
    base_iter = 1000
    finetune_iter = 1000
    kappa_step = 0.1

    # Generate chains
    positive_chain = [round(k, 1) for k in np.arange(0.1, 1.0 + kappa_step/2, kappa_step)]
    negative_chain = [round(k, 1) for k in np.arange(-0.1, -1.0 - kappa_step/2, -kappa_step)]

    print(f"{'='*80}")
    print(f"KAPPA SWEEP TRAINING (PARALLEL)")
    print(f"{'='*80}")
    print(f"Model: {modelfile}")
    print(f"Base model iterations (κ=0): {base_iter}")
    print(f"Fine-tune iterations per κ: {finetune_iter}")
    print(f"\nTraining strategy:")
    print(f"  Step 1: Train base (κ=0.0)")
    print(f"  Step 2: Train BOTH chains in PARALLEL:")
    print(f"    • Positive: {' → '.join([f'{k:.1f}' for k in positive_chain])}")
    print(f"    • Negative: {' → '.join([f'{k:.1f}' for k in negative_chain])}")
    print(f"\nTotal models: 21")
    print(f"Expected speedup: ~2x vs sequential")
    print(f"{'='*80}\n")

    # Step 1: Train base model
    print("\nSTEP 1: Training base model (κ=0.0)")
    print("-" * 80)
    print(f"NOTE: Base model will train using max_iter from model config")

    base_cmd = [
        'python3', 'do.py', modelfile, 'train'
    ]

    if not run_command(base_cmd, f"Training base model"):
        print("Failed to train base model. Exiting.")
        sys.exit(1)

    print(f"\n✓ Base model trained successfully")
    base_path = f"work/data/{base_name}/{base_name}.pkl"
    print(f"  Saved to: {base_path}")

    # Step 2: Train both chains in parallel
    print("\n" + "="*80)
    print("STEP 2: Training BOTH chains IN PARALLEL")
    print("="*80)
    print("\nStarting parallel processes...")
    print("(Logs from both chains will be interleaved)\n")

    # Queues for inter-process communication
    result_queue = Queue()
    log_queue = Queue()

    # Start log printer process
    stop_signal = "STOP_LOGGING"
    log_process = Process(target=log_printer, args=(log_queue, stop_signal))
    log_process.start()

    # Start both training chains
    start_time = time.time()

    pos_process = Process(
        target=train_chain,
        args=("POSITIVE", positive_chain, modelfile, base_name, finetune_iter, result_queue, log_queue)
    )

    neg_process = Process(
        target=train_chain,
        args=("NEGATIVE", negative_chain, modelfile, base_name, finetune_iter, result_queue, log_queue)
    )

    pos_process.start()
    neg_process.start()

    # Wait for both to complete
    pos_process.join()
    neg_process.join()

    elapsed_time = time.time() - start_time

    # Stop log printer
    log_queue.put(stop_signal)
    log_process.join()

    # Collect results
    all_results = [("0.0", "", base_path, "✓ BASE")]

    for _ in range(2):  # Get results from both processes
        chain_name, results = result_queue.get()
        all_results.extend(results)

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
            display_path = path.replace(f"work/data/{base_name}/", "")
            print(f"  {status:12s} κ={kappa:>5s}  →  {display_path}")
        else:
            print(f"  {status:12s} κ={kappa:>5s}")

    successful = [r for r in all_results if "✓" in r[3]]
    failed = [r for r in all_results if "✗" in r[3]]

    print(f"\n{'='*80}")
    print(f"Total successful: {len(successful)}/21")
    print(f"Total failed: {len(failed)}")

    if failed:
        print(f"\n⚠️  Some models failed to train:")
        for kappa, suffix, _, _ in failed:
            print(f"  - κ={kappa}")
    else:
        print("\n🎉 All 21 models trained successfully!")

    print(f"\n⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print(f"💾 All models saved in: work/data/{base_name}/")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
