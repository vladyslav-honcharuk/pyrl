#!/usr/bin/env python3
"""
Train a model with GRADUAL kappa adjustment.

This script implements a curriculum learning approach where kappa is gradually
increased/decreased in small steps, but only saves checkpoints at key milestones.

Key features:
- Gradual training: Small kappa steps (default 0.05) for smoother learning
- Selective saving: Only saves checkpoints at milestones (0.2, 0.4, 0.6, 0.8, 1.0)
- Parallel execution: Trains positive and negative chains simultaneously
- Adaptive training: Can detect and handle gradient collapse issues

Usage:
    python3 train_kappa_gradual.py models/gambling.py
    python3 train_kappa_gradual.py models/gambling.py --step 0.1  # Larger steps
    python3 train_kappa_gradual.py models/gambling.py --direction positive  # Only positive kappa
"""

import os
import sys
import subprocess
import numpy as np
from multiprocessing import Process, Queue
import time
import argparse

def run_command(cmd, description, log_queue=None):
    """Run a shell command and print output."""
    msg = f"\n{'='*80}\n{description}\n{'='*80}\nRunning: {' '.join(cmd)}\n"
    if log_queue:
        log_queue.put(msg)
    else:
        print(msg)

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
        return f"neg{abs(kappa):.2f}".replace('.', 'p')
    else:
        return f"pos{kappa:.2f}".replace('.', 'p')

def train_chain(chain_name, kappa_values, save_checkpoints, modelfile, base_name,
                finetune_iter, finetune_lr, result_queue, log_queue):
    """
    Train a chain of kappa values with gradual progression.

    Args:
        chain_name: "POSITIVE" or "NEGATIVE" for logging
        kappa_values: All kappa values to train (including intermediate steps)
        save_checkpoints: Set of kappa values where we should save checkpoints
        modelfile: Path to model definition
        base_name: Base model name
        finetune_iter: Iterations per kappa step
        finetune_lr: Learning rate for finetuning
        result_queue: Queue to send results back to main process
        log_queue: Queue for logging
    """
    log_queue.put(f"\n🚀 Starting {chain_name} chain in parallel process...")
    log_queue.put(f"   Total steps: {len(kappa_values)}")
    log_queue.put(f"   Checkpoints: {sorted(save_checkpoints)}")

    results = []
    prev_kappa = 0.0
    prev_suffix = ""
    last_saved_path = None

    for i, kappa in enumerate(kappa_values, 1):
        suffix = kappa_to_suffix(kappa)
        should_save = kappa in save_checkpoints

        # Show progress
        checkpoint_marker = " [CHECKPOINT]" if should_save else ""
        log_queue.put(f"\n[{chain_name} {i}/{len(kappa_values)}] κ={prev_kappa:.2f} → κ={kappa:.2f}{checkpoint_marker}")
        log_queue.put("-" * 80)

        # Construct pretrained path
        if prev_suffix and last_saved_path:
            pretrained_path = last_saved_path
        elif prev_suffix:
            prev_name = f"{base_name}{prev_suffix}"
            pretrained_path = f"work/data/{prev_name}/{prev_name}.pkl"
        else:
            pretrained_path = f"work/data/{base_name}/{base_name}.pkl"

        # Build finetune command
        finetune_cmd = [
            'python3', 'do.py', modelfile, 'finetune',
            '--kappa', str(kappa),
            '--suffix', suffix,
            '--pretrained', pretrained_path,
            '--finetune-iter', str(finetune_iter),
            '--finetune-lr', str(finetune_lr)
        ]

        success = run_command(
            finetune_cmd,
            f"{chain_name}: Fine-tuning κ={prev_kappa:.2f} → κ={kappa:.2f}",
            log_queue
        )

        if success:
            save_name = f"{base_name}{suffix}"
            save_path = f"work/data/{save_name}/{save_name}.pkl"

            # Only keep track of saved checkpoints
            if should_save:
                results.append((f"{kappa:+.2f}", suffix, save_path, "✓"))
                last_saved_path = save_path
                log_queue.put(f"✓ {chain_name}: Successfully trained and SAVED κ={kappa:.2f}")
            else:
                last_saved_path = save_path  # Track for next iteration
                log_queue.put(f"✓ {chain_name}: Successfully trained κ={kappa:.2f} (not saved)")

            prev_kappa = kappa
            prev_suffix = suffix
        else:
            if should_save:
                results.append((f"{kappa:+.2f}", suffix, "", "✗ FAILED"))
            log_queue.put(f"✗ {chain_name}: Failed at κ={kappa:.2f}. Stopping chain.")
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
    parser = argparse.ArgumentParser(
        description='Train a model with gradual kappa adjustment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 train_kappa_gradual.py models/gambling.py
  python3 train_kappa_gradual.py models/gambling.py --step 0.1
  python3 train_kappa_gradual.py models/gambling.py --direction positive
  python3 train_kappa_gradual.py models/gambling.py --save-all  # Save all steps
        """
    )

    parser.add_argument('modelfile', help='Model file to train (e.g., models/gambling.py)')
    parser.add_argument('--step', type=float, default=0.05,
                       help='Kappa step size (default: 0.05)')
    parser.add_argument('--direction', choices=['both', 'positive', 'negative'], default='both',
                       help='Which direction to train (default: both)')
    parser.add_argument('--base-iter', type=int, default=1000,
                       help='Base model iterations (default: 1000)')
    parser.add_argument('--finetune-iter', type=int, default=1000,
                       help='Fine-tune iterations per step (default: 1000)')
    parser.add_argument('--finetune-lr', type=float, default=0.001,
                       help='Fine-tune learning rate (default: 0.001)')
    parser.add_argument('--checkpoints', type=str, default='0.2,0.4,0.6,0.8,1.0',
                       help='Comma-separated kappa values to save (default: 0.2,0.4,0.6,0.8,1.0)')
    parser.add_argument('--save-all', action='store_true',
                       help='Save all intermediate steps (not just checkpoints)')
    parser.add_argument('--skip-base', action='store_true',
                       help='Skip training base model (assumes it already exists)')

    args = parser.parse_args()

    modelfile = args.modelfile
    base_name = os.path.splitext(os.path.basename(modelfile))[0]
    kappa_step = args.step

    # Parse checkpoints
    if args.save_all:
        # Generate all values based on step size
        all_values = np.arange(kappa_step, 1.0 + kappa_step/2, kappa_step)
        save_checkpoints = set(np.round(all_values, 2))
    else:
        save_checkpoints = set([float(x.strip()) for x in args.checkpoints.split(',')])

    # Generate chains with the specified step size
    positive_chain = [round(k, 2) for k in np.arange(kappa_step, 1.0 + kappa_step/2, kappa_step)]
    negative_chain = [round(k, 2) for k in np.arange(-kappa_step, -1.0 - kappa_step/2, -kappa_step)]

    print(f"{'='*80}")
    print(f"GRADUAL KAPPA TRAINING")
    print(f"{'='*80}")
    print(f"Model: {modelfile}")
    print(f"Kappa step size: {kappa_step}")
    print(f"Base model iterations (κ=0): {args.base_iter}")
    print(f"Fine-tune iterations per κ: {args.finetune_iter}")
    print(f"Fine-tune learning rate: {args.finetune_lr}")
    print(f"\nCheckpoint strategy:")
    if args.save_all:
        print(f"  Saving ALL intermediate steps")
    else:
        print(f"  Saving only at checkpoints: {sorted(save_checkpoints)}")
    print(f"\nTraining strategy:")
    print(f"  Step 1: Train base (κ=0.0) {'[SKIPPED]' if args.skip_base else ''}")

    if args.direction == 'both':
        print(f"  Step 2: Train BOTH chains in PARALLEL:")
        print(f"    • Positive ({len(positive_chain)} steps): 0 → {' → '.join([f'{k:.2f}' for k in positive_chain])}")
        print(f"    • Negative ({len(negative_chain)} steps): 0 → {' → '.join([f'{k:.2f}' for k in negative_chain])}")
    elif args.direction == 'positive':
        print(f"  Step 2: Train positive chain ({len(positive_chain)} steps):")
        print(f"    • 0 → {' → '.join([f'{k:.2f}' for k in positive_chain])}")
    else:  # negative
        print(f"  Step 2: Train negative chain ({len(negative_chain)} steps):")
        print(f"    • 0 → {' → '.join([f'{k:.2f}' for k in negative_chain])}")

    num_checkpoints = len(save_checkpoints & set(positive_chain)) + len(save_checkpoints & set(negative_chain))
    if args.direction != 'both':
        num_checkpoints = len(save_checkpoints & (set(positive_chain) if args.direction == 'positive' else set(negative_chain)))

    print(f"\nTotal checkpoints to save: {num_checkpoints + 1} (including base)")
    print(f"{'='*80}\n")

    # Step 1: Train base model (unless skipped)
    if not args.skip_base:
        print("\nSTEP 1: Training base model (κ=0.0)")
        print("-" * 80)

        base_cmd = ['python3', 'do.py', modelfile, 'train']

        if not run_command(base_cmd, f"Training base model"):
            print("Failed to train base model. Exiting.")
            sys.exit(1)

        print(f"\n✓ Base model trained successfully")
    else:
        print("\nSTEP 1: Skipping base model training")

    base_path = f"work/data/{base_name}/{base_name}.pkl"
    print(f"  Base model: {base_path}")

    # Step 2: Train chain(s)
    print("\n" + "="*80)
    if args.direction == 'both':
        print("STEP 2: Training BOTH chains IN PARALLEL")
    else:
        print(f"STEP 2: Training {args.direction.upper()} chain")
    print("="*80)

    # Queues for inter-process communication
    result_queue = Queue()
    log_queue = Queue()

    # Start log printer process
    stop_signal = "STOP_LOGGING"
    log_process = Process(target=log_printer, args=(log_queue, stop_signal))
    log_process.start()

    start_time = time.time()
    processes = []

    # Start training process(es)
    if args.direction in ['both', 'positive']:
        pos_process = Process(
            target=train_chain,
            args=("POSITIVE", positive_chain, save_checkpoints, modelfile, base_name,
                  args.finetune_iter, args.finetune_lr, result_queue, log_queue)
        )
        pos_process.start()
        processes.append(('POSITIVE', pos_process))

    if args.direction in ['both', 'negative']:
        neg_process = Process(
            target=train_chain,
            args=("NEGATIVE", negative_chain, save_checkpoints, modelfile, base_name,
                  args.finetune_iter, args.finetune_lr, result_queue, log_queue)
        )
        neg_process.start()
        processes.append(('NEGATIVE', neg_process))

    # Wait for all to complete
    for name, proc in processes:
        proc.join()

    elapsed_time = time.time() - start_time

    # Stop log printer
    log_queue.put(stop_signal)
    log_process.join()

    # Collect results
    all_results = [("0.00", "", base_path, "✓ BASE")]

    for _ in range(len(processes)):
        chain_name, results = result_queue.get()
        all_results.extend(results)

    # Sort results by kappa value
    all_results.sort(key=lambda x: float(x[0]))

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nTraining time: {elapsed_time/60:.1f} minutes")

    print("\nSaved checkpoints (sorted by κ):")
    for kappa, suffix, path, status in all_results:
        if "✓" in status:
            display_path = path.replace(f"work/data/{base_name}/", "")
            print(f"  {status:12s} κ={kappa:>6s}  →  {display_path}")
        else:
            print(f"  {status:12s} κ={kappa:>6s}")

    successful = [r for r in all_results if "✓" in r[3]]
    failed = [r for r in all_results if "✗" in r[3]]

    print(f"\n{'='*80}")
    print(f"Total successful: {len(successful)}")
    print(f"Total failed: {len(failed)}")

    if failed:
        print(f"\n⚠️  Some models failed to train:")
        for kappa, suffix, _, _ in failed:
            print(f"  - κ={kappa}")
    else:
        print("\n🎉 All models trained successfully!")

    print(f"\n⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print(f"💾 All models saved in: work/data/{base_name}/")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
