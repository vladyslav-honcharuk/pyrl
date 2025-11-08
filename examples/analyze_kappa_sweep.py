#!/usr/bin/env python3
"""
Run comprehensive analysis on all kappa sweep models.

This script analyzes all 21 models from the kappa sweep training:
- Generates trial data (trials-a and trials-b)
- Creates behavioral heatmaps
- Analyzes value network neurons
- Plots temporal activity

Usage:
    python3 analyze_kappa_sweep.py models/gambling.py
    python3 analyze_kappa_sweep.py models/gambling.py --parallel  # Run in parallel (faster)
"""

import os
import sys
import subprocess
import numpy as np
from multiprocessing import Process, Queue
import time

def kappa_to_suffix(kappa):
    """Convert kappa value to filename suffix."""
    if kappa == 0:
        return ""
    elif kappa < 0:
        return f"neg{abs(kappa):.1f}".replace('.', 'p')
    else:
        return f"pos{kappa:.1f}".replace('.', 'p')

def run_single_analysis(kappa, suffix, modelfile, base_name, analysis_action, args_list, log_queue=None):
    """Run a single analysis command for a model."""
    # Build command with suffix if needed
    cmd = ['python3', 'do.py', modelfile]
    if suffix:
        cmd.extend(['--suffix', suffix])
    cmd.extend(['run', 'analysis/gambling.py', analysis_action] + args_list)

    msg = f"  [{analysis_action}] Running..."
    if log_queue:
        log_queue.put(msg)
    else:
        print(msg)

    result = subprocess.run(cmd, cwd=os.path.dirname(__file__),
                          capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = f"  [{analysis_action}] ✗ FAILED\n{result.stderr}"
        if log_queue:
            log_queue.put(error_msg)
        else:
            print(error_msg)
        return False

    success_msg = f"  [{analysis_action}] ✓ Complete"
    if log_queue:
        log_queue.put(success_msg)
    else:
        print(success_msg)

    return True

def run_analysis_for_model(kappa, suffix, modelfile, base_name, log_queue=None):
    """Run all analysis steps for a single model."""
    model_name = f"{base_name}{suffix}" if suffix else base_name
    model_path = f"work/data/{model_name}/{model_name}.pkl"

    msg = f"\n{'='*80}\nAnalyzing κ={kappa:+.1f} ({model_name})\n{'='*80}"
    if log_queue:
        log_queue.put(msg)
    else:
        print(msg)

    # Check if model exists
    if not os.path.exists(model_path):
        error_msg = f"✗ Model not found: {model_path}"
        if log_queue:
            log_queue.put(error_msg)
        else:
            print(error_msg)
        return False

    # Run each analysis step separately (the do() function only handles one action at a time)
    all_success = True

    # Step 1: Generate trials-a with activity data
    if not run_single_analysis(kappa, suffix, modelfile, base_name,
                               'trials-a', ['2'], log_queue):
        all_success = False

    # Step 2: Generate trials-b with activity data
    if not run_single_analysis(kappa, suffix, modelfile, base_name,
                               'trials-b', ['2'], log_queue):
        all_success = False

    # Step 3: Behavioral heatmap
    if not run_single_analysis(kappa, suffix, modelfile, base_name,
                               'behavior', [], log_queue):
        all_success = False

    # Step 4: Value neurons analysis
    if not run_single_analysis(kappa, suffix, modelfile, base_name,
                               'value-neurons', [str(kappa)], log_queue):
        all_success = False

    # Step 5: Temporal activity plots
    if not run_single_analysis(kappa, suffix, modelfile, base_name,
                               'temporal-activity', ['value', '3', str(kappa)], log_queue):
        all_success = False

    if all_success:
        success_msg = f"✓ All analyses complete for κ={kappa:+.1f}"
    else:
        success_msg = f"⚠️  Some analyses failed for κ={kappa:+.1f}"

    if log_queue:
        log_queue.put(success_msg)
    else:
        print(success_msg)

    return all_success

def analyze_sequential(kappa_values, modelfile, base_name):
    """Run analysis sequentially on all models."""
    print(f"\n{'='*80}")
    print("SEQUENTIAL ANALYSIS")
    print(f"{'='*80}\n")

    start_time = time.time()
    results = []

    for kappa in kappa_values:
        suffix = kappa_to_suffix(kappa)
        success = run_analysis_for_model(kappa, suffix, modelfile, base_name)
        results.append((kappa, suffix, success))

    elapsed_time = time.time() - start_time
    return results, elapsed_time

def analyze_parallel_worker(kappa, suffix, modelfile, base_name, result_queue, log_queue):
    """Worker function for parallel analysis."""
    success = run_analysis_for_model(kappa, suffix, modelfile, base_name, log_queue)
    result_queue.put((kappa, suffix, success))

def log_printer(log_queue, stop_signal):
    """Process that prints logs from multiple processes."""
    while True:
        msg = log_queue.get()
        if msg == stop_signal:
            break
        print(msg, flush=True)

def analyze_parallel(kappa_values, modelfile, base_name, max_parallel=4):
    """Run analysis in parallel on multiple models."""
    print(f"\n{'='*80}")
    print(f"PARALLEL ANALYSIS (max {max_parallel} concurrent)")
    print(f"{'='*80}\n")

    start_time = time.time()
    results = []

    # Setup communication queues
    result_queue = Queue()
    log_queue = Queue()
    stop_signal = "STOP_LOGGING"

    # Start log printer
    log_process = Process(target=log_printer, args=(log_queue, stop_signal))
    log_process.start()

    # Process models in batches
    i = 0
    while i < len(kappa_values):
        # Start a batch of processes
        processes = []
        batch_size = min(max_parallel, len(kappa_values) - i)

        for j in range(batch_size):
            kappa = kappa_values[i + j]
            suffix = kappa_to_suffix(kappa)

            p = Process(
                target=analyze_parallel_worker,
                args=(kappa, suffix, modelfile, base_name, result_queue, log_queue)
            )
            p.start()
            processes.append(p)

        # Wait for batch to complete
        for p in processes:
            p.join()

        # Collect results from this batch
        for _ in range(batch_size):
            results.append(result_queue.get())

        i += batch_size

    # Stop log printer
    log_queue.put(stop_signal)
    log_process.join()

    elapsed_time = time.time() - start_time
    return results, elapsed_time

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_kappa_sweep.py <model_file> [--parallel] [--max-parallel N]")
        print("Example: python3 analyze_kappa_sweep.py models/gambling.py")
        print("Example: python3 analyze_kappa_sweep.py models/gambling.py --parallel --max-parallel 4")
        sys.exit(1)

    modelfile = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(modelfile))[0]

    # Parse options
    parallel = '--parallel' in sys.argv
    max_parallel = 4  # Default

    if '--max-parallel' in sys.argv:
        try:
            idx = sys.argv.index('--max-parallel')
            max_parallel = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Error: --max-parallel requires an integer argument")
            sys.exit(1)

    # All kappa values (sorted)
    kappa_step = 0.1
    negative_kappas = [round(k, 1) for k in np.arange(-1.0, -0.0, kappa_step)]
    positive_kappas = [0.0] + [round(k, 1) for k in np.arange(0.1, 1.0 + kappa_step/2, kappa_step)]
    all_kappas = sorted(negative_kappas + positive_kappas)

    print(f"{'='*80}")
    print(f"KAPPA SWEEP ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {modelfile}")
    print(f"Total models: {len(all_kappas)}")
    print(f"Kappa values: {', '.join([f'{k:+.1f}' for k in all_kappas])}")
    print(f"\nAnalysis steps for each model:")
    print(f"  1. Generate trials-a (2 per condition)")
    print(f"  2. Generate trials-b (2 per condition)")
    print(f"  3. Behavioral heatmap")
    print(f"  4. Value network neuron analysis")
    print(f"  5. Temporal activity plots")

    if parallel:
        print(f"\nMode: PARALLEL (max {max_parallel} concurrent)")
    else:
        print(f"\nMode: SEQUENTIAL")

    print(f"{'='*80}\n")

    # Run analysis
    if parallel:
        results, elapsed_time = analyze_parallel(all_kappas, modelfile, base_name, max_parallel)
    else:
        results, elapsed_time = analyze_sequential(all_kappas, modelfile, base_name)

    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal time: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")

    # Sort results by kappa
    results.sort(key=lambda x: x[0])

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    print(f"\nResults by κ:")
    for kappa, suffix, success in results:
        status = "✓" if success else "✗ FAILED"
        model_name = f"{base_name}{suffix}" if suffix else base_name
        print(f"  {status}  κ={kappa:+.1f}  ({model_name})")

    print(f"\n{'='*80}")
    print(f"Total successful: {len(successful)}/{len(all_kappas)}")
    print(f"Total failed: {len(failed)}")

    if failed:
        print(f"\n⚠️  Some analyses failed:")
        for kappa, suffix, _ in failed:
            print(f"  - κ={kappa:+.1f}")
    else:
        print(f"\n🎉 All {len(all_kappas)} analyses completed successfully!")

    print(f"\n💾 Results saved in: work/figs/{base_name}*/")
    print(f"{'='*80}\n")

    # Exit with error code if any failed
    sys.exit(0 if len(failed) == 0 else 1)

if __name__ == '__main__':
    main()
