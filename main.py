#!/usr/bin/env python3

"""
Based on information in A Common-Sense Guide to Data Structures and Algorithms, Second Edition by Jay Wengrow.
This script runs empirical tests against functions to calculate their time and space complexity in terms of Big O.
Test results are plotted on a graph along with the theoretical complexity to show how well the two align.
All functions tested in this repo come straight from the book.
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
from benchmarks.chapter_01 import LinearSearch, PrintNumbersVersionOneBenchmark, PrintNumbersVersionTwoBenchmark
from benchmarks.chapter_02 import BinarySearch
from benchmarks.chapter_04 import BubbleSort
from benchmarks.chapter_05 import SelectionSort
from benchmarks.io import load_results, plot_results, save_results
from datetime import datetime
from decouple import Csv, config
from multiprocessing import Pool


def run_benchmark(benchmark_obj, n):
    # Test only Time (High precision, no memory tracking overhead)
    benchmark_obj.setup(n)

    start_time = time.perf_counter()
    benchmark_obj.run_algorithm()
    duration = time.perf_counter() - start_time

    # Test only Memory (Track allocations, ignoring the slower execution time)
    benchmark_obj.setup(n)

    tracemalloc.start()
    benchmark_obj.run_algorithm()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return duration, peak_memory


def main():
    # Check if loading from file
    load_file = config('LOAD_RESULTS', default='', cast=str)
    output_format = config('OUTPUT_FORMAT', default='window', cast=str).lower()

    if load_file:
        print(f"Loading results from: {load_file}")
        results_data = load_results(load_file)
        n_sizes = np.array(results_data['n_sizes'])
        plot_results(results_data, n_sizes, output_format)
        return

    # List complexity sizes for benchmarks
    n_sizes = np.logspace(2, 5, 20, dtype=int)

    # Map of available benchmark names to their classes
    AVAILABLE_BENCHMARKS = {
        'PrintNumbersVersionOneBenchmark': PrintNumbersVersionOneBenchmark,
        'PrintNumbersVersionTwoBenchmark': PrintNumbersVersionTwoBenchmark,
        'LinearSearch': LinearSearch,
        'BubbleSort': BubbleSort,
        'BinarySearch': BinarySearch,
        'SelectionSort': SelectionSort,
    }

    # Get list of benchmarks to run from environment variable (default: SelectionSort)
    benchmark_names = config('BENCHMARKS', default='SelectionSort', cast=Csv())
    tests = [AVAILABLE_BENCHMARKS[name.strip()]() for name in benchmark_names
             if name.strip() in AVAILABLE_BENCHMARKS] or [SelectionSort()]

    # Store all results for export
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'n_sizes': n_sizes.tolist(),
        'tests': []
    }

    # Initialize graph based on number of tests to run
    fig_height = 6 * len(tests) if len(tests) > 3 else 5 * len(tests)
    fig, axs = plt.subplots(len(tests), 2, figsize=(16, fig_height), squeeze=False)
    fig.suptitle('Theoretical vs. Empirical Performance')

    for i, test in enumerate(tests):
        print(f"Benchmarking {test.name}...")
        print(f"  Running {len(n_sizes)} test sizes from {n_sizes[0]:,} to {n_sizes[-1]:,}...")

        y_time = []
        y_space = []

        # Run Benchmarks in parallel
        with Pool() as pool:
            results = pool.starmap(run_benchmark, [(test, n) for n in n_sizes])

        for idx, (t, s) in enumerate(results, 1):
            y_time.append(t)
            y_space.append(s)
            print(f"  [{idx}/{len(n_sizes)}] n={n_sizes[idx-1]:,}: {t:.4f}s", flush=True)

        y_time = np.array(y_time)
        y_space = np.array(y_space)

        # Plot Time
        time_label, time_func = test.time_complexity
        theory_time_raw = np.array([time_func(n) for n in n_sizes])
        scale_t = np.max(y_time) / np.max(theory_time_raw) if np.max(theory_time_raw) > 0 else 1

        axs[i, 0].scatter(n_sizes, y_time, color='blue', label='Measured Time', alpha=0.6)
        axs[i, 0].plot(n_sizes, theory_time_raw * scale_t, color='red', linestyle='--', label=f'{time_label}')
        axs[i, 0].set_title(f"{test.name}: Time Complexity")
        axs[i, 0].set_xlabel("N")
        axs[i, 0].set_ylabel("Seconds")
        axs[i, 0].legend()
        axs[i, 0].grid(True, alpha=0.3)

        # Plot Space
        space_label, space_func = test.space_complexity
        theory_space_raw = np.array([space_func(n) for n in n_sizes])
        scale_s = np.max(y_space) / np.max(theory_space_raw) if np.max(theory_space_raw) > 0 else 1

        axs[i, 1].scatter(n_sizes, y_space, color='orange', label='Measured Space', alpha=0.6)
        axs[i, 1].plot(n_sizes, theory_space_raw * scale_s, color='green', linestyle='--', label=f'{space_label}')
        axs[i, 1].set_title(f"{test.name}: Space Complexity")
        axs[i, 1].set_xlabel("N")
        axs[i, 1].set_ylabel("Peak Bytes")
        axs[i, 1].legend()
        axs[i, 1].grid(True, alpha=0.3)

        # Store results for export
        results_data['tests'].append({
            'name': test.name,
            'time_complexity_label': time_label,
            'space_complexity_label': space_label,
            'time_measurements': y_time.tolist(),
            'space_measurements': y_space.tolist(),
            'theory_time': (theory_time_raw * scale_t).tolist(),
            'theory_space': (theory_space_raw * scale_s).tolist()
        })

    # Save results to file
    save_results(results_data)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_format == 'pdf':
        import subprocess
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = Path("results") / f"benchmark_results_{timestamp}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"\nPDF saved to: {pdf_path}")
        print("Opening PDF...")
        plt.close()
        subprocess.run(['open', pdf_path])  # macOS
    else:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
