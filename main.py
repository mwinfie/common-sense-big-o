#!/usr/bin/env python3
# Based on information in A Common-Sense Guide to Data Structures and Algorithms, Second Edition by Jay Wengrow.
# This script runs empirical tests against functions to calculate their time and space complexity in terms of Big O.
# Test results are plotted on a graph along with the theoretical complexity to show how well the two align.
# All functions tested in this repo come straight from the book.
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from multiprocessing import Pool
from benchmarks.chapter_01 import PrintNumbersVersionOneBenchmark, PrintNumbersVersionTwoBenchmark, LinearSearch
from benchmarks.chapter_02 import BinarySearch
from benchmarks.chapter_04 import BubbleSort

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
    # List complexity sizes for benchmarks
    n_sizes = np.logspace(4, 8, 50, dtype=int)

    # List tests to run
    tests = [
            # PrintNumbersVersionOneBenchmark(),
            # PrintNumbersVersionTwoBenchmark(),
            # LinearSearch(),
            # BubbleSort(),
            BinarySearch()
        ]

    # Initialize graph based on number of tests to run
    fig, axs = plt.subplots(len(tests), 2, figsize=(12, 5 * len(tests)), squeeze=False)
    fig.suptitle('Theoretical vs. Empirical Performance')

    for i, test in enumerate(tests):
        print(f"Benchmarking {test.name}...")
        
        y_time = []
        y_space = []

        # Run Benchmarks in parallel
        with Pool() as pool:
            results = pool.starmap(run_benchmark, [(test, n) for n in n_sizes])
        
        for t, s in results:
            y_time.append(t)
            y_space.append(s)

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()