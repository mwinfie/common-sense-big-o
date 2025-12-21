#!/usr/bin/env python3
# Based on information in A Common-Sense Guide to Data Structures and Algorithms, Second Edition by Jay Wengrow.
# This script runs empirical tests against functions to calculate their time and space complexity in terms of Big O.
# Test results are plotted on a graph along with the theoretical complexity to show how well the two align.
# All functions tested in this repo come straight from the book.
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc

def get_benchmarks(func, *args, **kwargs):
    """
    Returns a tuple of (execution_time, peak_memory).
    Executes the tested function TWICE to ensure measurements don't interfere with each other.
    """
    # Test only Time (High precision, no memory tracking overhead)
    start = time.perf_counter()
    func(*args, **kwargs)
    duration = time.perf_counter() - start
    
    # Test only Memory (Track allocations, ignoring the slower execution time)
    tracemalloc.start()
    func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return duration, peak

def print_numbers_version_one(num):
    number = 2

    while number <= num:
        # If number is even, print it:
        if number % 2 == 0:
            #print(number)
            pass
        
        number += 1

def print_numbers_version_two(num):
    number = 2

    while number <= num:
        #print(number)

        # Increase number by 2, which, by definition is the next even number:
        number += 2

def main():
    # List complexity sizes to measure and init necessary variables
    n_sizes = np.logspace(5, 6, 7, dtype=int)
    i = 0

    # List functions to test
    functions = [
            { "name": print_numbers_version_one},
            { "name": print_numbers_version_two}
        ]

    # Initialize graph based on number of functions to test
    fig, axs = plt.subplots( len(functions), 2, figsize=(10, 4))
    fig.suptitle('Theoretical vs. Empirical Performance')

    for function in functions:

        name = function["name"].__name__
    
        x = np.array(n_sizes)
        y_time = np.array([])
        y_space = np.array([])

        for n in n_sizes:
            time, space = get_benchmarks(function["name"], n)
            y_time = np.append(y_time, time)
            y_space = np.append(y_space, space)

        # Draw theoretical trend line with scatterplot of test results for Time and Space
        axs[i, 0].scatter(x, y_time, label='O(N)')
        axs[i, 0].set_title('Function {}: Time'.format( name ))
        axs[i, 0].set_xlabel('N')
        axs[i, 0].set_ylabel('T(N)')
        z = np.polyfit(x, y_time, 1)
        p = np.poly1d(z)
        axs[i, 0].plot(x, p(x), color="red", linestyle="--", label='Trendline')

        axs[i, 1].scatter(x, y_space, label='O(N)')
        axs[i, 1].set_title('Function {}: Space'.format( name ))
        axs[i, 1].set_xlabel('N')
        axs[i, 1].set_ylabel('S(N)')
        z = np.polyfit(x, y_space, 1)
        p = np.poly1d(z)
        axs[i, 1].plot(x, p(x), color="red", linestyle="--", label='Trendline')

        i += 1
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

if __name__ == "__main__":
    main()