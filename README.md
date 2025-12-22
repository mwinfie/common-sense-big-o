# Algorithm Performance Benchmarking Suite

A Python-based empirical testing framework for analyzing and visualizing the time and space complexity of classic data structures and algorithms. This project implements algorithms covered in [*A Common-Sense Guide to Data Structures and Algorithms* by Jay Wengrow](https://pragprog.com/titles/jwdsal2/a-common-sense-guide-to-data-structures-and-algorithms-second-edition/) and validates their theoretical Big O complexity through real-world performance measurements.

## Background

This project was developed as my capstone project for the [OK Coders](https://www.techlahoma.org/ok-coders/) Python Module 2 course covering content from *Automate the Boring Stuff with Python* by Al Sweigart. I built this project to compare theoretical complexity to observed performance. This project provides visual confirmation that algorithmic implementations behave according to their expected Big O classifications.

## Features

- **Automated Performance Testing**: Executes algorithms across varying input sizes to measure actual runtime and memory usage
- **Dual Complexity Analysis**: Separately tests time complexity (execution duration) and space complexity (peak memory allocation)
- **Visual Validation**: Generates side-by-side graphs comparing theoretical Big O curves against empirical measurements
- **Extensible Architecture**: Modular design allows easy addition of new algorithms through a common benchmark interface
- **Precision Instrumentation**: Uses `time.perf_counter()` for high-resolution timing and `tracemalloc` for memory profiling

## Project Structure

```
common-sense-big-o/
├── main.py                      # Entry point and benchmarking orchestration
├── algorithms/                  # Algorithm implementations by chapter
│   ├── chapter_01.py           # Linear search, iteration patterns
│   ├── chapter_04.py           # Bubble sort
│   └── ...
├── benchmarks/                  # Benchmark test harnesses
│   ├── benchmark_interface.py  # Abstract base class for all benchmarks
│   ├── chapter_01.py           # Chapter 1 algorithm benchmarks
│   ├── chapter_04.py           # Chapter 4 algorithm benchmarks
│   └── ...
└── env/                        # Python virtual environment
```

### Architecture

- **`algorithms/`**: Contains pure algorithm implementations directly from the textbook, organized by chapter
- **`benchmarks/`**: Houses test harnesses that implement the `BenchmarkStrategy` interface, defining how each algorithm should be set up and measured
- **`benchmark_interface.py`**: Defines the abstract contract that all benchmark classes must fulfill
- **`main.py`**: Orchestrates benchmark execution, data collection, and visualization using matplotlib

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd common-sense-big-o
```

2. Create and activate a virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # On macOS/Linux
# env\Scripts\activate   # On Windows
```

3. Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Running Benchmarks

1. Activate the virtual environment (if not already active):
```bash
source env/bin/activate
```

2. Execute the main script:
```bash
python3 main.py
```

3. The script will:
   - Run benchmarks for all enabled algorithms
   - Display progress in the terminal
   - Generate matplotlib graphs showing time and space complexity
   - Present visualizations comparing theoretical vs. empirical performance

### Selecting Algorithms to Benchmark

Edit the `tests` list in `main.py` to enable/disable specific benchmarks:

```python
tests = [
    # PrintNumbersVersionOneBenchmark(),
    # PrintNumbersVersionTwoBenchmark(),
    # LinearSearch(),
    BubbleSort()
]
```

Uncomment lines to activate additional benchmarks.

### Adjusting Input Sizes

Modify the `n_sizes` configuration in `main.py` to test different input ranges:

```python
n_sizes = np.logspace(5, 6, 10, dtype=int)  # 10 values from 10^5 to 10^6
```

## Example Output

### Bubble Sort Performance Analysis

![Bubble Sort Time Complexity](docs/screenshots/bubble_sort.png)
*Figure 1: Empirical runtime (blue scatter) closely follows theoretical O(N²) curve (red dashed line). Memory usage validates expected O(1) constant space complexity*

## Technologies Used

- **Python 3.14**: Core programming language
- **NumPy**: Numerical computing and array operations
- **Matplotlib**: Data visualization and plotting
- **tracemalloc**: Built-in memory profiling
- **time.perf_counter()**: High-resolution performance timing

## Roadmap

### Planned Features

- [ ] Implement algorithms from remaining book chapters
- [ ] Add multiprocessing support to parallelize benchmark execution
- [ ] Add warmup runs to account for Python interpreter optimization
- [ ] Create interactive terminal UI for algorithm and input size selection
- [ ] Switch from venv to uv

### Algorithms To Implement

#### Chapter 2: Why Algorithms Matter
- [ ] Binary search

#### Chapter 5: Optimizing Code with and wiBig O
- [ ] Selection Sort

#### Chapter 6: Optimizing for Optimistic Scenarios
- [ ] Insertion Sort

#### Chapter 11: Recursion
- [ ] Factorial (recursive)
- [ ] Fibonacci (recursive and optimized)

#### Chapter 12: Dynamic Programming
- [ ] Memoized algorithms

#### Chapter 13: Recursive Algorithms for Speed
- [ ] Quicksort
- [ ] Quickselect

*Additional chapters to be added as I progress*