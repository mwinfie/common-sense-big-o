"""Benchmark I/O utilities for saving, loading, and plotting results."""

import json
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from datetime import datetime
from pathlib import Path


def save_results(results_data, filename=None):
    """Save benchmark results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename

    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


def load_results(filepath):
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def plot_results(results_data, n_sizes, output_format='window'):
    """Generate plots from saved results data."""
    tests_data = results_data['tests']

    # Calculate figure size
    fig_height = 6 * len(tests_data) if len(tests_data) > 3 else 5 * len(tests_data)
    fig, axs = plt.subplots(len(tests_data), 2, figsize=(16, fig_height), squeeze=False)
    fig.suptitle('Theoretical vs. Empirical Performance', y=0.995)

    if len(tests_data) > 3:
        fig.canvas.manager.set_window_title('Benchmark Results (use toolbar to pan/zoom)')
        plt.subplots_adjust(left=0.1, right=0.95, top=0.98, bottom=0.05)

    for i, test_data in enumerate(tests_data):
        y_time = np.array(test_data['time_measurements'])
        y_space = np.array(test_data['space_measurements'])
        time_label = test_data['time_complexity_label']
        space_label = test_data['space_complexity_label']
        test_name = test_data['name']
        theory_time = np.array(test_data['theory_time'])
        theory_space = np.array(test_data['theory_space'])

        # Plot Time
        axs[i, 0].scatter(n_sizes, y_time, color='blue', label='Measured Time', alpha=0.6)
        axs[i, 0].plot(n_sizes, theory_time, color='red', linestyle='--', label=f'{time_label}')
        axs[i, 0].set_title(f"{test_name}: Time Complexity")
        axs[i, 0].set_xlabel("N")
        axs[i, 0].set_ylabel("Seconds")
        axs[i, 0].legend()
        axs[i, 0].grid(True, alpha=0.3)

        # Plot Space
        axs[i, 1].scatter(n_sizes, y_space, color='orange', label='Measured Space', alpha=0.6)
        axs[i, 1].plot(n_sizes, theory_space, color='green', linestyle='--', label=f'{space_label}')
        axs[i, 1].set_title(f"{test_name}: Space Complexity")
        axs[i, 1].set_xlabel("N")
        axs[i, 1].set_ylabel("Peak Bytes")
        axs[i, 1].legend()
        axs[i, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98], h_pad=3.0, w_pad=3.0)

    if output_format == 'pdf':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = Path("results") / f"benchmark_results_{timestamp}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"\nPDF saved to: {pdf_path}")
        print("Opening PDF...")
        plt.close()
        subprocess.run(['open', pdf_path])  # macOS
    else:
        plt.show()
