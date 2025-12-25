from .benchmark_interface import BenchmarkStrategy
from algorithms.chapter_05 import selection_sort
import random

class SelectionSort(BenchmarkStrategy):
    @property
    def name(self):
        return selection_sort.__name__

    def setup(self, n):
        self.data_list = random.sample(range(1, n * 2), n)

    def run_algorithm(self):
        selection_sort(self.data_list)

    @property
    def time_complexity(self):
        return "O(N^2)", lambda n: n ** 2

    @property
    def space_complexity(self):
        return "O(1)", lambda n: 1