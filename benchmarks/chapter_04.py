from .benchmark_interface import BenchmarkStrategy
from algorithms.chapter_04 import bubble_sort
import random

class BubbleSort(BenchmarkStrategy):
    @property
    def name(self):
        return bubble_sort.__name__

    def setup(self, n):
        self.data_list = random.sample(range(1, n * 2), n)

    def run_algorithm(self):
        bubble_sort(self.data_list)

    @property
    def time_complexity(self):
        return "O(N^2)", lambda n: n * n

    @property
    def space_complexity(self):
        return "O(1)", lambda n: 1