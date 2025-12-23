from .benchmark_interface import BenchmarkStrategy
from algorithms.chapter_04 import bubble_sort

class BubbleSort(BenchmarkStrategy):
    @property
    def name(self):
        return bubble_sort.__name__

    def setup(self, n):
        self.data_list = list(range(n, 0, -1))

    def run_algorithm(self):
        bubble_sort(self.data_list)

    @property
    def time_complexity(self):
        return "O(N^2)", lambda n: n * n

    @property
    def space_complexity(self):
        return "O(1)", lambda n: 1