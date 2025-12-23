from .benchmark_interface import BenchmarkStrategy
from algorithms.chapter_02 import binary_search
import numpy as np
import random

class BinarySearch(BenchmarkStrategy):
    @property
    def name(self):
        return binary_search.__name__

    def setup(self, n):
        # Create a list n long with one unique item inserted at a random index to search for
        self.data_list = list(range(0, n))
        self.search_value = random.randint(0, n - 1)

    def run_algorithm(self):
        binary_search(self.data_list, self.search_value)

    @property
    def time_complexity(self):
        return "O(log(N))", lambda n: np.log(n)

    @property
    def space_complexity(self):
        return "O(1)", lambda n: 1