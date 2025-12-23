from .benchmark_interface import BenchmarkStrategy
from algorithms.chapter_01 import print_numbers_version_one, print_numbers_version_two, linear_search
import random

class PrintNumbersVersionOneBenchmark(BenchmarkStrategy):
    @property
    def name(self):
        return print_numbers_version_one.__name__

    def setup(self, n):
        self.input_data = n

    def run_algorithm(self):
        print_numbers_version_one(self.input_data)

    @property
    def time_complexity(self):
        return "O(N)", lambda n: n

    @property
    def space_complexity(self):
        return "O(1)", lambda n: 1

class PrintNumbersVersionTwoBenchmark(BenchmarkStrategy):
    @property
    def name(self):
        return print_numbers_version_two.__name__

    def setup(self, n):
        self.input_data = n

    def run_algorithm(self):
        print_numbers_version_two(self.input_data)

    @property
    def time_complexity(self):
        return "O(N)", lambda n: n

    @property
    def space_complexity(self):
        return "O(1)", lambda n: 1

class LinearSearch(BenchmarkStrategy):
    @property
    def name(self):
        return linear_search.__name__

    def setup(self, n):
        # Create a list n long with one unique item inserted at a random index to search for
        self.data_list = ["hello"] * n
        self.search_value = "world"
        self.data_list.insert(random.randint(0, n), self.search_value)

    def run_algorithm(self):
        linear_search(self.data_list, self.search_value)

    @property
    def time_complexity(self):
        return "O(N)", lambda n: n

    @property
    def space_complexity(self):
        return "O(1)", lambda n: 1