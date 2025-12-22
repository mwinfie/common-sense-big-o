from abc import ABC, abstractmethod

class BenchmarkStrategy(ABC):
    def __init__(self):
        self.input_data = None

    @abstractmethod
    def setup(self, n):
        pass

    @abstractmethod
    def run_algorithm(self):
        pass
        
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def time_complexity(self):
        pass

    @property
    @abstractmethod
    def space_complexity(self):
        pass