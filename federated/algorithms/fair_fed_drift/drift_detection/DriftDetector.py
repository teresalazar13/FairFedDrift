from abc import abstractmethod


class DriftDetector:

    def __init__(self, name):
        self.name = name

    def set_specs(self, args):
        raise NotImplementedError("Must implement set_specs")

    @abstractmethod
    def drift_detected(self, values_best, timestep):
        raise NotImplementedError("Must implement drift_detected")

    @abstractmethod
    def get_worst_results(self, results_list):
        raise NotImplementedError("Must implement get_worst_results")

    @abstractmethod
    def get_next_best_results(self, results_matrix):
        raise NotImplementedError("Must implement get_next_best_results")

