from abc import abstractmethod


class DriftDetector:

    def __init__(self, name):
        self.name = name

    def set_specs(self, args):
        pass

    @abstractmethod
    def drift_detected(self, timestep, values_best):
        raise NotImplementedError("Must implement drift_detected")
