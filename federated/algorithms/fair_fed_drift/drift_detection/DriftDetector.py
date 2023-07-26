from abc import abstractmethod


class DriftDetector:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def drift_detected(self, timestep, values_best):
        raise NotImplementedError("Must implement drift_detected")
