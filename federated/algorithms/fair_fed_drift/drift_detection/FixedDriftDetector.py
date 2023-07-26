from federated.algorithms.fair_fed_drift.drift_detection.DriftDetector import DriftDetector


class FixedDriftDetector(DriftDetector):

    def __init__(self):
        name = "fixed"
        super().__init__(name)
        self.threshold = 0.8

    def drift_detected(self, timestep, values_best):
        drift = []
        for value in values_best:
            if timestep > 0 and value < self.threshold:
                drift.append(1)
            else:
                drift.append(0)

        return drift
