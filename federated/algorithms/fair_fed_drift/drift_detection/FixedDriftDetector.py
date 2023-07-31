from federated.algorithms.fair_fed_drift.drift_detection.DriftDetector import DriftDetector


class FixedDriftDetector(DriftDetector):

    def __init__(self):
        name = "fixed"
        self.thresholds = []
        super().__init__(name)

    def set_thresholds(self, args):
        thresholds = [float(t) for t in args.thresholds]
        self.thresholds = thresholds

    def drift_detected(self, timestep, values_best):
        drift = []
        for value, threshold in zip(values_best, self.thresholds):  # for each metric m
            if timestep > 0 and value < threshold:
                drift.append(1)
            else:
                drift.append(0)

        return drift
