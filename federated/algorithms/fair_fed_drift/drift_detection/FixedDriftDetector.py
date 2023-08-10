from federated.algorithms.fair_fed_drift.drift_detection.DriftDetector import DriftDetector


class FixedDriftDetector(DriftDetector):

    def __init__(self):
        name = "fixed"
        self.thresholds = []
        super().__init__(name)

    def set_specs(self, args):
        thresholds = [float(t) for t in args.thresholds]
        self.thresholds = thresholds

    def drift_detected(self, results_list, timestep=1):
        drift = []

        for result, threshold in zip(results_list, self.thresholds):  # for each metric m
            if timestep > 0 and result < threshold:
                drift.append(1)
            else:
                drift.append(0)

        return drift

    def get_worst_results(self, results_list):
        worst_results = results_list[0]
        worst_drift_count = sum(self.drift_detected(worst_results))
        worst_sum_results = sum(worst_results)

        for results in results_list:
            drift_count = sum(self.drift_detected(results))
            sum_results = sum(results)

            # If there are more drifts or the results are worse
            if drift_count > worst_drift_count or \
                    (drift_count == worst_drift_count and sum_results < worst_sum_results):
                worst_drift_count = drift_count
                worst_sum_results = sum(results)
                worst_results = results

        return worst_results

    def get_next_best_results(self, results_matrix):
        best_row = None
        best_col = None
        best_sum_results = 1000

        for row in range(len(results_matrix)):
            for col in range(len(results_matrix[row])):
                results = results_matrix[row][col]
                drift_count = sum(self.drift_detected(results))
                sum_results = sum(results)

                # If no drift and the results are better (hierarchical clustering)
                if drift_count == 0 and sum_results > best_sum_results:
                    best_sum_results = sum(results)
                    best_row = row
                    best_col = col

        return best_row, best_col
