from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fair_fed_drift.fed_drift import perform_fl
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged


class FairFedDrift(Algorithm):

    def __init__(self):
        self.metrics_clustering = [LossPrivileged(), LossUnprivileged()]
        self.thresholds = []
        name = "FairFedDrift"
        color = "orange"
        super().__init__(name, color)

    def set_specs(self, args):
        threshold_p = float(args.thresholds[0])
        threshold_up = float(args.thresholds[1])
        self.thresholds = [threshold_p, threshold_up]
        super().set_subfolders("{}/loss_p-{}/loss_up-{}".format(self.name, threshold_p, threshold_up))

    def perform_fl(self, seed, clients_data, dataset):
        return perform_fl(self, seed, clients_data, dataset)
