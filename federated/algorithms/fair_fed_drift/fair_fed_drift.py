from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fair_fed_drift.fed_drift import test_models, update_global_models, merge_global_models, \
    train_and_average, print_clients_identities
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged
from metrics.MetricFactory import get_metrics


class FairFedDrift(Algorithm):

    def __init__(self):
        self.metrics_clustering = [LossPrivileged(), LossUnprivileged()]
        self.thresholds = []
        name = "FairFedDrift"
        super().__init__(name)

    def set_specs(self, args):
        threshold_p = float(args.thresholds[0])
        threshold_up = float(args.thresholds[1])
        self.thresholds = [threshold_p, threshold_up]
        super().set_subfolders("{}/loss_p-{}/loss_up-{}".format(self.name, threshold_p, threshold_up))

    def perform_fl(self, seed, clients_data, dataset):
        pass
