import math

from federated.algorithms.Algorithm import Algorithm
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged


class FairFedDriftIA(Algorithm):

    def __init__(self):
        self.metrics_clustering = [LossPrivileged(), LossUnprivileged()]
        self.thresholds = []
        self.window = None
        name = "FairFedDrift-IA"
        color = "black"
        marker = "o"
        super().__init__(name, color, marker)

    def set_specs(self, args):
        threshold_p = float(args.thresholds[0])
        threshold_up = float(args.thresholds[1])
        self.thresholds = [threshold_p, threshold_up]
        self.window = math.inf
        if args.window:
            self.window = int(args.window)
        super().set_subfolders("{}/window-{}/loss_p-{}/loss_up-{}".format(self.name, self.window, threshold_p, threshold_up))

    def perform_fl(self, seed, clients_data, dataset):

        return
