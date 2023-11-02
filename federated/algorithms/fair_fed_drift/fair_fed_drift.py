from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fair_fed_drift.fed_drift import test_models, update_global_models, merge_global_models, \
    train_and_average, print_clients_identities, setup
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
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        global_models, clients_identities = setup(seed, clients_data, dataset)

        for timestep in range(1, dataset.n_timesteps):
            print_clients_identities(clients_identities)
            # STEP 1 - Test each client's data on previous clustering identities (previous timestep)
            test_models(global_models, clients_data[timestep], clients_metrics, dataset, timestep - 1, seed)

            if timestep != dataset.n_timesteps - 1:
                # STEP 2 - Recalculate Global Models (cluster identities) and detect concept drift
                global_models, clients_identities = update_global_models(
                    self.metrics_clustering, self.thresholds, clients_data[timestep], global_models, dataset, seed,
                    timestep, clients_identities
                )

                # STEP 3 - Merge Global Models from previous timestep
                global_models = merge_global_models(
                    self.metrics_clustering, self.thresholds, global_models, dataset, seed, timestep - 1
                )

                # STEP 4 - Train and average models
                global_models = train_and_average(clients_data[timestep], global_models, dataset, seed, timestep)

        return clients_metrics, clients_identities
