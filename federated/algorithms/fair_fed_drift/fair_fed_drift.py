from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fair_fed_drift.ClientIdentity import ClientIdentity
from federated.algorithms.fair_fed_drift.GlobalModels import GlobalModels
from federated.algorithms.fair_fed_drift.fed_drift import test_models, update_global_models, merge_global_models, \
    train_and_average, print_clients_identities, get_init_model, WORST_LOSS, print_matrix, set_clients, \
    create_global_models_drifted_clients
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
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        global_model = global_models.create_new_global_model(init_model)
        clients_identities = [[ClientIdentity(global_model.id, global_model.name)] for _ in range(dataset.n_clients)]
        minimum_loss_clients = [[[WORST_LOSS for _ in range(len(self.metrics_clustering))]] for _ in range(dataset.n_clients)]  # used for drift detection

        for timestep in range(dataset.n_timesteps):
            print_clients_identities(clients_identities)
            print_matrix(minimum_loss_clients)

            # STEP 4 - Train and average models with data from this timestep
            global_models = train_and_average(
                clients_data[timestep], global_models, dataset, seed, timestep, clients_identities
            )
            set_clients(global_models, clients_identities, clients_data[timestep])  # to be used in merge for sampling

            timestep_to_test = timestep + 1
            if timestep_to_test == dataset.n_timesteps:
                timestep_to_test = 0

            # STEP 1 - Test each client's data on data from next timestep
            test_models(
                global_models, clients_data[timestep_to_test], clients_metrics, dataset, clients_identities, seed
            )

            if timestep != dataset.n_timesteps - 1:
                # STEP 2 - Recalculate Global Models using data from next timestep
                # (cluster identities for next timestep)
                # and detect concept drift
                global_models, global_models_to_create, clients_identities, minimum_loss_clients = update_global_models(
                    self.metrics_clustering, self.thresholds, clients_data[timestep_to_test], global_models, dataset,
                    seed, clients_identities, minimum_loss_clients
                )

                # STEP 3 - Merge Global Models  # TODO - move to after training
                global_models = merge_global_models(self.metrics_clustering, self.thresholds, global_models, dataset, seed)

                # STEP 2.1 - create global models from drifted clients
                # (has to be after merge - don't want to try to merge drifted clients)
                global_models, clients_identities = create_global_models_drifted_clients(
                    global_models, global_models_to_create, clients_identities
                )

        clients_identities_string = print_clients_identities(clients_identities)

        return clients_metrics, clients_identities, clients_identities_string
