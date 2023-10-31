from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fair_fed_drift.ClientData import ClientData
from federated.algorithms.fair_fed_drift.GlobalModels import GlobalModels
from federated.algorithms.fair_fed_drift.fed_drift import get_init_model, update_clients_identities, test_models, \
    update_global_models, merge_global_models, train_and_average
from metrics.Loss import Loss
from metrics.MetricFactory import get_metrics


WORST_LOSS = 1000


class FairFedDrift(Algorithm):

    def __init__(self):
        self.metric_clustering = Loss()
        self.threshold = None
        name = "FairFedDrift"
        super().__init__(name)

    def set_specs(self, args):
        self.threshold = float(args.threshold)
        super().set_subfolders("{}/loss-{}".format(
            self.name, self.threshold)
        )

    def perform_fl(self, seed, clients_data, dataset):
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        global_model = global_models.create_new_global_model(init_model)
        for client_id in range(dataset.n_clients):
            cd = ClientData(clients_data[0][client_id][0], clients_data[0][client_id][1], clients_data[0][client_id][2])
            global_model.set_client(client_id, cd)
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        clients_identities = [[] for _ in range(dataset.n_clients)]

        for timestep in range(dataset.n_timesteps):
            clients_identities = update_clients_identities(clients_identities, dataset.n_clients, global_models)

            # STEP 1 - Test each client's data on previous clustering identities
            test_models(global_models, clients_data[timestep], clients_metrics, dataset, seed)
            global_models.reset_clients()

            if timestep != dataset.n_timesteps - 1:
                # STEP 2 - Recalculate Global Models (cluster identities) and detect concept drift
                global_models = update_global_models(
                    self.metric_clustering, self.threshold, clients_data[timestep], global_models, dataset, seed, timestep
                )

                # STEP 3 - Merge Global Models
                global_models = merge_global_models(
                    self.metric_clustering, self.threshold, global_models, dataset, seed
                )

                # STEP 4 - Train and average models
                global_models = train_and_average(clients_data[timestep], global_models, dataset, seed, timestep)

        return clients_metrics, clients_identities
