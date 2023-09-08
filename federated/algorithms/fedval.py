from federated.algorithms.Algorithm import Algorithm, average_weights
from federated.model import NN_model
from metrics.MetricFactory import get_metrics, get_metrics_by_names


class FedVal(Algorithm):

    def __init__(self):
        self.metric_clustering = None
        name = "fedval"
        super().__init__(name)

    def set_specs(self, args):
        self.metric_clustering = get_metrics_by_names(args.metrics)[0]
        metrics_string = args.metrics[0]
        super().set_subfolders("{}/metric-{}".format(self.name, metrics_string))

    def perform_fl(self, seed, clients_data, dataset):
        global_model = NN_model(dataset.input_shape, seed, dataset.is_image)
        clients_metrics = [get_metrics(dataset.is_image) for _ in range(dataset.n_clients)]

        for timestep in range(dataset.n_timesteps):
            # STEP 1 - Test
            super().test(clients_data[timestep], clients_metrics, global_model, dataset)

            # STEP 2 - Train and average models
            for cround in range(dataset.n_rounds):
                local_weights_list = []
                client_scaling_factors_list = []
                for client in range(dataset.n_clients):
                    x, y, s, _ = clients_data[timestep][client]
                    global_weights = global_model.get_weights()
                    local_model = NN_model(dataset.input_shape, seed, dataset.is_image)
                    local_model.compile(dataset.is_image)
                    local_model.set_weights(global_weights)
                    local_model.learn(x, y)
                    local_weights_list.append(local_model.get_weights())

                    # calculate fairness of each client (weights on aggregation)
                    pred = global_model.predict(x)
                    y_true, y_pred = super().get_y(y, pred, dataset.is_image)
                    res = self.metric_clustering.calculate(y_true, y_pred, s)
                    client_scaling_factors_list.append(res)

                    # K.clear_session()
                    print("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

                new_global_weights = average_weights(local_weights_list, client_scaling_factors_list)
                global_model.set_weights(new_global_weights)
                print("Averaged models on timestep {} cround {}".format(timestep, cround))

        # Client identity is always 0 (only one global model)
        client_identities = [[] for _ in range(dataset.n_clients)]
        for i in range(dataset.n_clients):
            for _ in range(dataset.n_timesteps):
                client_identities[i].append(0)

        return clients_metrics, client_identities
