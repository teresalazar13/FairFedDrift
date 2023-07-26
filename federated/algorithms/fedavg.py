from tensorflow.keras import backend as K

from federated.algorithms.Algorithm import Algorithm
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


class FedAvg(Algorithm):

    def __init__(self):
        name = "fedavg"
        super().__init__(name)

    def perform_fl(self, seed, clients_data, dataset):
        global_model = NN_model(dataset.n_features, seed, dataset.is_image)
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
                    local_model = NN_model(dataset.n_features, seed, dataset.is_image)
                    local_model.compile(dataset.is_image)
                    local_model.set_weights(global_weights)
                    local_model.learn(x, y)
                    local_weights_list.append(local_model.get_weights())
                    client_scaling_factors_list.append(len(x))
                    #K.clear_session()
                    print("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

                new_global_weights = super().average_weights(local_weights_list, client_scaling_factors_list)
                global_model.set_weights(new_global_weights)
                print("Averaged models on timestep {} cround {}".format(timestep, cround))

        # Client identity is always 0 (only one global model)
        client_identities = [[] for _ in range(dataset.n_clients)]
        for i in range(dataset.n_clients):
            for _ in range(dataset.n_timesteps):
                client_identities[i].append(0)

        return clients_metrics, client_identities
