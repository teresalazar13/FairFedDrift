from federated.algorithms.Algorithm import Algorithm, average_weights, test
from federated.algorithms.Identity import Identity
from federated.model import NN_model
from metrics.MetricFactory import get_metrics
import logging


class FedAvg(Algorithm):

    def __init__(self):
        name = "FedAvg"
        color = "blue"
        marker = "+"
        super().__init__(name, color, marker)

    def perform_fl(self, seed, clients_data, dataset):
        global_model = NN_model(dataset, seed)
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        # Train with data from first timestep
        global_model = train_and_average(global_model, dataset, clients_data, 0, seed)

        for timestep in range(1, dataset.n_timesteps):
            test(clients_data[timestep], clients_metrics, global_model, dataset)
            global_model = train_and_average(global_model, dataset, clients_data, timestep, seed)

        # Clients identities are always 0 (only one global model)
        clients_identities = [[] for _ in range(dataset.n_clients)]
        for i in range(dataset.n_clients):
            for _ in range(1, dataset.n_timesteps):
                clients_identities[i].append(Identity(0, 0))

        return clients_metrics, clients_identities


def train_and_average(global_model, dataset, clients_data, timestep, seed):
    for cround in range(dataset.n_rounds):
        local_weights_list = []
        client_scaling_factors_list = []
        print(f"GLOBAL model weight count: {len(global_model.get_weights())}")
        for client in range(dataset.n_clients):
            x, y, s, _ = clients_data[timestep][client]
            global_weights = global_model.get_weights()
            local_model = NN_model(dataset, seed)
            local_model.compile(dataset)
            local_model.set_weights(global_weights)
            local_model.learn(x, y)
            local_weights_list.append(local_model.get_weights())
            client_scaling_factors_list.append(len(x))
            # K.clear_session()
            logging.info("Trained model timestep {} cround {} client {}".format(timestep, cround, client))
            print(f"LOCAL model weight count: {len(local_model.get_weights())}")

        new_global_weights = average_weights(local_weights_list, client_scaling_factors_list)
        print(f"AVG GLOBAL model weight count: {len(new_global_weights)}")
        global_model.set_weights(new_global_weights)
        logging.info("Averaged models on timestep {} cround {}".format(timestep, cround))

    return global_model