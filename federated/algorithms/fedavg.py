from federated.algorithms.Algorithm import Algorithm, average_weights, test
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


class FedAvg(Algorithm):

    def __init__(self):
        name = "FedAvg"
        color = "blue"
        super().__init__(name, color)

    def perform_fl(self, seed, clients_data, dataset):
        global_model = NN_model(dataset, seed)
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]

        for timestep in range(dataset.n_timesteps):
            global_model = train_and_average(global_model, dataset, clients_data, timestep, seed)
            timestep_to_test = timestep + 1
            if timestep_to_test == dataset.n_timesteps:
                timestep_to_test = 0
            test(clients_data[timestep_to_test], clients_metrics, global_model, dataset)

        # Clients identities are always 0 (only one global model)
        clients_identities = [[] for _ in range(dataset.n_clients)]
        for i in range(dataset.n_clients):
            for _ in range(dataset.n_timesteps):
                clients_identities[i].append(0)

        return clients_metrics, clients_identities, ""


def train_and_average(global_model, dataset, clients_data, timestep, seed):
    for cround in range(dataset.n_rounds):
        local_weights_list = []
        client_scaling_factors_list = []
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
            print("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

        new_global_weights = average_weights(local_weights_list, client_scaling_factors_list)
        global_model.set_weights(new_global_weights)
        #clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]  # TODO - remove
        #test(clients_data[timestep + 1], clients_metrics, global_model, dataset)  # TODO - remove
        print("Averaged models on timestep {} cround {}".format(timestep, cround))

    return global_model