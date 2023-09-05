from federated.algorithms.Algorithm import Algorithm, average_weights
from federated.model import NN_model
from metrics.BalancedAccuracy import divide_one
from metrics.MetricFactory import get_metrics
import numpy as np
import pandas as pd


class FedAvgLR(Algorithm):

    def __init__(self):
        name = "fedavg_lr"
        super().__init__(name)

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
                    x, y, s, y_original = clients_data[timestep][client]
                    global_weights = global_model.get_weights()
                    local_model = NN_model(dataset.input_shape, seed, dataset.is_image)
                    local_model.compile(dataset.is_image)
                    local_model.set_weights(global_weights)
                    sample_weights = calculate_sample_weights(y_original, s)
                    local_model.learn(x, y, sample_weights=sample_weights)
                    local_weights_list.append(local_model.get_weights())
                    client_scaling_factors_list.append(len(x))
                    #K.clear_session()
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


def calculate_sample_weights(y_original, s):
    df = pd.DataFrame({"y": y_original, "s": s})
    sample_weights = np.zeros(len(df))

    for target in df["y"].unique():
        observed_value_priv = len(df[(df['y'] == target) & (df['s'] == 1)]) / len(df)
        observed_value_unpriv = len(df[(df['y'] == target) & (df['s'] == 0)]) / len(df)
        expected_value_priv = (len(df[(df['y'] == target)]) / len(df)) * (len(df[(df['s'] == 1)]) / len(df))
        expected_value_unpriv = (len(df[(df['y'] == target)]) / len(df)) * (len(df[(df['s'] == 0)]) / len(df))
        weight_value_priv = divide_one(expected_value_priv, observed_value_priv)
        weight_value_unpriv = divide_one(expected_value_unpriv, observed_value_unpriv)
        print("Weights target {} - priv: {:.2f}, unpriv: {:.2f}".format(target, weight_value_priv, weight_value_unpriv))
        sample_weights[df[(df['y'] == target) & (df['s'] == 1)].index.tolist()] = weight_value_priv
        sample_weights[df[(df['y'] == target) & (df['s'] == 0)].index.tolist()] = weight_value_unpriv

    return sample_weights
