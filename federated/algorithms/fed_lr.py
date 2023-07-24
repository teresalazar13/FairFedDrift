import pandas as pd
import numpy as np
from tensorflow.keras import backend as K

from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fedavg import calculate_global_weights, get_y
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


class FedLR(Algorithm):

    def __init__(self):
        name = "fed_lr"
        super().__init__(name)

    def perform_fl(self, n_timesteps, n_crounds, n_clients, n_features, clients_data, seed, is_image):
        global_model = NN_model(n_features, seed, is_image)
        clients_metrics = [get_metrics(is_image) for _ in range(n_clients)]

        for timestep in range(n_timesteps):
            # STEP 1 - test
            for client_data, client_metrics in zip(clients_data[timestep], clients_metrics):
                x, y, s, y_original = client_data
                pred = global_model.predict(x)
                y, pred = get_y(y, pred, is_image)
                for client_metric in client_metrics:
                    res = client_metric.update(y, pred, s)
                    print(res, client_metric.name)

            # STEP 2 - Train and average models
            for cround in range(n_crounds):
                local_weights_list = []
                client_scaling_factors_list = []
                for client in range(n_clients):
                    x, y, s, y_original = clients_data[timestep][client]
                    global_weights = global_model.get_weights()
                    local_model = NN_model(n_features, seed, is_image)
                    local_model.compile(is_image)
                    local_model.set_weights(global_weights)
                    sample_weights = calculate_sample_weights(y_original, s)
                    local_model.learn(x, y, sample_weights=sample_weights)
                    local_weights_list.append(local_model.get_weights())
                    client_scaling_factors_list.append(len(x))
                    #K.clear_session()
                    print("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

                new_global_weights = calculate_global_weights(local_weights_list, client_scaling_factors_list)
                global_model.set_weights(new_global_weights)
                print("Averaged models on timestep {} cround {}".format(timestep, cround))

        # Client identity is always 0 (only one global model)
        client_identities = [[] for _ in range(n_clients)]
        for i in range(n_clients):
            for _ in range(n_timesteps):
                client_identities[i].append(0)

        return clients_metrics, client_identities


def calculate_sample_weights(y_, s):
    df = pd.DataFrame({"y": y_, "s": s})
    sample_weights = np.zeros(len(df))

    for target in df["y"].unique():
        observed_value_priv = len(df[(df['y'] == target) & (df['s'] == 1)]) / len(df)
        observed_value_unpriv = len(df[(df['y'] == target) & (df['s'] == 0)]) / len(df)
        expected_value_priv = (len(df[(df['y'] == target)]) / len(df)) * (len(df[(df['s'] == 1)]) / len(df))
        expected_value_unpriv = (len(df[(df['y'] == target)]) / len(df)) * (len(df[(df['s'] == 0)]) / len(df))
        weight_value_priv = expected_value_priv / observed_value_priv
        weight_value_unpriv = expected_value_unpriv / observed_value_unpriv
        print("Weights class {} - priv: {:.2f}, unpriv: {:.2f}".format(target, weight_value_priv, weight_value_unpriv))
        sample_weights[df[(df['y'] == target) & (df['s'] == 1)].index.tolist()] = weight_value_priv
        sample_weights[df[(df['y'] == target) & (df['s'] == 0)].index.tolist()] = weight_value_unpriv

    return sample_weights
