from tensorflow.keras import backend as K
import tensorflow as tf

from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fedavg import get_y
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


class FairAggregation(Algorithm):

    def __init__(self):
        name = "fair_aggregation"
        super().__init__(name)

    def perform_fl(self, n_timesteps, n_crounds, n_clients, n_features, clients_data, seed, is_image):
        global_model = NN_model(n_features, seed, is_image)
        clients_metrics = [get_metrics(is_image) for _ in range(n_clients)]

        for timestep in range(n_timesteps):
            # STEP 1 - test
            for client_data, client_metrics in zip(clients_data[timestep], clients_metrics):
                x, y, s = client_data
                pred = global_model.predict(x)
                y, pred = get_y(y, pred, is_image)
                for client_metric in client_metrics:
                    res = client_metric.update(y, pred, s)
                    print(res, client_metric.name)

            # STEP 2 - Train and average models
            for cround in range(n_crounds):
                local_weights_list = []
                client_scaling_factors = []
                local_res_list = []
                local_res_sum = [0 for _ in range(len(clients_metrics[0]))]
                for client in range(n_clients):
                    x, y, s = clients_data[timestep][client]
                    global_weights = global_model.get_weights()
                    local_model = NN_model(n_features, seed, is_image)
                    local_model.compile(is_image)
                    local_model.set_weights(global_weights)
                    local_model.learn(x, y)
                    local_weights_list.append(local_model.get_weights())
                    client_scaling_factors.append(len(x))

                    # calculate fairness of each client (weights on aggregation)
                    pred = global_model.predict(x)
                    y, pred = get_y(y, pred, is_image)
                    local_res = []
                    for k, client_metric in enumerate(clients_metrics[client]):
                        res = client_metric.calculate(y, pred, s)
                        print(k, res)
                        local_res.append(res)
                        local_res_sum[k] += res
                    local_res_list.append(local_res)

                    # K.clear_session()
                    print("Trained model timestep {} cround {} client {}".format(timestep, cround, client))
                scales = calculate_scales(client_scaling_factors, local_res_list, local_res_sum)
                new_global_weights = calculate_global_weights(local_weights_list, scales)
                global_model.set_weights(new_global_weights)
                print("Averaged models on timestep {} cround {}".format(timestep, cround))

        # Client identity is always 0 (only one global model)
        client_identities = [[] for _ in range(n_clients)]
        for i in range(n_clients):
            for _ in range(n_timesteps):
                client_identities[i].append(0)

        return clients_metrics, client_identities


def calculate_scales(client_scaling_factors, local_res_list, local_res_sum):
    scales = [0 for _ in  range(len(client_scaling_factors))]
    global_count = sum(client_scaling_factors)
    for i, (local_count, local_res) in enumerate(zip(client_scaling_factors, local_res_list)):
        scales[i] += local_count / global_count
        for k in range(len(local_res)):
            scales[i] += local_res[k] / local_res_sum[k]

    sum_scales = sum(scales)
    weighted_scales = []
    for i in range(len(scales)):
        weighted_scales.append(scales[i] / sum_scales)

    return weighted_scales


def calculate_global_weights(local_weights_list, scales):
    scaled_local_weights_list = []
    for local_weights, scale in zip(local_weights_list, scales):
        scaled_local_weights = []
        for i in range(len(local_weights)):
            scaled_local_weights.append(scale * local_weights[i])
        scaled_local_weights_list.append(scaled_local_weights)

    global_weights = []
    for grad_list_tuple in zip(*scaled_local_weights_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        global_weights.append(layer_mean)

    return global_weights
