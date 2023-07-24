import tensorflow as tf
from tensorflow.keras import backend as K
import math
from copy import deepcopy

from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fedavg import get_y
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


class FedMom(Algorithm):

    def __init__(self):
        name = "fedmom"
        self.beta = 0.9
        super().__init__(name)

    def perform_fl(self, n_timesteps, n_crounds, n_clients, n_features, clients_data, seed, is_image):
        global_model = NN_model(n_features, seed, is_image)
        clients_metrics = [get_metrics(is_image) for _ in range(n_clients)]
        previous_momentum = [tf.zeros_like(weight) for weight in global_model.get_weights()]
        previous_global_weights = deepcopy(global_model.get_weights())
        iteration = 1

        for timestep in range(n_timesteps):
            # STEP 1 - test
            for client_data, client_metrics in zip(clients_data[timestep], clients_metrics):
                x, y, s, _ = client_data
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
                    x, y, s, _ = clients_data[timestep][client]
                    global_weights = deepcopy(global_model.get_weights())
                    local_model = NN_model(n_features, seed, is_image)
                    local_model.compile(is_image)
                    local_model.set_weights(global_weights)
                    local_model.learn(x, y)
                    local_weights_list.append(deepcopy(local_model.get_weights()))
                    client_scaling_factors_list.append(len(x))
                    # K.clear_session()

                    print("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

                    """
                    # TODO - remove
                    pred = global_model.predict(x)
                    y, pred = get_y(y, pred, is_image)
                    for k, client_metric in enumerate(clients_metrics[client]):
                        res = client_metric.calculate(y, pred, s)
                        print(k, res)"""

                new_global_weights, new_momentum = calculate_local_updates_momentum(
                    local_weights_list, client_scaling_factors_list, previous_global_weights, previous_momentum,
                    self.beta, iteration
                )
                global_model.set_weights(deepcopy(new_global_weights))
                previous_global_weights = deepcopy(new_global_weights)
                previous_momentum = deepcopy(new_momentum)
                print("Averaged models on timestep {} cround {}".format(timestep, cround))
                iteration += 1

        # Client identity is always 0 (only one global model)
        client_identities = [[] for _ in range(n_clients)]
        for i in range(n_clients):
            for _ in range(n_timesteps):
                client_identities[i].append(0)

        return clients_metrics, client_identities


def calculate_local_updates_momentum(
        local_weights_list, client_scaling_factors, previous_global_weights, previous_momentum, beta, iteration
):
    scaled_local_updates_list = []
    global_count = sum(client_scaling_factors)

    for local_weights, local_count in zip(local_weights_list, client_scaling_factors):
        scale = local_count / global_count
        scaled_local_weights = []
        for layer in range(len(local_weights)):
            scaled_local_weights.append(scale * (local_weights[layer] - previous_global_weights[layer]))
        scaled_local_updates_list.append(scaled_local_weights)

    new_global_weights = []
    new_momentum = []
    for layer, grad_list_tuple in enumerate(zip(*scaled_local_updates_list)):
        alpha = tf.math.reduce_sum(grad_list_tuple, axis=0)
        new_momentum_l = beta * previous_momentum[layer] + (1-beta) * alpha
        new_momentum.append(deepcopy(new_momentum_l))
        bias_correction = new_momentum_l / (1.0 - math.pow(beta, iteration))
        new_global_weights.append(deepcopy(previous_global_weights[layer] + bias_correction))

    return new_global_weights, new_momentum
