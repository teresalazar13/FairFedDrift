import time
import tensorflow as tf
from tensorflow.keras import backend as K

from federated_online.algorithms.Algorithm import Algorithm
from federated_online.model import NN_model_online
from metrics_online.MetricFactory import get_metrics


class FedAvg(Algorithm):

    def __init__(self):
        name = "fedavg"
        super().__init__(name)

    def perform_fl(self, n_rounds, n_clients, n_features, clients_data, seed, is_image):
        global_model = NN_model_online(n_features, seed, is_image)
        clients_metrics = [get_metrics(is_image) for _ in range(n_clients)]

        for round in range(n_rounds):
            global_weights = global_model.get_weights()
            local_weights_list = []
            client_scaling_factors = []

            for client in range(n_clients):
                start = time.time()
                local_model = NN_model_online(n_features, seed, is_image)
                local_model.compile(is_image)
                local_model.set_weights(global_weights)
                client_size = len(clients_data[round][client][0])
                for i in range(client_size):
                    x_i = clients_data[round][client][0][i]
                    y_i = clients_data[round][client][1][i]
                    s_i = clients_data[round][client][2][i]
                    logit_pred = local_model.predict_one(x_i)
                    y_pred = 0
                    if logit_pred[0][0] > 0.5:
                        y_pred = 1
                    for metric in clients_metrics[client]:
                        metric.update(y_i, y_pred, s_i)
                    local_model.learn_one(x_i, y_i)
                local_weights_list.append(local_model.get_weights())
                client_scaling_factors.append(client_size)
                K.clear_session()
                end = time.time()
                print("Time of round {} client {}: {}s".format(round + 1, client + 1, end - start))

            new_global_weights = self.calculate_global_weights(local_weights_list, client_scaling_factors)
            global_model.set_weights(new_global_weights)

        return clients_metrics

    def calculate_global_weights(self, local_weights_list, client_scaling_factors):
        scaled_local_weights_list = []
        global_count = sum(client_scaling_factors)

        for local_weights, local_count in zip(local_weights_list, client_scaling_factors):
            scale = local_count / global_count
            scaled_local_weights = []
            for i in range(len(local_weights)):
                scaled_local_weights.append(scale * local_weights[i])

            scaled_local_weights_list.append(scaled_local_weights)

        global_weights = []
        for grad_list_tuple in zip(*scaled_local_weights_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            global_weights.append(layer_mean)

        return global_weights
