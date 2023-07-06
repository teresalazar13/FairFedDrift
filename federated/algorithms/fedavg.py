import tensorflow as tf
from tensorflow.keras import backend as K

from federated.algorithms.Algorithm import Algorithm
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


class FedAvg(Algorithm):

    def __init__(self):
        name = "fedavg"
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
                client_scaling_factors_list = []
                for client in range(n_clients):
                    x, y, s = clients_data[timestep][client]
                    global_weights = global_model.get_weights()
                    local_model = NN_model(n_features, seed, is_image)
                    local_model.compile(is_image)
                    local_model.set_weights(global_weights)
                    local_model.learn(x, y)
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


def get_y(y, pred, is_image):
    y_new = []
    pred_new = []
    for y_i, pred_i in zip(y, pred):
        if not is_image:
            pred_new.append(0)
            if pred_i[0][0] > 0.5:
                pred_new.append(1)
            y_new.append(y_i)
        else:
            y_new.append(y_i.argmax())
            pred_new.append(pred_i.argmax())

    return y_new, pred_new


def calculate_global_weights(local_weights_list, client_scaling_factors):
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
