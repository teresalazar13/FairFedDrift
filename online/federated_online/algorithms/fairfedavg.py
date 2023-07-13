import time
import tensorflow as tf
from tensorflow.keras import backend as K

from federated_online.algorithms.Algorithm import Algorithm
from federated_online.model import NN_model_online
from metrics_online.MetricFactory import get_metrics


class FairFedAvg(Algorithm):

    def __init__(self):
        name = "fair-fedavg"
        super().__init__(name)

    def perform_fl(self, n_rounds, n_clients, n_features, clients_data, seed, is_image):
        initial_global_model = NN_model_online(n_features, seed, is_image)
        global_models = [initial_global_model]
        client_gm_ids = [0 for _ in range(n_clients)]
        client_gm_ids_col = [[] for _ in range(n_clients)]
        clients_metrics = [get_metrics(is_image) for _ in range(n_clients)]

        for round in range(n_rounds):
            local_weights_list = [[] for _ in range(len(global_models))]
            client_scaling_factors = [[] for _ in range(len(global_models))]

            for client in range(n_clients):
                continue_loop = True
                check_for_concept_drift = True
                start = time.time()
                client_size = len(clients_data[round][client][0])
                while continue_loop:
                    gm_id = client_gm_ids[client]
                    global_weights = global_models[gm_id].get_weights()
                    local_model = NN_model_online(n_features, seed, is_image)
                    local_model.compile(is_image)
                    local_model.set_weights(global_weights)
                    local_model = train_model(round, clients_data, client, client_size, local_model, is_image, clients_metrics, check_for_concept_drift)
                    if local_model != -1:  # if no concept drift
                        local_weights_list[gm_id].append(local_model.get_weights())
                        client_scaling_factors[gm_id].append(client_size)
                        K.clear_session()
                        end = time.time()
                        print("Time of round {} client {}: {}s".format(round, client, end - start))
                        continue_loop = False
                    else:
                        new_model_id = len(global_models)
                        client_gm_ids[client] = new_model_id  # update client's global model is
                        new_model = NN_model_online(n_features, seed, is_image)  # create new.csv global model
                        new_model.set_weights(global_models[gm_id].get_weights())
                        global_models.append(new_model)  # add new.csv global model to list
                        local_weights_list.append([])
                        client_scaling_factors.append([])
                        continue_loop = True
                        check_for_concept_drift = False
                        print("New model created {} for client {}".format(new_model_id, client))
                        print(client_gm_ids)
                        # TODO - increase learning rate
                        # TODO - check if there are other models to train
                        # TODO - merge models (?)
                client_gm_ids_col[client].extend([client_gm_ids[client]] * client_size)

            for i in range(len(global_models)):
                if len(local_weights_list[i]) > 0:
                    print("Aggregating for global model", i, "len-", len(local_weights_list[i]))
                    new_global_weights = self.calculate_global_weights(local_weights_list[i], client_scaling_factors[i])
                    global_models[i].set_weights(new_global_weights)
                else:
                    print("NOT Aggregating for global model", i, "len-", len(local_weights_list[i]))

        return clients_metrics, client_gm_ids_col

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


def train_model(round, clients_data, client, client_size, local_model, is_image, clients_metrics, check_for_concept_drift):
    for i in range(client_size):
        x_i = clients_data[round][client][0][i]
        y_i = clients_data[round][client][1][i]
        s_i = clients_data[round][client][2][i]
        logit_pred = local_model.predict_one(x_i)
        local_model.learn_one(x_i, y_i)
        y_i, y_pred = get_y(y_i, logit_pred, is_image)

        for metric in range(len(clients_metrics[client])):
            res = clients_metrics[client][metric].update(y_i, y_pred, s_i)
            if check_for_concept_drift and round >= 5 and metric == 0 and res < 0.7:  # if accuracy decreases to 0.7 after 5th round
                print("Drift detected at round {} client {}".format(round, client))
                new_size = len(clients_metrics[client][metric].res) - (i+1)
                clients_metrics[client][metric].res = clients_metrics[client][metric].res[:new_size]
                for metric_to_reset in range(len(clients_metrics[client])):
                    new_size = len(clients_metrics[client][metric_to_reset].res) - (i+2)
                    clients_metrics[client][metric_to_reset].res = clients_metrics[client][metric_to_reset].res[:new_size]
                return -1

    return local_model


def get_y(y_i, logit_pred, is_image):
    if not is_image:
        y_pred = 0
        if logit_pred[0][0] > 0.5:
            y_pred = 1
    else:
        y_i = y_i.argmax()
        y_pred = logit_pred.argmax()

    return y_i, y_pred
