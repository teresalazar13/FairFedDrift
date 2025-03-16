from abc import abstractmethod
import logging
import copy
import torch
import tensorflow as tf


class Algorithm:

    def __init__(self, name, color, marker):
        self.name = name
        self.subfolders = name
        self.color = color
        self.marker = marker

    def set_subfolders(self, subfolders):
        self.subfolders = subfolders

    @abstractmethod
    def perform_fl(self, seed, clients_data, dataset):
        raise NotImplementedError("Must implement perform_fl")

    def set_specs(self, args):
        pass

def test(clients_data_timestep, clients_metrics, global_model, dataset):
    for client_data, client_metrics in zip(clients_data_timestep, clients_metrics):
        x, y_true_raw, s, _ = client_data
        y_pred_raw = global_model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, dataset.n_classes)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            logging.info("{}-{}".format(res, client_metric.name))

def get_y(y_true_raw, y_pred_raw, n_classes):
    y_true = []
    y_pred = []

    for y_true_original_i, y_pred_original_i in zip(y_true_raw, y_pred_raw):
        if n_classes == 2:
            y_pred_new_i = 0
            if y_pred_original_i[0] > 0.5:
                y_pred_new_i = 1
            y_pred.append(y_pred_new_i)
            y_true.append(y_true_original_i)
        else:
            y_pred.append(y_pred_original_i.argmax())
            y_true.append(y_true_original_i.argmax())

    return y_true, y_pred


def average_weights(is_pt, weights_list, scaling_factors):
    if not is_pt:  # tensorflow averaging
        scaled_local_weights_list = []
        global_count = sum(scaling_factors)
        for local_weights, local_count in zip(weights_list, scaling_factors):
            scale = local_count / global_count
            scaled_local_weights = []
            for i in range(len(local_weights)):
                scaled_local_weights.append(scale * local_weights[i])
            scaled_local_weights_list.append(scaled_local_weights)
        global_weights = []
        for grad_list_tuple in zip(*scaled_local_weights_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            global_weights.append(layer_mean)

    else:  # PyTorch averaging
        global_weights = copy.deepcopy(weights_list[0])
        global_count = sum(scaling_factors)
        for key in global_weights.keys():
            global_weights[key] *= scaling_factors[0]  # Scale first model's weights
            for i in range(1, len(weights_list)):
                global_weights[key] += weights_list[i][key] * scaling_factors[i]  # Scale other models' weights
            global_weights[key] = torch.div(global_weights[key], global_count)  # Normalize by total samples

    return global_weights
