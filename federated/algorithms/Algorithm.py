from abc import abstractmethod
import logging


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
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, dataset.is_binary_target)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            logging.info("{}-{}".format(res, client_metric.name))

def get_y(y_true_raw, y_pred_raw, is_binary_target):
    y_true = []
    y_pred = []

    for y_true_original_i, y_pred_original_i in zip(y_true_raw, y_pred_raw):
        if is_binary_target:
            y_pred_new_i = 0
            if y_pred_original_i[0] > 0.5:
                y_pred_new_i = 1
            y_pred.append(y_pred_new_i)
            y_true.append(y_true_original_i)
        else:
            y_pred.append(y_pred_original_i.argmax())
            y_true.append(y_true_original_i.argmax())

    return y_true, y_pred



def scale_weights(local_weights, local_count):
    scaled_local_weights = []
    for i in range(len(local_weights)):
        scaled_local_weights.append(local_count * local_weights[i])

    return scaled_local_weights
