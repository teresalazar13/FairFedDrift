from abc import abstractmethod
import tensorflow as tf


class Algorithm:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def perform_fl(self, seed, clients_data, dataset):
        raise NotImplementedError("Must implement perform_fl")

    def set_specs(self, args):
        pass

    def test(self, clients_data_timestep, clients_metrics, global_model, dataset):
        for client_data, client_metrics in zip(clients_data_timestep, clients_metrics):
            x, y, s, _ = client_data
            pred = global_model.predict(x)
            y, pred = get_y(y, pred, dataset.is_image)
            for client_metric in client_metrics:
                res = client_metric.update(y, pred, s)
                print(res, client_metric.name)

    def get_y(self, y, pred, is_image):
        y_new = []
        pred_new = []

        for y_i, pred_i in zip(y, pred):
            if not is_image:
                pred_new_i = 0
                if pred_i[0] > 0.5:
                    pred_new_i = 1
                pred_new.append(pred_new_i)
                y_new.append(y_i)
            else:
                y_new.append(y_i.argmax())
                pred_new.append(pred_i.argmax())

        return y_new, pred_new

    def average_weights(self, weights_list, scaling_factors):
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

        return global_weights
