import logging
import numpy as np

from federated.algorithms.Algorithm import Algorithm, average_weights, get_y
from federated.algorithms.Identity import Identity
from federated.model import NN_model
from metrics.MetricFactory import get_metrics


# Has access to client identities and clusters models based on that
class Oracle(Algorithm):

    def __init__(self):
        name = "Oracle"
        color = "red"
        marker = "x"
        super().__init__(name, color, marker)

    def perform_fl(self, seed, clients_data, dataset):
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        global_models, clients_identities = setup(seed, dataset)
        # Train with data from first timestep
        global_models = train_and_average(global_models, dataset, clients_data, 0, seed)

        for timestep in range(1, dataset.n_timesteps):
            for client_id in range(dataset.n_clients):
                drift_id = dataset.drift_ids[timestep - 1][client_id]
                clients_identities[client_id].append(Identity(drift_id, drift_id))
            test_models(global_models, clients_data, clients_metrics, dataset, timestep, clients_identities)
            global_models = train_and_average(global_models, dataset, clients_data, timestep, seed)

        return clients_metrics, clients_identities


def setup(seed, dataset):
    global_models = []
    for i in range(dataset.n_drifts):
        global_models.append(NN_model(dataset, seed))
    clients_identities = [[] for _ in range(dataset.n_clients)]

    return global_models, clients_identities


def train_and_average(global_models, dataset, clients_data, timestep, seed):
    for cround in range(dataset.n_rounds):
        local_weights_list = [[] for _ in range(len(global_models))]
        local_scales_list = [[] for _ in range(len(global_models))]
        for client_id in range(dataset.n_clients):
            # 1 - Get client data of each global model of all timesteps until timestep (inclusive)
            x_shape = list(clients_data[0][client_id][0].shape)
            y_shape = list(clients_data[0][client_id][1].shape)
            x_shape[0] = 0
            y_shape[0] = 0
            x = [np.empty(x_shape, dtype=np.float32) for _ in range(len(global_models))]  # create x, y for each drift (for each global model)
            y = [np.empty(y_shape, dtype=np.float32) for _ in range(len(global_models))]
            logging.info("Getting data of client {} until timestep {} (inclusive)".format(client_id, timestep))
            for t_line in range(timestep + 1):
                x_line, y_line, _, __ = clients_data[t_line][client_id]
                gm_id = dataset.drift_ids[t_line][client_id]
                logging.info("Getting data of client {}, timestep {}, identity {}".format(client_id, t_line, gm_id))
                x[gm_id] = np.concatenate([x[gm_id], x_line])
                y[gm_id] = np.concatenate([y[gm_id], y_line])

            for gm_id, (xx, yy) in enumerate(zip(x, y)):  # xx is data from a global model of client client_id
                if len(xx) > 0:
                    global_weights = global_models[gm_id].get_weights()
                    local_model = NN_model(dataset, seed)
                    local_model.compile(dataset)
                    local_model.set_weights(global_weights)
                    local_model.learn(xx, yy)
                    local_weights_list[gm_id].append(local_model.get_weights())
                    local_scales_list[gm_id].append(len(xx))
                    # K.clear_session()
                    logging.info("Trained timestep {} model {} cround {} client {}".format(timestep, gm_id, cround, client_id))
                else:
                    logging.info("Did not train timestep {} model {} cround {} client {}".format(timestep, gm_id, cround, client_id))
        logging.info("")
        for gm_id in range(len(global_models)):
            if len(local_weights_list[gm_id]) > 0:
                new_global_weights = average_weights(local_weights_list[gm_id], local_scales_list[gm_id])
                global_models[gm_id].set_weights(new_global_weights)
                logging.info("Averaged models on timestep {} cround {} of model {}".format(timestep, cround, gm_id))
            else:
                logging.info("Did not average models on timestep {} cround {} of model {}".format(timestep, cround, gm_id))
        logging.info("")

    return global_models

def test_models(global_models, clients_data, clients_metrics, dataset, timestep, clients_identities):
    for client_id, (client_data, client_metrics) in enumerate(zip(clients_data[timestep], clients_metrics)):
        x, y_true_raw, s, _ = client_data
        model = global_models[clients_identities[client_id][-1].id]
        y_pred_raw = model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, dataset.is_binary_target)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            logging.info("Client {}, timestep {}: {} - {}".format(client_id, timestep, res, client_metric.name))
