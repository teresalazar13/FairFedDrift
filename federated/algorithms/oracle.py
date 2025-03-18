import logging
import time
import numpy as np
from federated.algorithms.Algorithm import Algorithm, average_weights, get_y
from federated.algorithms.Identity import Identity
from federated.nn_model import NN_model
from metrics.MetricFactory import get_metrics


# Has access to client identities and clusters models based on that
class Oracle(Algorithm):

    def __init__(self):
        name = "Oracle"
        color = "red"
        marker = "x"
        super().__init__(name, color, marker)

    def perform_fl(self, seed, clients_data, dataset):
        clients_metrics = [get_metrics(dataset.is_pt) for _ in range(dataset.n_clients)]
        global_models, clients_identities, gm_has_been_trained = setup(seed, dataset)
        # Train with data from first timestep
        global_models, gm_has_been_trained = train_and_average(global_models, dataset, clients_data, 0, gm_has_been_trained, seed)

        for timestep in range(1, dataset.n_timesteps):
            for client_id in range(dataset.n_clients):
                drift_id = dataset.drift_ids[timestep - 1][client_id]
                clients_identities[client_id].append(Identity(drift_id, drift_id))
            test_models(global_models, clients_data, clients_metrics, dataset, timestep, clients_identities)
            global_models, gm_has_been_trained = train_and_average(global_models, dataset, clients_data, timestep, gm_has_been_trained, seed)

        return clients_metrics, clients_identities


def setup(seed, dataset):
    global_models = []
    gm_has_been_trained = []

    for i in range(dataset.n_drifts):
        global_models.append(None)
        gm_has_been_trained.append(False)

    global_models[0] = NN_model(dataset, seed)
    gm_has_been_trained[0] = True
    clients_identities = [[] for _ in range(dataset.n_clients)]

    return global_models, clients_identities, gm_has_been_trained


def train_and_average(global_models, dataset, clients_data, timestep, gm_has_been_trained, seed):
    for cround in range(dataset.n_rounds):
        local_weights_list = [[] for _ in range(len(global_models))]
        local_scales_list = [[] for _ in range(len(global_models))]
        for client_id in range(dataset.n_clients):
            start = time.time()
            # Get client data of each global model of all timesteps until timestep (inclusive)
            x_shape = list(clients_data[0][client_id][0].shape)
            y_shape = list(clients_data[0][client_id][1].shape)
            x_shape[0] = 0
            y_shape[0] = 0
            # create x, y for each drift (for each global model id)
            x = [np.empty(x_shape, dtype=np.float32) for _ in range(len(global_models))]
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
                    if gm_has_been_trained[gm_id]:
                        global_weights = global_models[gm_id].get_weights()
                    else:
                        global_models[gm_id] = NN_model(dataset, seed)
                        global_weights = global_models[0].get_weights()
                        global_models[gm_id].set_weights(global_weights)
                        gm_has_been_trained[gm_id] = True
                        logging.info("First time on model {} - copying model 0".format(gm_id))
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
            time_taken = round(time.time() - start)
            logging.info("time {}s".format(time_taken))

        logging.info("")
        for gm_id in range(len(global_models)):
            if len(local_weights_list[gm_id]) > 0:
                new_global_weights = average_weights(dataset.is_pt, local_weights_list[gm_id], local_scales_list[gm_id])
                global_models[gm_id].set_weights(new_global_weights)
                logging.info("Averaged models on timestep {} cround {} of model {}".format(timestep, cround, gm_id))
            else:
                logging.info("Did not average models on timestep {} cround {} of model {}".format(timestep, cround, gm_id))
        logging.info("")

    return global_models, gm_has_been_trained


def test_models(global_models, clients_data, clients_metrics, dataset, timestep, clients_identities):
    for client_id, (client_data, client_metrics) in enumerate(zip(clients_data[timestep], clients_metrics)):
        x, y_true_raw, s, _ = client_data
        model = global_models[clients_identities[client_id][-1].id]
        y_pred_raw = model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, dataset.n_classes)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            logging.info("Client {}, timestep {}: {} - {}".format(client_id, timestep, res, client_metric.name))
