import math
import logging
import numpy as np

from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.Identity import Identity
from federated.algorithms.drift.GlobalModels import GlobalModels
from federated.algorithms.drift.fed_drift import get_init_model, train_and_average, get_start_window
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged
from metrics.MetricFactory import get_metrics


class FairFedDriftIA(Algorithm):

    def __init__(self):
        self.metrics_clustering = [LossPrivileged(), LossUnprivileged()]
        self.thresholds = []
        self.window = None
        name = "FairFedDrift-IA"
        color = "black"
        marker = "o"
        super().__init__(name, color, marker)

    def set_specs(self, args):
        threshold_p = float(args.thresholds[0])
        threshold_up = float(args.thresholds[1])
        self.thresholds = [threshold_p, threshold_up]
        self.window = math.inf
        if args.window:
            self.window = int(args.window)
        super().set_subfolders("{}/window-{}/loss_p-{}/loss_up-{}".format(self.name, self.window, threshold_p, threshold_up))

    def perform_fl(self, seed, clients_data, dataset):
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        global_model = global_models.create_new_global_model(init_model)
        client_instances_identities = [[[Identity(global_model.identity.id, global_model.identity.name)] * len(clients_data[0][client_id][0])] for client_id in range(dataset.n_clients)]

        # Train with data from first timestep
        clients_data_models, n_clients_data_models = self.get_clients_data_from_models(
            global_models, client_instances_identities, clients_data
        )
        train_and_average(global_models, dataset, seed, 0, clients_data_models)
        exit()
        return

    def get_clients_data_from_models(self, global_models, clients_instances_identities, clients_data):
        clients_data_models = {}
        n_clients_data_models = {}

        for global_model in global_models.models:
            clients_data_model = {}
            n = 0
            for client_id, client_instances_identities in enumerate(clients_instances_identities):
                x_shape = list(clients_data[0][client_id][0].shape)
                y_shape = list(clients_data[0][client_id][1].shape)
                s_shape = list(clients_data[0][client_id][2].shape)
                x_shape[0] = 0
                y_shape[0] = 0
                s_shape[0] = 0
                client_data_x = np.empty(x_shape, dtype=np.float32)
                client_data_y = np.empty(y_shape, dtype=np.float32)
                client_data_s = np.empty(s_shape, dtype=np.float32)

                has_trained_model = False
                for timestep in range(get_start_window(client_instances_identities, self.window), len(client_instances_identities)):
                    x, y, s, _ = clients_data[timestep][client_id]
                    logging.info("Getting data of client {} for model {} on timestep {}".format(
                        client_id, global_model.identity.name, timestep
                    ))
                    for instance_id in range(len(client_instances_identities[timestep])):
                        if client_instances_identities[timestep][instance_id].id == global_model.identity.id:
                            has_trained_model = True
                            client_data_x = np.append(client_data_x, [x[instance_id]], axis=0)
                            client_data_y = np.append(client_data_y, [y[instance_id]], axis=0)
                            client_data_s = np.append(client_data_s, [s[instance_id]], axis=0)
                if has_trained_model:
                    clients_data_model[client_id] = [client_data_x, client_data_y, client_data_s]
                    logging.info("Model {} client {} with {} data instances".format(
                        global_model.identity.id, client_id, len(client_data_x))
                    )
                n += len(client_data_x)
            if clients_data_model:
                clients_data_models[global_model.identity.id] = clients_data_model
                n_clients_data_models[global_model.identity.id] = n

        return clients_data_models, n_clients_data_models
