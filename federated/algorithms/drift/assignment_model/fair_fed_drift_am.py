import math
import logging
import numpy as np

from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.Identity import Identity
from federated.algorithms.drift.GlobalModels import GlobalModels
from federated.algorithms.drift.assignment_model.assignment_model import AssignmentModel
from federated.algorithms.drift.fed_drift import get_init_model, train_and_average, get_clients_data_from_models, \
    test_models, print_clients_identities, test_client_on_model
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged
from metrics.MetricFactory import get_metrics

WORST_LOSS = 1000


class FairFedDriftAM(Algorithm):

    def __init__(self):
        self.metrics_clustering = [LossPrivileged(), LossUnprivileged()]
        self.thresholds = []
        self.window = None
        name = "FairFedDrift-AM"
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
        clients_identities = [[Identity(global_model.identity.id, global_model.identity.name)] for _ in range(dataset.n_clients)]
        clients_identities_printing = [[Identity(global_model.identity.id, global_model.identity.name)] for _ in range(dataset.n_clients)]
        previous_loss_clients = [[[WORST_LOSS for _ in range(len(self.metrics_clustering))]] for _ in range(dataset.n_clients)]  # used for drift detection

        # STEP 0.1 - Train with data from first timestep
        clients_data_models, n_clients_data_models = get_clients_data_from_models(
            global_models, clients_identities, clients_data, self.window
        )
        global_models = train_and_average(global_models, dataset, seed, 0, clients_data_models)

        # STEP 0.2 - Train assignment model with data from first timestep
        assignment_model = AssignmentModel(seed)
        assignment_model.compile()
        assignment_model = self.train_assignment_model(assignment_model, clients_data)

        for timestep in range(1, dataset.n_timesteps - 1):
            logging.info("Current Global Models")
            for gm in global_models.models:
                logging.info("id: {}, name: {}".format(gm.identity.id, gm.identity.name))

            # STEP 1 - Test each client's data using previous identities
            logging.info("STEP 1 - test (timestep: {})".format(timestep))
            test_models(global_models, clients_data[timestep], clients_metrics, dataset, clients_identities, seed)

            # STEP 2 - Select most likely model or create new for each client
            logging.info("STEP 2 - Update (timestep: {})".format(timestep))
            global_models, clients_identities, previous_loss_clients, clients_new_models = self.update(
                clients_data[timestep], global_models, dataset, clients_identities, previous_loss_clients,
                assignment_model
            )

            # STEP 3 - Add models from drifted clients
            logging.info("STEP 3 - Add models from drifted clients (timestep: {})".format(timestep))
            for client_id in clients_new_models:
                new_global_model = global_models.create_new_global_model(get_init_model(dataset, seed))
                clients_identities[client_id].append(Identity(new_global_model.identity.id, new_global_model.identity.name))

            # TODO - STEP 4 - Merge Global Models
            # logging.info("STEP 4 - Merge (timestep: {})".format(timestep))
            # global_models, clients_identities = self.merge(clients_data, global_models, dataset, seed, clients_identities)

            # STEP 5.1 - Train and average global models with data from this timestep
            logging.info("STEP 5 - Train and average (timestep: {})".format(timestep))
            clients_data_models, n_clients_data_models = get_clients_data_from_models(
                global_models, clients_identities, clients_data, self.window
            )
            global_models = train_and_average(global_models, dataset, seed, timestep, clients_data_models)

            # TODO STEP 5.2 - Train and average assignment model with data (maybe not all data?) from this timestep

            for client_id in range(dataset.n_clients):
                clients_identities_printing[client_id].append(clients_identities[client_id][-1])

            logging.info("Clients identities (for model) (timestep: {})".format(timestep))
            print_clients_identities(clients_identities)
            # since clients identities change when merging we want to print originals
            logging.info("Clients identities (for printing) (timestep: {})".format(timestep))
            print_clients_identities(clients_identities_printing)

        # test on data from last timestep
        test_models(
            global_models, clients_data[dataset.n_timesteps - 1], clients_metrics, dataset, clients_identities, seed
        )

        return clients_metrics, clients_identities_printing

    def train_assignment_model(self, assignment_model, clients_data):
        timestep = 0  # for the first timestep, the concept is 0
        for client_id in range(len(clients_data[timestep])):
            x, y, _, __ = clients_data[timestep][client_id]
            x_batch = get_slices(x, assignment_model.num_pairs)
            y_batch = get_slices(y, assignment_model.num_pairs)
            ground_truth = np.zeros(((len(x) // assignment_model.num_pairs), assignment_model.num_classes), dtype=int)
            ground_truth[:, 0] = 1
            assignment_model.learn(x_batch, y_batch, ground_truth)

        return assignment_model

    def update(
        self, clients_data_timestep, global_models, dataset, clients_identities, previous_loss_clients,
        assignment_model
    ):
        clients_new_models = []  # client ids that drifted and created new global models

        for client_id, client_data in enumerate(clients_data_timestep):
            x, y = client_data[:2]
            x_batch = get_slices(x, assignment_model.num_pairs)
            y_batch = get_slices(y, assignment_model.num_pairs)
            pred = assignment_model.predict(x_batch, y_batch)  # get model from assignment model
            pred_ = np.argmax(pred, axis=1)
            row_sums = np.sum(pred, axis=0)
            model_id = np.argmax(row_sums)
            logging.info(pred)
            logging.info(pred_)
            logging.info(row_sums)
            logging.info(model_id)
            global_model_selected = global_models.get_model(model_id)
            results = test_client_on_model(
                self.metrics_clustering, global_model_selected.model, dataset.is_binary_target, client_data[:3]
            )  # client_data[:3] -> x, y, s
            previous_loss_clients[client_id].append(results)
            logging.info(results)

            drift_detected = False
            for loss, previous_loss, threshold in zip(results, previous_loss_clients[client_id][-1], self.thresholds):
                if loss > (threshold + previous_loss):
                    drift_detected = True

            if drift_detected:
                logging.info("Drift detected at client {}".format(client_id))
                clients_new_models.append(client_id)
            else:
                logging.info("No drift detected at client {}. Best global model id:{}, name{}".format(
                    client_id, global_model_selected.identity.id, global_model_selected.identity.name
                ))
                clients_identities[client_id].append(
                    Identity(global_model_selected.identity.id, global_model_selected.identity.name)
                )

        return global_models, clients_identities, previous_loss_clients, clients_new_models

    # TODO
    def merge(self, clients_data, global_models, dataset, seed, clients_identities):
        # return global_models, clients_identities
        pass


def get_slices(x, vector_size):
    return np.array([x[i * vector_size:(i + 1) * vector_size] for i in range(len(x) // vector_size)])
