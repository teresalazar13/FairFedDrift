import math
import logging
import numpy as np

from federated.algorithms.Algorithm import Algorithm, average_weights
from federated.algorithms.Identity import Identity
from federated.algorithms.drift.GlobalModels import GlobalModels
from federated.algorithms.drift.assignment_model.assignment_model import AssignmentModel
from federated.algorithms.drift.fed_drift import get_init_model, train_and_average, get_clients_data_from_models, \
    test_models, print_clients_identities, test_client_on_model, print_matrix, merge_global_models, get_start_window
from metrics.LossPrivileged import LossPrivileged
from metrics.LossUnprivileged import LossUnprivileged
from metrics.MetricFactory import get_metrics

WORST_LOSS = 1000
WORST_DISTANCE_THRESHOLD = -1
DISTANCE_THRESHOLD = 0.9  # TODO - this should be hyperparameter -> DO THIS NOW


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
        super().set_subfolders(
            "{}/window-{}/loss_p-{}/loss_up-{}".format(self.name, self.window, threshold_p, threshold_up))

    def perform_fl(self, seed, clients_data, dataset):
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        global_model = global_models.create_new_global_model(init_model)
        clients_identities = [[Identity(global_model.identity.id, global_model.identity.name)] for _ in
                              range(dataset.n_clients)]
        clients_identities_printing = [[Identity(global_model.identity.id, global_model.identity.name)] for _ in
                                       range(dataset.n_clients)]
        previous_loss_clients = [[[WORST_LOSS for _ in range(len(self.metrics_clustering))]] for _ in
                                 range(dataset.n_clients)]  # used for drift detection

        # STEP 0.1 - Train with data from first timestep
        clients_data_models, _ = get_clients_data_from_models(
            global_models, clients_identities, clients_data, self.window
        )
        global_models = train_and_average(global_models, dataset, seed, 0, clients_data_models)

        # STEP 0.2 - Train assignment model with data from first timestep
        assignment_model = get_init_assignment_model(seed)
        client_ids_drifted_gmid = {client_id: 0 for client_id in range(dataset.n_clients)}
        assignment_model = train_and_average_assignment_model(assignment_model, seed, 0, clients_data, client_ids_drifted_gmid)

        for timestep in range(1, dataset.n_timesteps - 1):
            logging.info("Current Global Models")
            for gm in global_models.models:
                logging.info("id: {}, name: {}".format(gm.identity.id, gm.identity.name))

            # STEP 1 - Test each client's data using previous identities
            logging.info("STEP 1 - test (timestep: {})".format(timestep))
            test_models(global_models, clients_data[timestep], clients_metrics, dataset, clients_identities, seed)

            # STEP 2 - Select most likely model or create new for each client
            logging.info("STEP 2 - Update (timestep: {})".format(timestep))
            global_models, clients_identities, previous_loss_clients, client_ids_drifted, clients_assignment_probs = \
                self.update(
                    clients_data[timestep], global_models, dataset, clients_identities, previous_loss_clients,
                    assignment_model
                )

            # STEP 3 - Add models from drifted clients
            logging.info("STEP 3 - Add models from drifted clients (timestep: {})".format(timestep))
            client_ids_drifted_gmid = {}
            for client_id in client_ids_drifted:
                new_global_model = global_models.create_new_global_model(get_init_model(dataset, seed))
                client_ids_drifted_gmid[client_id] = new_global_model.identity.id
                clients_identities[client_id].append(
                    Identity(new_global_model.identity.id, new_global_model.identity.name))
                assignment_model.add_class()

            # STEP 4 - Merge Global Models
            #logging.info("STEP 4 - Merge (timestep: {})".format(timestep))
            #global_models, clients_identities = self.merge(
            #    clients_data, global_models, dataset, seed, clients_identities, clients_assignment_probs
            #)

            # STEP 5.1 - Train and average global models with data from this timestep
            logging.info("STEP 5 - Train and average Global Models (timestep: {})".format(timestep))
            clients_data_models, _ = get_clients_data_from_models(
                global_models, clients_identities, clients_data, self.window
            )
            global_models = train_and_average(global_models, dataset, seed, timestep, clients_data_models)

            # STEP 5.2.1 - Train and average assignment model with data from drifted clients
            if len(client_ids_drifted_gmid.keys()) > 0:
                logging.info(
                    "STEP 5.2.1 - Train and average Assignment Model drifted clients (timestep: {})".format(timestep)
                )
                assignment_model = train_and_average_assignment_model(assignment_model, seed, timestep, clients_data, client_ids_drifted_gmid)
            else:
                logging.info(
                    "STEP 5.2.1 - No drifted clients (timestep: {}) - NOT Training and averaging Assignment Model"
                    .format(timestep)
                )

            # TODO STEP 5.2.2 - Train and average assignment model with data from merged clients

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

    def update(
            self, clients_data_timestep, global_models, dataset, clients_identities, previous_loss_clients,
            assignment_model
    ):
        client_ids_drifted = []  # client ids that drifted and created new global models
        clients_assignment_probs = []  # probabilities of clients belonging to models

        for client_id, client_data in enumerate(clients_data_timestep):
            x, y = client_data[:2]
            x_batch = get_slices(x, assignment_model.num_pairs)
            y_batch = get_slices(y, assignment_model.num_pairs)
            pred = assignment_model.predict(x_batch, y_batch)  # get model from assignment model
            pred_ = np.argmax(pred, axis=1)
            row_sums = np.sum(pred, axis=0)
            clients_assignment_probs.append(row_sums)
            model_id = np.argmax(row_sums)
            logging.info(pred)
            logging.info(pred_)
            logging.info(row_sums)
            logging.info(model_id)
            global_model_selected = global_models.get_model(model_id)
            results = test_client_on_model(
                self.metrics_clustering, global_model_selected.model, dataset.is_binary_target, client_data[:3]
            )  # client_data[:3] -> x, y, s
            logging.info(results)

            drift_detected = False
            for loss, previous_loss, threshold in zip(results, previous_loss_clients[client_id][-1], self.thresholds):
                logging.info("Loss: {}, Previous Loss: {}, Threshold: {}".format(loss, previous_loss, threshold))
                if loss > (threshold + previous_loss):
                    drift_detected = True
            previous_loss_clients[client_id].append(results)

            if drift_detected:
                logging.info("Drift detected at client {}".format(client_id))
                client_ids_drifted.append(client_id)
            else:
                logging.info("No drift detected at client {}. Best global model id:{}, name{}".format(
                    client_id, global_model_selected.identity.id, global_model_selected.identity.name
                ))
                clients_identities[client_id].append(
                    Identity(global_model_selected.identity.id, global_model_selected.identity.name)
                )

        return global_models, clients_identities, previous_loss_clients, client_ids_drifted, clients_assignment_probs

    # TODO - update assignment model when merging
    def merge(self, clients_data, global_models, dataset, seed, clients_identities, clients_assignment_probs):
        size = global_models.n_models
        if size > 30:
            logging.info("Number of global models > 30")
            exit(1)

        n_clients_data_models = self.get_n_clients_data_from_models(global_models, clients_identities, clients_data)
        logging.info(n_clients_data_models)
        if len(n_clients_data_models.keys()) == 1:
            return global_models, clients_identities

        distances = calculate_cosine_similarity_matrix(clients_assignment_probs)
        while True:  # While we can still merge global models
            print_matrix(distances)
            id_0, id_1, found = get_closest_models(distances)
            if found:
                global_models, distances, clients_identities, n_clients_data_models = merge_global_models(
                    dataset, seed, global_models, id_0, id_1, distances, clients_identities, n_clients_data_models,
                    self.window
                )
            else:
                return global_models, clients_identities

    def get_n_clients_data_from_models(self, global_models, clients_identities, clients_data):
        n_clients_data_models = {}

        for global_model in global_models.models:
            n = 0
            for client_id, client_identities in enumerate(clients_identities):
                has_trained_model = False
                n_client_data = 0
                for timestep in range(get_start_window(client_identities, self.window), len(client_identities)):
                    if client_identities[timestep].id == global_model.identity.id:
                        logging.info("Getting data of client {} for model {} on timestep {}".format(
                            client_id, global_model.identity.name, timestep
                        ))
                        has_trained_model = True
                        x, y, s, _ = clients_data[timestep][client_id]
                        n_client_data += len(x)
                if has_trained_model:
                    logging.info("Model {} client {} with {} data instances".format(
                        global_model.identity.id, client_id, n_client_data)
                    )
                n += n_client_data
            if n > 0:
                n_clients_data_models[global_model.identity.id] = n

        return n_clients_data_models


def calculate_cosine_similarity_matrix(clients_assignment_probs):
    clients_assignment_probs = np.array(clients_assignment_probs)
    num_models = clients_assignment_probs.shape[1]
    similarity_matrix = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(num_models):
            v1 = clients_assignment_probs[:, i]
            v2 = clients_assignment_probs[:, j]
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            similarity_matrix[i, j] = dot_product / (norm_v1 * norm_v2)

    return similarity_matrix


def get_closest_models(distances):
    best_row = None
    best_col = None
    best_result = WORST_DISTANCE_THRESHOLD

    for row in range(len(distances)):
        for col in range(len(distances[row])):
            result = distances[row][col]
            if result > best_result:
                best_result = result
                best_row = row
                best_col = col

    if best_result >= DISTANCE_THRESHOLD:
        return best_row, best_col, True
    else:
        return None, None, False


def get_slices(x, vector_size):
    return np.array([x[i * vector_size:(i + 1) * vector_size] for i in range(len(x) // vector_size)])


def train_and_average_assignment_model(assignment_model, seed, timestep, clients_data, client_ids_drifted_gmid):
    for cround in range(assignment_model.n_rounds):
        local_weights = []
        local_scales = []

        for client_id, gm_id in client_ids_drifted_gmid.items():
            x, y, _, __ = clients_data[timestep][client_id]
            x_batch = get_slices(x, assignment_model.num_pairs)
            y_batch = get_slices(y, assignment_model.num_pairs)
            ground_truth = np.zeros(((len(x) // assignment_model.num_pairs), assignment_model.num_classes), dtype=int)
            ground_truth[:, gm_id] = 1
            local_assignment_model = get_assign_model_copy(assignment_model, seed)
            local_assignment_model.learn(x_batch, y_batch, ground_truth)
            logging.info(
                "Trained assignment model timestep {} cround {} on client {} with assignment (gm id) {}"
                .format(timestep, cround, client_id, gm_id)
            )
            local_weights.append(local_assignment_model.get_weights())
            local_scales.append(len(x))

        new_global_weights = average_weights(local_weights, local_scales)
        assignment_model.set_weights(new_global_weights)
        logging.info("Averaged local assignment models on timestep {} cround {}".format(timestep, cround))

    return assignment_model


def get_assign_model_copy(assignment_model, seed):
    weights = assignment_model.get_weights()
    model = get_init_assignment_model(seed)
    model.set_weights(weights)

    return model


def get_init_assignment_model(seed):
    model = AssignmentModel(seed)
    model.compile()

    return model
