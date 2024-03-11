import math
import numpy as np

from federated.algorithms.Algorithm import Algorithm, get_y, average_weights
from federated.algorithms.Identity import Identity
from federated.algorithms.drift.GlobalModels import GlobalModels
from federated.model import NN_model
from metrics.Loss import Loss
from metrics.MetricFactory import get_metrics
import logging

WORST_LOSS = 1000
BEST_LOSS = 0


class FedDrift(Algorithm):

    def __init__(self):
        self.metrics_clustering = [Loss()]
        self.thresholds = []
        self.window = None
        name = "FedDrift"
        color = "green"
        marker = "v"
        super().__init__(name, color, marker)

    def set_specs(self, args):
        loss_threshold = float(args.thresholds[0])
        self.thresholds = [loss_threshold]
        window = math.inf
        if args.window:
            window = args.window
        self.window = window
        super().set_subfolders("{}/window-{}/loss-{}".format(self.name, window, loss_threshold))

    def perform_fl(self, seed, clients_data, dataset):
        return perform_fl(self, seed, clients_data, dataset)


def perform_fl(self, seed, clients_data, dataset):
    clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
    global_models = GlobalModels()
    init_model = get_init_model(dataset, seed)
    global_model = global_models.create_new_global_model(init_model)
    clients_identities = [[Identity(global_model.identity.id, global_model.identity.name)] for _ in range(dataset.n_clients)]
    clients_identities_printing = [[Identity(global_model.identity.id, global_model.identity.name)] for _ in range(dataset.n_clients)]
    previous_loss_clients = [[[WORST_LOSS for _ in range(len(self.metrics_clustering))]] for _ in range(dataset.n_clients)]  # used for drift detection

    # Train with data from first timestep
    global_models = train_and_average(clients_data, global_models, dataset, seed, 0, clients_identities)

    for timestep in range(1, dataset.n_timesteps - 1):
        logging.info("Minimum Loss Clients")
        print_matrix(previous_loss_clients)
        logging.info("Current Global Models")
        for gm in global_models.models:
            logging.info("id: {}, name: {}".format(gm.identity.id, gm.identity.name))

        # STEP 1 - Test each client's data using previous identities
        logging.info("STEP 1 - test (timestep: {})".format(timestep))
        test_models(global_models, clients_data[timestep], clients_metrics, dataset, clients_identities, seed)

        # STEP 2 - Select best Models or create new for each client
        logging.info("STEP 2 - Update (timestep: {})".format(timestep))
        global_models, clients_identities, previous_loss_clients, clients_new_models = update(
            self.metrics_clustering, self.thresholds, clients_data[timestep], global_models, dataset, clients_identities,
            previous_loss_clients
        )

        # STEP 3 - Add models from drifted clients
        logging.info("STEP 3 - Add models from drifted clients (timestep: {})".format(timestep))
        for client_id in clients_new_models:
            new_global_model = global_models.create_new_global_model(get_init_model(dataset, seed))
            clients_identities[client_id].append(Identity(new_global_model.identity.id, new_global_model.identity.name))

        # STEP 4 - Merge Global Models
        logging.info("STEP 4 - Merge (timestep: {})".format(timestep))
        global_models, clients_identities = merge(
            self.metrics_clustering, self.thresholds, clients_data, global_models, dataset, seed, clients_identities
        )

        # STEP 5 - Train and average models with data from this timestep
        logging.info("STEP 5 - Train and average (timestep: {})".format(timestep))
        global_models = train_and_average(clients_data, global_models, dataset, seed, timestep, clients_identities)
        for client_id in range(dataset.n_clients):
            clients_identities_printing[client_id].append(clients_identities[client_id][-1])

        logging.info("Clients identities (for model) (timestep: {})".format(timestep))
        print_clients_identities(clients_identities)
        # since clients identities change when merging we want to print originals
        logging.info("Clients identities (for printing) (timestep: {})".format(timestep))
        print_clients_identities(clients_identities_printing)

    # test on data from last timestep
    test_models(global_models, clients_data[dataset.n_timesteps - 1], clients_metrics, dataset, clients_identities, seed)

    return clients_metrics, clients_identities_printing


def get_init_model(dataset, seed):
    model = NN_model(dataset, seed)
    model.compile(dataset)

    return model


def test_models(global_models, clients_data_timestep, clients_metrics, dataset, clients_identities, seed):
    for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
        x, y_true_raw, s, _ = client_data
        global_model_id = clients_identities[client_id][-1].id
        model = get_model_copy(global_models, global_model_id, dataset, seed)
        y_pred_raw = model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, dataset.is_binary_target)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            logging.info("Client {}: {} - {}".format(client_id, res, client_metric.name))


def update(
    metrics_clustering, thresholds, clients_data_timestep, global_models, dataset, clients_identities,
    previous_loss_clients
):
    clients_new_models = []
    for client_id, client_data in enumerate(clients_data_timestep):
        # Calculate results on all global models
        results_global_models = {}
        for global_model in global_models.models:
            results = test_client_on_model(metrics_clustering, global_model.model, dataset.is_binary_target, client_data[:3])  # client_data[:2] -> x, y, s
            results_global_models[global_model] = results

        # Get Model for client given results_global_models
        best_global_model, best_results = get_best_model(results_global_models, previous_loss_clients[client_id], thresholds)
        previous_loss_clients[client_id].append(best_results)
        if best_global_model:
            logging.info("No drift detected at client {}. Best global model id:{}, name{}".format(
                client_id, best_global_model.identity.id, best_global_model.identity.name
            ))
            clients_identities[client_id].append(Identity(best_global_model.identity.id, best_global_model.identity.name))
        else:
            logging.info("Drift detected at client {}".format(client_id))
            clients_new_models.append(client_id)

    return global_models, clients_identities, previous_loss_clients, clients_new_models


def get_best_model(model_results_dict, previous_loss_client, thresholds):
    best_model = None  # if there is not a drift
    minimum_loss_client_new_a = None  # if there is not a drift
    minimum_loss_client_new_b = None # if there is a drift (no models where loss < previous_loss + threshold)

    for model, results in model_results_dict.items():
        drift_detected = False
        for loss, previous_loss, threshold in zip(results, previous_loss_client[-1], thresholds):
            if loss > (threshold + previous_loss):  # if drift detected
                drift_detected = True

        if not drift_detected:
            if not minimum_loss_client_new_a or sum(results) < sum(minimum_loss_client_new_a):
                best_model = model
                minimum_loss_client_new_a = results
        else:
            if not minimum_loss_client_new_b or sum(results) < sum(minimum_loss_client_new_b):
                minimum_loss_client_new_b = results

    if minimum_loss_client_new_a:
        return best_model, minimum_loss_client_new_a
    else:
        return None, minimum_loss_client_new_b


def test_client_on_model(metrics_clustering, model, is_binary_target, client_data):
    x, y_true_raw, s = client_data
    y_pred_raw = model.predict(x)
    y_true, y_pred = get_y(y_true_raw, y_pred_raw, is_binary_target)

    results = []
    for metric_clustering in metrics_clustering:
        result = metric_clustering.calculate(y_true, y_pred, y_true_raw, y_pred_raw, s)
        results.append(result)

    return results


def merge(metrics_clustering, thresholds, clients_data, global_models, dataset, seed, clients_identities):
    size = global_models.n_models
    if size > 30:
        logging.info("Number of global models > 30")
        raise Exception("Number of global models > 30")
    distances = [[WORST_LOSS for _ in range(size)] for __ in range(size)]
    clients_data_models, n_clients_data_models = get_clients_data_from_models(
        global_models, [c[:-1] for c in clients_identities], clients_data  # do not include data from current timestep, (hence [:-1])
    )
    model_ids = list(n_clients_data_models.keys())

    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            id_i = model_ids[i]
            id_j = model_ids[j]
            if id_i in n_clients_data_models and id_j in n_clients_data_models:
                model_i = global_models.get_model(id_i)
                model_j = global_models.get_model(id_j)
                logging.info("Models [id:{},name:{}] and [id:{},name:{}] have been trained. Checking distances".format(
                    id_i, model_i.identity.name, id_j, model_j.identity.name)
                )
                Lij = get_losses_of_model_on_data(model_i, model_j, dataset, metrics_clustering, clients_data_models[id_j])
                Lii = get_losses_of_model_on_data(model_i, model_i, dataset, metrics_clustering, clients_data_models[id_i])
                Lji = get_losses_of_model_on_data(model_j, model_i, dataset, metrics_clustering, clients_data_models[id_i])
                Ljj = get_losses_of_model_on_data(model_j, model_j, dataset, metrics_clustering, clients_data_models[id_j])
                # Calculate maximum difference between constituents of different clusters
                max_L1 = calculate_maximum_distance(Lij, Lii, thresholds)  #max_L1 -> max Lij-Lii
                max_L2 = calculate_maximum_distance(Lji, Ljj, thresholds)  #max_L2 -> max Lji-Ljj

                d = max(max_L1, max_L2, 0)
                logging.info("Distances between models {} and {} are: {}".format(id_i, id_j, d))
                distances[id_i][id_j] = d
                distances[id_j][id_i] = d

    while True:  # While we can still merge global models
        print_matrix(distances)
        id_0, id_1, found = get_next_best_results(distances)
        if found:
            global_models, distances, clients_identities, n_clients_data_models = merge_global_models(
                dataset, seed, global_models, id_0, id_1, distances, clients_identities, n_clients_data_models
            )
        else:
            return global_models, clients_identities


# Calculate maximum difference between constituents of different clusters
# max_L1 -> max Lij-Lii
# max_L2 -> max Lji-Ljj
def calculate_maximum_distance(L1, L2, thresholds):
    max_L = 0
    for l1 in L1:
        for l2 in L2:
            distance = 0
            for l1res, l2res, threshold in zip(l1, l2, thresholds):
                if l1res - l2res > threshold:
                    return WORST_LOSS
                else:
                    distance += l1res - l2res
            if distance > max_L:
                max_L = distance

    return max_L


# Get clients data that belong to models (and number of points they have trained)
def get_clients_data_from_models(global_models, clients_identities, clients_data):
    clients_data_models = {}
    n_clients_data_models = {}

    for global_model in global_models.models:
        clients_data_model = {}
        n = 0
        for client_id, client_identities in enumerate(clients_identities):
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
            for timestep, client_identity in enumerate(client_identities):
                if client_identity.id == global_model.identity.id:
                    logging.info("Getting data of client {} for model {} on timestep {}".format(
                        client_id, global_model.identity.name, timestep
                    ))
                    has_trained_model = True
                    x, y, s, _ = clients_data[timestep][client_id]
                    client_data_x = np.concatenate([client_data_x, x])
                    client_data_y = np.concatenate([client_data_y, y])
                    client_data_s = np.concatenate([client_data_s, s])
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


def get_losses_of_model_on_data(global_model_model, global_model_data, dataset, metrics_clustering, clients_data_model):
    losses = []

    for client_id, client_data in clients_data_model.items():
        logging.info("Getting losses model {} on data of client {} trained on model {} ".format(
            global_model_model.identity.id, client_id, global_model_data.identity.id
        ))
        results = test_client_on_model(
            metrics_clustering, global_model_model.model, dataset.is_binary_target, client_data
        )
        logging.info("Results: {}".format(results))
        losses.append(results)

    return losses


def get_model_copy(global_models, global_model_id, dataset, seed):
    global_model = global_models.get_model(global_model_id)
    weights = global_model.model.get_weights()
    model = get_init_model(dataset, seed)
    model.set_weights(weights)

    return model


def merge_global_models(
    dataset, seed, global_models, id_0, id_1, distances, clients_identities, n_clients_data_models
):
    global_model_0 = global_models.get_model(id_0)
    global_model_1 = global_models.get_model(id_1)
    scales = [n_clients_data_models[id_0], n_clients_data_models[id_1]]
    weights = [global_model_0.model.get_weights(), global_model_1.model.get_weights()]
    new_global_model_weights = average_weights(weights, scales)
    new_global_model = get_init_model(dataset, seed)
    new_global_model.set_weights(new_global_model_weights)

    # Create new global Model
    new_global_model_created = global_models.create_new_global_model(
        new_global_model, global_model_0.identity.name, global_model_1.identity.name
    )
    logging.info("Created global model id:{}, name:{} from models [id:{}, name:{}] and [id:{}, name:{}]".format(
        new_global_model_created.identity.id, new_global_model_created.identity.name,
        global_model_0.identity.id, global_model_0.identity.name,
        global_model_1.identity.id, global_model_1.identity.name
    ))

    for client_id, client_identities in enumerate(clients_identities):
        for timestep, identity in enumerate(client_identities):
            if identity.id == id_0 or identity.id == id_1:
                clients_identities[client_id][timestep] = new_global_model_created.identity
                logging.info("Changed identity of client {} timestep {} to [id:{},name:{}] (previous[id:{},name:{}])".format(
                    client_id, timestep, new_global_model_created.identity.id, new_global_model_created.identity.name,
                    identity.id, identity.name,
                ))

    # Create new column and row for new model id and update distances
    logging.info("Updating Distances")
    new_row = []
    for i in range(len(distances)):
        new_row.append(max(distances[id_0][i], distances[id_1][i]))
    for i in range(len(distances)):
        distances[i].append(new_row[i])
    new_row.append(WORST_LOSS)
    distances.append(new_row)

    # Reset Distances of deleted models
    distances[id_0][id_1] = WORST_LOSS
    distances[id_1][id_0] = WORST_LOSS
    for i in range(len(distances)):
        distances[id_0][i] = WORST_LOSS
        distances[id_1][i] = WORST_LOSS
        distances[i][id_0] = WORST_LOSS
        distances[i][id_1] = WORST_LOSS

    global_models.deleted_merged_model(id_0)
    global_models.deleted_merged_model(id_1)

    # Update n_clients_data_models
    logging.info("Old n_clients_data_models: {}".format(n_clients_data_models))
    n_clients_data_models[new_global_model_created.identity.id] = n_clients_data_models[id_0] + n_clients_data_models[id_1]
    n_clients_data_models[id_0] = 0
    n_clients_data_models[id_1] = 0
    logging.info("New n_clients_data_models: {}".format(n_clients_data_models))

    return global_models, distances, clients_identities, n_clients_data_models


def get_next_best_results(results_matrix):
    best_row = None
    best_col = None
    best_result = WORST_LOSS

    for row in range(len(results_matrix)):
        for col in range(len(results_matrix[row])):
            result = results_matrix[row][col]
            if result < best_result:
                best_result = result
                best_row = row
                best_col = col

    if best_result != WORST_LOSS:
        return best_row, best_col, True
    else:
        return None, None, False


def train_and_average(clients_data, global_models, dataset, seed, timestep, clients_identities):
    clients_data_models, n_clients_data_models = get_clients_data_from_models(
        global_models, clients_identities, clients_data
    )

    for cround in range(dataset.n_rounds):
        local_weights_list = [[] for _ in range(global_models.n_models)]
        local_scales_list = [[] for _ in range(global_models.n_models)]

        for global_model_id, clients_data in clients_data_models.items():
            for client_id, client_data in clients_data.items():
                x, y, _ = client_data
                local_model = get_model_copy(global_models, global_model_id, dataset, seed)
                local_model.learn(x, y)
                logging.info("Trained model {} timestep {} cround {} client {}".format(
                    global_model_id, timestep, cround, client_id)
                )
                local_weights_list[global_model_id].append(local_model.get_weights())
                local_scales_list[global_model_id].append(len(x))

        for global_model_id, (local_weights, local_scales) in enumerate(zip(local_weights_list, local_scales_list)):
            if len(local_weights) > 0:
                new_global_weights = average_weights(local_weights, local_scales)
                global_models.get_model(global_model_id).model.set_weights(new_global_weights)
                logging.info("Averaged models on timestep {} cround {} of cluster {}".format(timestep, cround, global_model_id))
            else:
                logging.info("Did not average models on timestep {} cround {} of cluster {}".format(timestep, cround, global_model_id))

    return global_models


def print_matrix(matrix):
    for row in matrix:
        if type(row[0]) == list:
            logging.info(' '.join('{:10}'.format('-'.join(str(i) for i in item)) for item in row))
        else:
            logging.info(' '.join('{:10}'.format(item) for item in row))


def print_clients_identities(clients_identities):
    logging.info("\nClients identities")
    string = "Clients identities\n"

    for timestep in range(len(clients_identities[0])):
        logging.info("Timestep {}".format(timestep))
        string += "\nTimestep " + str(timestep) + "\n"

        identities = {}
        for client in range(len(clients_identities)):
            client_identity_timestep = clients_identities[client][timestep].name
            if client_identity_timestep in identities:
                identities[client_identity_timestep].append(client)
            else:
                identities[client_identity_timestep] = [client]

        for model_name, clients in identities.items():
            logging.info("Model name {}: {}".format(model_name, clients))
            string += "Model name " + model_name + ":" + ",".join([str(c) for c in clients]) + "\n"

    return string
