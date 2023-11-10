from federated.algorithms.Algorithm import Algorithm, get_y, average_weights
from federated.algorithms.fair_fed_drift.ClientData import ClientData
from federated.algorithms.fair_fed_drift.ClientIdentity import ClientIdentity
from federated.algorithms.fair_fed_drift.GlobalModels import GlobalModels
from federated.model import NN_model
from metrics.Loss import Loss
from metrics.MetricFactory import get_metrics

WORST_LOSS = 1000
BEST_LOSS = 0


class FedDrift(Algorithm):

    def __init__(self):
        self.metrics_clustering = [Loss()]
        self.thresholds = []
        name = "FedDrift"
        super().__init__(name)

    def set_specs(self, args):
        loss_threshold = float(args.thresholds[0])
        self.thresholds = [loss_threshold]
        super().set_subfolders("{}/loss-{}".format(self.name, loss_threshold))

    def perform_fl(self, seed, clients_data, dataset):
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        global_model = global_models.create_new_global_model(init_model)
        clients_identities = [[ClientIdentity(global_model.id, global_model.name)] for _ in range(dataset.n_clients)]
        minimum_loss_clients = [[[WORST_LOSS for _ in range(len(self.metrics_clustering))]] for _ in range(dataset.n_clients)]  # used for drift detection

        for timestep in range(dataset.n_timesteps):
            print_clients_identities(clients_identities)
            print_matrix(minimum_loss_clients)

            # STEP 4 - Train and average models with data from this timestep
            global_models = train_and_average(
                clients_data[timestep], global_models, dataset, seed, timestep, clients_identities
            )
            set_clients(global_models, clients_identities, clients_data[timestep])  # to be used in merge for sampling

            timestep_to_test = timestep + 1
            if timestep_to_test == dataset.n_timesteps:
                timestep_to_test = 0

            # STEP 1 - Test each client's data on data from next timestep
            test_models(
                global_models, clients_data[timestep_to_test], clients_metrics, dataset, clients_identities, seed
            )

            if timestep != dataset.n_timesteps - 1:
                # STEP 2 - Recalculate Global Models using data from next timestep
                # (cluster identities for next timestep)
                # and detect concept drift
                global_models, global_models_to_create, clients_identities, minimum_loss_clients = update_global_models(
                    self.metrics_clustering, self.thresholds, clients_data[timestep_to_test], global_models, dataset,
                    seed, clients_identities, minimum_loss_clients
                )

                # STEP 3 - Merge Global Models
                global_models = merge_global_models(self.metrics_clustering, self.thresholds, global_models, dataset, seed)

                # STEP 2.1 - create global models from drifted clients
                # (has to be after merge - don't want to try to merge drifted clients)
                global_models, clients_identities = create_global_models_drifted_clients(
                    global_models, global_models_to_create, clients_identities
                )

        clients_identities_string = print_clients_identities(clients_identities)

        return clients_metrics, clients_identities, clients_identities_string


def get_init_model(dataset, seed):
    model = NN_model(dataset, seed)
    model.compile(dataset)

    return model


def test_models(global_models, clients_data_timestep, clients_metrics, dataset, clients_identities, seed):
    for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
        x, y_true_raw, s, _ = client_data
        global_model_id = clients_identities[client_id][-1].id
        model = get_model_client(global_models, global_model_id, dataset, seed)
        y_pred_raw = model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, dataset.is_binary_target)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            print(res, client_metric.name)


def print_clients_identities(clients_identities):
    print("\nClients identities")
    string = "Clients identities\n"

    for timestep in range(len(clients_identities[0])):
        print("\nTimestep ", timestep)
        string += "\nTimestep " + str(timestep) + "\n"

        identities = {}
        for client in range(len(clients_identities)):
            client_identity_timestep = clients_identities[client][timestep].name
            if client_identity_timestep in identities:
                identities[client_identity_timestep].append(client)
            else:
                identities[client_identity_timestep] = [client]

        for model_id, clients in identities.items():
            print("Model id: ", model_id, ":", clients)
            string += "Model id: " + model_id + ":" + ",".join([str(c) for c in clients]) + "\n"

    return string


def get_model_client(global_models, global_model_id, dataset, seed):
    global_model = global_models.get_model(global_model_id)
    weights = global_model.model.get_weights()
    model = get_init_model(dataset, seed)
    model.set_weights(weights)

    return model


def update_global_models(
        metrics_clustering, thresholds, clients_data_timestep, global_models, dataset, seed, clients_identities,
        minimum_loss_clients
):
    global_models_to_create = []

    for client_id, client_data_raw in enumerate(clients_data_timestep):
        x, y, s, _ = client_data_raw
        client_data = ClientData(x, y, s)

        # Calculate results on all global models
        results_global_models = {}
        for global_model in global_models.models:
            results = test_client_on_model(metrics_clustering, global_model.model, client_data, dataset.is_binary_target)
            results_global_models[global_model] = results

        # Get Model for client given results_global_models
        best_global_model, best_results = get_best_model(results_global_models, minimum_loss_clients[client_id], thresholds)
        minimum_loss_clients[client_id].append(best_results)
        if best_global_model:
            print("No drift detected at client {}".format(client_id))
            clients_identities[client_id].append(ClientIdentity(best_global_model.id, best_global_model.name))
        else:
            print("Drift detected at client {}".format(client_id))
            model = get_init_model(dataset, seed)
            global_models_to_create.append([model, client_id, client_data])

    return global_models, global_models_to_create, clients_identities, minimum_loss_clients


def set_clients(global_models, clients_identities, clients_data_timestep):
    for client_id in range(len(clients_identities)):
        global_model_id = clients_identities[client_id][-1].id
        client_data_raw = clients_data_timestep[client_id]
        x, y, s, _ = client_data_raw
        client_data = ClientData(x, y, s)
        global_models.get_model(global_model_id).set_client(client_id, client_data)

    return global_models


def create_global_models_drifted_clients(global_models, global_models_to_create, clients_identities):
    for model, client_id, client_data in global_models_to_create:
        new_global_model = global_models.create_new_global_model(model)
        clients_identities[client_id].append(ClientIdentity(new_global_model.id, new_global_model.name))

    return global_models, clients_identities


def test_client_on_model(metrics_clustering, model, client_data, is_binary_target):
    y_pred_raw = model.predict(client_data.x)
    y_true, y_pred = get_y(client_data.y, y_pred_raw, is_binary_target)

    results = []
    for metric_clustering in metrics_clustering:
        result = metric_clustering.calculate(y_true, y_pred, client_data.y, y_pred_raw, client_data.s)
        results.append(result)

    return results


def train_and_average(clients_data_timestep, global_models, dataset, seed, timestep, clients_identities):
    for cround in range(dataset.n_rounds):
        local_weights_list = [[] for _ in range(global_models.current_size)]
        local_scales_list = [[] for _ in range(global_models.current_size)]

        for client_id, client_data in enumerate(clients_data_timestep):
            x, y, s, y_original = client_data
            global_model_id = clients_identities[client_id][-1].id
            local_model = get_model_client(global_models, global_model_id, dataset, seed)
            local_model.learn(x, y)
            print("Trained model timestep {} cround {} client {}".format(timestep, cround, client_id))
            local_weights_list[global_model_id].append(local_model.get_weights())
            local_scales_list[global_model_id].append(len(x))

        for global_model_id, (local_weights, local_scales) in enumerate(zip(local_weights_list, local_scales_list)):
            if len(local_weights) > 0:
                new_global_weights = average_weights(local_weights, local_scales)
                global_models.get_model(global_model_id).model.set_weights(new_global_weights)
                print("Averaged models on timestep {} cround {} of cluster {}".format(timestep, cround, global_model_id))
            else:
                print("Did not average models on timestep {} cround {} of cluster {}".format(timestep, cround, global_model_id))

    return global_models


def get_results_list(global_model_model, global_model_data, dataset, metrics_clustering):
    results_list = []
    for client_id in global_model_data.clients.keys():
        partial_client_data = global_model_data.get_partial_client_data(client_id)
        results = test_client_on_model(
            metrics_clustering, global_model_model.model, partial_client_data,
            dataset.is_binary_target
        )
        results_list.append(results)

    return results_list


def get_distance(global_model_a, global_model_b, dataset, metrics_clustering, thresholds):
    results_list_a_b = get_results_list(global_model_a, global_model_b, dataset, metrics_clustering)
    results_list_a_a = get_results_list(global_model_a, global_model_a, dataset, metrics_clustering)

    maximum_distance = BEST_LOSS

    for results_a_b in results_list_a_b:
        for results_a_a in results_list_a_a:
            distance_list = []
            for result_a_b, result_a_a, threshold in zip(results_a_b, results_a_a, thresholds):
                distance = abs(result_a_b - result_a_a)
                if distance > threshold:
                    return WORST_LOSS  # EXIT
                else:
                    distance_list.append(distance)
            distance = sum(distance_list)
            if distance > maximum_distance:
                maximum_distance = distance

    return maximum_distance


def merge_global_models(metrics_clustering, thresholds, global_models, dataset, seed):
    size = global_models.current_size
    if size > 25:
        raise Exception("Number of global models > 25")
    distances = [[WORST_LOSS for _ in range(len(thresholds)) for _ in range(size)] for _ in range(size)]  # TODO (?)

    for i in range(len(global_models.models) - 1):
        for j in range(i + 1, len(global_models.models)):
            id_i = global_models.models[i].id
            id_j = global_models.models[j].id
            L_1 = get_distance(global_models.models[i], global_models.models[j], dataset, metrics_clustering, thresholds)
            L_2 = get_distance(global_models.models[j], global_models.models[i], dataset, metrics_clustering, thresholds)
            if L_1 == WORST_LOSS or L_2 == WORST_LOSS:
                d = WORST_LOSS
            else:
                d = max(L_1 - L_2, L_2 - L_1, 0)
            print(id_i, id_j, L_1, L_2, d)
            distances[id_i][id_j] = d
            distances[id_j][id_i] = d

    while True:  # While we can still merge global models
        print_matrix(distances)
        id_0, id_1, found = get_next_best_results(distances)
        if found:
            print("Merged models {} and {}".format(id_0, id_1))
            global_models, distances = merge_global_models_spec(dataset, seed, global_models, id_0, id_1, distances)
        else:
            return global_models


def merge_global_models_spec(dataset, seed, global_models, id_0, id_1, distances):
    global_model_0 = global_models.get_model(id_0)
    global_model_1 = global_models.get_model(id_1)
    scales = [global_model_0.n_points, global_model_1.n_points]
    weights = [global_model_0.model.get_weights(), global_model_1.model.get_weights()]
    new_global_model_weights = average_weights(weights, scales)
    new_global_model = get_init_model(dataset, seed)
    new_global_model.set_weights(new_global_model_weights)

    # Create new global Model
    new_global_model_created = global_models.create_new_global_model(
        new_global_model, global_model_0.name, global_model_1.name
    )
    for client_id, client_data in global_model_0.clients.items():
        new_global_model_created.set_client(client_id, client_data)
    for client_id, client_data in global_model_1.clients.items():
        new_global_model_created.set_client(client_id, client_data)

    # Create new column and row for new model id and update distances
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

    return global_models, distances


def print_matrix(matrix):
    for d in matrix:
        print(" ".join([str(a) for a in d]))

def get_best_model(model_results_dict, minimum_loss_client, thresholds):
    best_model = None  # if there is not a drift
    minimum_loss_client_new_a = None  # if there is not a drift
    minimum_loss_client_new_b = None # if there is a drift (no models where loss < previous_loss + threshold)

    for model, results in model_results_dict.items():
        drift_detected = False
        for loss, previous_loss, threshold in zip(results, minimum_loss_client[-1], thresholds):
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
