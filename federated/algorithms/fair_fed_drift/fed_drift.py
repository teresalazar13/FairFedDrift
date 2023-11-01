from federated.algorithms.Algorithm import Algorithm, get_y, average_weights
from federated.algorithms.fair_fed_drift.ClientData import ClientData
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
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        global_model = global_models.create_new_global_model(init_model)
        for client_id in range(dataset.n_clients):
            cd = ClientData(clients_data[0][client_id][0], clients_data[0][client_id][1], clients_data[0][client_id][2])
            global_model.set_client(client_id, cd)
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        clients_identities = [[] for _ in range(dataset.n_clients)]

        for timestep in range(dataset.n_timesteps):
            clients_identities = update_clients_identities(clients_identities, dataset.n_clients, global_models)

            # STEP 1 - Test each client's data on previous clustering identities
            test_models(global_models, clients_data[timestep], clients_metrics, dataset, seed)
            global_models.reset_clients()

            if timestep != dataset.n_timesteps - 1:
                # STEP 2 - Recalculate Global Models (cluster identities) and detect concept drift
                global_models = update_global_models(
                    self.metrics_clustering, self.thresholds, clients_data[timestep], global_models, dataset, seed,
                    timestep
                )

                # STEP 3 - Merge Global Models
                global_models = merge_global_models(
                    self.metrics_clustering, self.thresholds, global_models, dataset, seed
                )

                # STEP 4 - Train and average models
                global_models = train_and_average(clients_data[timestep], global_models, dataset, seed, timestep)

        # TODO - fix saving of data of clients of global model to reflect original paper
        return clients_metrics, clients_identities


def get_init_model(dataset, seed):
    model = NN_model(dataset, seed)
    model.compile(dataset)

    return model


def update_clients_identities(clients_identities, n_clients, global_models):
    for client_id in range(n_clients):
        timestep_client_identities = get_timestep_client_identity(global_models, client_id)
        clients_identities[client_id].append(timestep_client_identities)
    print_clients_identities(clients_identities)

    return clients_identities


def get_timestep_client_identity(global_models, client_id):
    for model in global_models.models:
        for client in model.clients.keys():
            if client == client_id:
                return model.name

    raise Exception("Model for client {} no found".format(client_id))


def print_clients_identities(clients_identities):
    print("\nClients identities")

    for timestep in range(len(clients_identities[0])):
        print("\nTimestep ", timestep)
        identities = {}
        for client in range(len(clients_identities)):
            client_identity_timestep = clients_identities[client][timestep]
            if client_identity_timestep in identities:
                identities[client_identity_timestep].append(client)
            else:
                identities[client_identity_timestep] = [client]

        for model_id, clients in identities.items():
            print("Model id: ", model_id, ":", clients)


def test_models(global_models, clients_data_timestep, clients_metrics, dataset, seed):
    for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
        x, y_true_raw, s, _ = client_data
        model, _ = get_model_client(client_id, global_models, dataset, seed)
        y_pred_raw = model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, dataset.is_binary_target)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            print(res, client_metric.name)


def get_model_client(client_id, global_models, dataset, seed):
    for global_model in global_models.models:
        for client in global_model.clients:
            if client == client_id:
                weights = global_model.model.get_weights()
                model = get_init_model(dataset, seed)
                model.set_weights(weights)

                return model, global_model.id

    raise Exception("No model for client", client_id)


def update_global_models(metrics_clustering, thresholds, clients_data_timestep, global_models, dataset, seed, timestep):
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
        best_global_model = get_best_model(results_global_models, thresholds, timestep)
        if best_global_model:
            print("No drift detected at client {}".format(client_id))
            best_model_weights = best_global_model.model.get_weights()
            best_global_model_id = best_global_model.id
            model = get_init_model(dataset, seed)
            model.set_weights(best_model_weights)
            global_models.set_client_model(best_global_model_id, client_id, client_data)
        else:
            print("Drift detected at client {}".format(client_id))
            model = get_init_model(dataset, seed)
            global_models_to_create.append([model, client_id, client_data])

    for model, client_id, client_data in global_models_to_create:
        new_global_model = global_models.create_new_global_model(model)
        new_global_model.set_client(client_id, client_data)

    return global_models


def test_client_on_model(metrics_clustering, model, client_data, is_binary_target):
    y_pred_raw = model.predict(client_data.x)
    y_true, y_pred = get_y(client_data.y, y_pred_raw, is_binary_target)

    results = []
    for metric_clustering in metrics_clustering:
        result = metric_clustering.calculate(y_true, y_pred, client_data.y, y_pred_raw, client_data.s)
        results.append(result)

    return results


def train_and_average(clients_data_timestep, global_models, dataset, seed, timestep):
    for cround in range(dataset.n_rounds):
        local_weights_list = [[] for _ in range(global_models.current_size)]
        local_scales_list = [[] for _ in range(global_models.current_size)]

        for client_id, client_data in enumerate(clients_data_timestep):
            x, y, s, y_original = client_data
            local_model, global_model_id = get_model_client(client_id, global_models, dataset, seed)
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


def merge_global_models(metrics_clustering, thresholds, global_models, dataset, seed):
    size = global_models.current_size
    if size > 25:
        raise Exception("Number of global models > 25")
    all_distances = [[[WORST_LOSS for _ in range(len(thresholds))] for _ in range(size)] for _ in range(size)]

    for i in range(len(global_models.models)):
        for j in range(len(global_models.models)):
            id_i = global_models.models[i].id
            id_j = global_models.models[j].id
            if i != j and len(global_models.models[i].clients.keys()) > 0 and len(global_models.models[j].clients.keys()) > 0:
                results_list = []
                for client_id in global_models.models[j].clients.keys():
                    partial_client_data = global_models.models[j].get_partial_client_data(client_id)
                    results = test_client_on_model(
                        metrics_clustering, global_models.models[i].model, partial_client_data,
                        dataset.is_binary_target
                    )
                    results_list.append(results)
                all_distances[id_i][id_j] = get_worst_results(results_list)

    distances = [[[WORST_LOSS for _ in range(len(thresholds))] for _ in range(size)] for _ in range(size)]
    for i in range(len(global_models.models)):
        for j in range(len(global_models.models)):
            id_i = global_models.models[i].id
            id_j = global_models.models[j].id
            results_list = [all_distances[id_i][id_j], all_distances[id_j][id_i]]
            worst_results = get_worst_results(results_list)
            distances[id_i][id_j] = worst_results

    while True:  # While we can still merge global models
        print_matrix(distances)
        id_0, id_1 = get_next_best_results(distances, thresholds)
        if id_0 and id_1:
            print("Merged models {} and {}".format(id_0, id_1))
            global_models, distances = merge_global_models_spec(dataset, seed, global_models, id_0, id_1, distances)
        else:
            return global_models


def merge_global_models_spec(dataset, seed, global_models, id_0, id_1, distances):
    global_model_0 = global_models.get_model(id_0)
    global_model_1 = global_models.get_model(id_1)
    scales = [global_model_0.n_points, global_model_0.n_points]
    weights = [global_model_0.model.get_weights(), global_model_1.model.get_weights()]
    new_global_model_weights = average_weights(weights, scales)
    new_global_model = get_init_model(dataset, seed)
    new_global_model.set_weights(new_global_model_weights)

    # Create new global Model
    clients = global_model_0.clients
    clients.update(global_model_1.clients)
    new_global_model_created = global_models.create_new_global_model(
        new_global_model, global_model_0.name, global_model_1.name
    )
    for client_id, client_data in clients.items():
        new_global_model_created.set_client(client_id, client_data)

    # Create new column and row for new model id and update distances
    new_row = []
    for i in range(len(distances)):
        results_list = [distances[id_0][i], distances[id_1][i]]
        new_row.append(max(results_list))
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

def get_best_model(model_results_dict, thresholds, timestep):
    if timestep == 0:
        return list(model_results_dict.keys())[0]

    best_model = None
    best_results_sum = WORST_LOSS

    for model, results in model_results_dict.items():
        results_sum = sum(results)
        drift_detected = False
        for result, threshold in zip(results, thresholds):
            if result > threshold:  # if drift detected
                drift_detected = True

        if not drift_detected and results_sum < best_results_sum:
            best_model = model
            best_results_sum = results_sum

    return best_model


def get_next_best_results(results_matrix, thresholds):
    best_row = None
    best_col = None
    best_results = [WORST_LOSS for _ in range(len(thresholds))]
    best_results_sum = sum(best_results)

    for row in range(len(results_matrix)):
        for col in range(len(results_matrix[row])):
            results = results_matrix[row][col]

            results_sum = sum(results)
            drift_detected = False
            for result, threshold in zip(results, thresholds):
                if result > threshold:  # if drift detected
                    drift_detected = True

            if not drift_detected and results_sum < best_results_sum:
                best_results_sum = results_sum
                best_row = row
                best_col = col

    return best_row, best_col


def get_worst_results(results_list):
    worst_results = [BEST_LOSS for _ in range(len(results_list[0]))]
    worst_results_sum = sum(worst_results)

    for results in results_list:
        results_sum = sum(results)

        if results_sum > worst_results_sum:
            worst_results = results
            worst_results_sum = results_sum

    return worst_results
