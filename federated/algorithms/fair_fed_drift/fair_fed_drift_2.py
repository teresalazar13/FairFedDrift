from federated.algorithms.Algorithm import Algorithm, average_weights
from federated.algorithms.fair_fed_drift.ClientData import ClientData
from federated.algorithms.fair_fed_drift.GlobalModels import GlobalModels
from federated.algorithms.fair_fed_drift.drift_detection.DriftDetectorFactory import get_detector_by_name
from federated.model import NN_model
from metrics.MetricFactory import get_metrics, get_metrics_by_names
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MAX = -1

class FairFedDrift_2(Algorithm):
    def __init__(self):
        self.metrics_clustering = None
        self.drift_detector = None
        name = "fair_fed_drift_2"
        super().__init__(name)

    def set_specs(self, args):
        self.drift_detector = get_detector_by_name(args.drift_detector)
        self.drift_detector.set_specs(args)
        self.metrics_clustering = get_metrics_by_names(args.metrics)
        self.similarity = float(args.similarity)
        metrics_string = "-".join(args.metrics)
        thresholds_string = "-".join(args.thresholds)
        super().set_subfolders("{}/drift_detector-{}-{}/similarity-{}/metrics-{}".format(
            self.name, self.drift_detector.name, thresholds_string, args.similarity, metrics_string)
        )

    def perform_fl(self, seed, clients_data, dataset):
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        global_model = global_models.create_new_global_model(init_model)
        for client_id in range(dataset.n_clients):
            cd = ClientData(clients_data[0][client_id][0], clients_data[0][client_id][1], clients_data[0][client_id][2])
            global_model.set_client(client_id, cd)
        clients_metrics = [get_metrics(dataset.is_image) for _ in range(dataset.n_clients)]
        clients_identities = [[] for _ in range(dataset.n_clients)]

        for timestep in range(dataset.n_timesteps):
            clients_identities = update_clients_identities(clients_identities, dataset.n_clients, global_models)

            # STEP 1 - Test each client's data on previous clustering identities
            self.test_models(global_models, clients_data[timestep], clients_metrics, dataset, seed)
            global_models.reset_clients()

            if timestep != dataset.n_timesteps - 1:
                # STEP 2 - Recalculate Global Models (cluster identities) and detect concept drift
                global_models = self.update_global_models(clients_data[timestep], global_models, dataset, seed, timestep)

                # STEP 3 - Train and average models
                global_models = self.train_and_average(clients_data[timestep], global_models, dataset, seed, timestep)

                # STEP 4 - Merge Global Models
                global_models = self.merge_global_models(global_models, dataset, seed)

        return clients_metrics, clients_identities

    def merge_global_models(self, global_models, dataset, seed):
        size = global_models.current_size
        if size > 25:
            raise Exception("Number of global models > 25")
        distances = [[MAX for _ in range(size)] for _ in range(size)]

        for i in range(len(global_models.models)):
            for j in range(len(global_models.models)):
                id_i = global_models.models[i].id
                id_j = global_models.models[j].id
                if i != j and len(global_models.models[i].clients.keys()) > 0 and len(global_models.models[j].clients.keys()) > 0:
                    weights1 = global_models.models[i].model.get_weights()
                    weights2 = global_models.models[j].model.get_weights()
                    flattened_weights1 = np.concatenate([w.flatten() for w in weights1])
                    flattened_weights2 = np.concatenate([w.flatten() for w in weights2])
                    similarity = cosine_similarity([flattened_weights1], [flattened_weights2])[0][0]
                    distances[id_i][id_j] = similarity
                    distances[id_j][id_i] = similarity

        while True:  # While we can still merge global models
            print_matrix(distances)
            id_0, id_1 = self.get_next_best_distance(distances)
            if id_0 and id_1:
                print("Merged models {} and {}".format(id_0, id_1))
                global_models, distances = self.merge_global_models_spec(dataset, seed, global_models, id_0, id_1, distances)
            else:
                return global_models


    def merge_global_models_spec(self, dataset, seed, global_models, id_0, id_1, distances):
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
        new_global_model_created = global_models.create_new_global_model(new_global_model)
        for client_id, client_data in clients.items():
            new_global_model_created.set_client(client_id, client_data)

        # Create new column and row for new model id and update distances
        new_row = []
        for i in range(len(distances)):
            worst_distance = distances[id_0][i]
            if distances[id_1][i] < worst_distance:
                worst_distance = distances[id_1][i]
            new_row.append(worst_distance)
        for i in range(len(distances)):
            distances[i].append(new_row[i])
        new_row.append(MAX)
        distances.append(new_row)

        # Reset Distances of deleted models
        distances[id_0][id_1] = MAX
        distances[id_1][id_0] = MAX
        for i in range(len(distances)):
            distances[id_0][i] = MAX
            distances[id_1][i] = MAX
            distances[i][id_0] = MAX
            distances[i][id_1] = MAX

        global_models.deleted_merged_model(id_0)
        global_models.deleted_merged_model(id_1)

        return global_models, distances


    def train_and_average(self, clients_data_timestep, global_models, dataset, seed, timestep):
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

    def update_global_models(self, clients_data_timestep, global_models, dataset, seed, timestep):
        global_models_to_create = []

        for client_id, client_data_raw in enumerate(clients_data_timestep):
            x, y, s, _ = client_data_raw
            client_data = ClientData(x, y, s)

            # Calculate results on all global models
            results_global_models = {}
            for global_model in global_models.models:
                results = self.test_client_on_model(self.metrics_clustering, global_model.model, client_data, dataset.is_image)
                results_global_models[global_model] = results

            # Get Model for client given results_global_models and test
            model_weights, global_model_id = get_model_weights_for_client(results_global_models)
            model = get_init_model(dataset, seed)
            model.set_weights(model_weights)
            results_model = self.test_client_on_model(self.metrics_clustering, model, client_data, dataset.is_image)

            # Detect Drift
            drift_detected_metrics = self.drift_detector.drift_detected(results_model, timestep)
            if sum(drift_detected_metrics) >= 1:
                print("Drift detected at client {}".format(client_id))
                model = get_init_model(dataset, seed)
                global_models_to_create.append([model, client_id, client_data])
            else:
                print("No drift detected at client {}".format(client_id))
                global_models.set_client_model(global_model_id, client_id, client_data)

        for model, client_id, client_data in global_models_to_create:
            new_global_model = global_models.create_new_global_model(model)
            new_global_model.set_client(client_id, client_data)

        return global_models

    def test_client_on_model(self, metrics, model, client_data, is_image):
        results = []
        pred = model.predict(client_data.x)
        y_true, y_pred = super().get_y(client_data.y, pred, is_image)
        for client_metric in metrics:
            res = client_metric.calculate(y_true, y_pred, client_data.s)
            results.append(res)

        return results

    def test_models(self, global_models, clients_data_timestep, clients_metrics, dataset, seed):
        for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
            x, y, s, _ = client_data
            model, _ = get_model_client(client_id, global_models, dataset, seed)
            pred = model.predict(x)
            y_true, y_pred = super().get_y(y, pred, dataset.is_image)
            for client_metric in client_metrics:
                res = client_metric.update(y_true, y_pred, s)
                print(res, client_metric.name)
            for client_metric in self.metrics_clustering:
                client_metric.update(y_true, y_pred, s)

    def get_next_best_distance(self, distances_matrix):
        best_row = None
        best_col = None
        best_results = -1

        for row in range(len(distances_matrix)):
            for col in range(len(distances_matrix[row])):
                results = distances_matrix[row][col]
                if results > self.similarity and results > best_results:  # TODO - define hyperparameter - 0.0125
                    best_results = results
                    best_row = row
                    best_col = col

        return best_row, best_col

def get_init_model(dataset, seed):
    model = NN_model(dataset.input_shape, seed, dataset.is_image)
    model.compile(dataset.is_image)

    return model


def update_clients_identities(clients_identities, n_clients, global_models):
    for client_id in range(n_clients):
        timestep_client_identities = []
        for model in global_models.models:
            for client in model.clients.keys():
                if client == client_id:
                    timestep_client_identities.append(model.id)
        clients_identities[client_id].append(timestep_client_identities)
    print_clients_identities(clients_identities)

    return clients_identities


def print_clients_identities(clients_identities):
    print("\nClients identities")

    for timestep in range(len(clients_identities[0])):
        print("\nTimestep ", timestep)
        identities = {}
        for client in range(len(clients_identities)):
            client_identities_timestep = clients_identities[client][timestep]
            for model_id in client_identities_timestep:
                if model_id in identities:
                    identities[model_id].append(client)
                else:
                    identities[model_id] = [client]

        for model_id, clients in identities.items():
            print("Model id: ", model_id, ":", clients)


def print_matrix(matrix):
    for d in matrix:
        string = ""
        for a in d:
            string += "   {:.4f}".format(a)
        print(string)


def get_model_weights_for_client(results_global_models):
    best_results = []
    best_global_model = None
    model_id = None

    for global_model, results in results_global_models.items():
        if sum(results) > sum(best_results) or best_global_model is None:
            best_results = results
            best_global_model = global_model
            model_id = global_model.id

    return best_global_model.model.get_weights(), model_id

def get_model_client(client_id: int, global_models: GlobalModels, dataset, seed):
    for global_model in global_models.models:
        for client in global_model.clients:
            if client == client_id:
                weights = global_model.model.get_weights()
                model = get_init_model(dataset, seed)
                model.set_weights(weights)
                return model, global_model.id

    raise Exception("No model for client", client_id)