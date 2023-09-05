from federated.algorithms.Algorithm import Algorithm, average_weights
from federated.algorithms.fair_fed_drift.Client import Client
from federated.algorithms.fair_fed_drift.ClientData import ClientData
from federated.algorithms.fair_fed_drift.GlobalModels import GlobalModels, get_models_proportions
from federated.algorithms.fair_fed_drift.clustering.ClusteringFactory import get_clustering_by_name
from federated.algorithms.fair_fed_drift.drift_detection.DriftDetectorFactory import get_detector_by_name
from federated.algorithms.fedavg_lr import calculate_sample_weights
from federated.model import NN_model
from metrics.MetricFactory import get_metrics, get_metrics_by_names
import math
import numpy as np


class FairFedDrift(Algorithm):
    def __init__(self):
        self.metrics_clustering = None
        self.drift_detector = None
        self.clustering = None
        self.local_reweighting = False
        self.oversampling = False
        self.boost_factor = 0
        name = "fair_fed_drift"
        super().__init__(name)

    def set_specs(self, args):
        self.clustering = get_clustering_by_name(args.clustering)
        self.drift_detector = get_detector_by_name(args.drift_detector)
        self.drift_detector.set_specs(args)
        self.metrics_clustering = get_metrics_by_names(args.metrics)
        if "lr" in args and args.lr == "1":
            self.local_reweighting = True
            lr_string = "/with_lr"
        else:
            lr_string = ""
        if "oversampling" in args and args.oversampling == "1":
            self.oversampling = True
            oversampling_string = "/with_oversampling"
        else:
            oversampling_string = ""
        if "boost_factor" in args and args.boost_factor is not None:
            self.boost_factor = float(args.boost_factor)
            boost_factor_string = "/boost_{}".format(self.boost_factor)
        else:
            boost_factor_string = ""
        metrics_string = "-".join(args.metrics)
        thresholds_string = "-".join(args.thresholds)
        super().set_subfolders("{}/clustering-{}/drift_detector-{}-{}/metrics-{}{}{}{}".format(
            self.name, self.clustering.name, self.drift_detector.name, thresholds_string, metrics_string,
            lr_string, oversampling_string, boost_factor_string
        ))

    def perform_fl(self, seed, clients_data, dataset):
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        init_clients = [Client(id, 1) for id in range(dataset.n_clients)]
        global_models.create_new_global_model(init_model, init_clients)
        clients_metrics = [get_metrics(dataset.is_image) for _ in range(dataset.n_clients)]
        clients_identities = [[] for _ in range(dataset.n_clients)]

        for timestep in range(dataset.n_timesteps):
            clients_identities = update_clients_identities(clients_identities, dataset.n_clients, global_models)

            # STEP 1 - Test each client's data on previous clustering identities
            boost_weights_clients, clients_data_mistakes = self.test_models(
                global_models, clients_data[timestep], clients_metrics, dataset, timestep, seed
            )
            global_models.reset_clients()

            if timestep != dataset.n_timesteps - 1:
                # STEP 2 - Recalculate Global Models (cluster identities) and detect concept drift
                global_models = self.update_global_models(clients_data[timestep], global_models, dataset, seed,
                                                          timestep)

                # STEP 3 - Merge Global Models
                global_models = self.merge_global_models(global_models, dataset, seed)

                # STEP 4 - Train and average models
                global_models = self.train_and_average(
                    clients_data[timestep], global_models, dataset, seed, timestep, boost_weights_clients,
                    clients_data_mistakes
                )

        return clients_metrics, clients_identities

    def train_and_average(
            self, clients_data_timestep, global_models, dataset, seed, timestep, boost_weights_clients,
            clients_data_mistakes
    ):
        for cround in range(dataset.n_rounds):  # TODO - boost_weights_clients only assumes one round?
            local_weights_list = [[] for _ in range(global_models.current_size)]
            local_scales_list = [[] for _ in range(global_models.current_size)]

            for client_id, client_data in enumerate(clients_data_timestep):
                if timestep > 0 and self.oversampling is True and len(clients_data_mistakes[client_id][0]) > 0:  # TODO
                    a = client_data.copy()
                    x = np.concatenate((a[0], clients_data_mistakes[client_id][0]))
                    y = np.concatenate((a[1], clients_data_mistakes[client_id][1]))
                    s = np.concatenate((a[2], clients_data_mistakes[client_id][2]))
                    y_original = np.concatenate((a[3], clients_data_mistakes[client_id][3]))
                    p = np.random.permutation(len(a[0]))
                    x, y, s, y_original = x[p], y[p], s[p], y_original[p]
                else:
                    x, y, s, y_original = client_data
                local_model, global_model_amounts = self.get_model_client(client_id, global_models, dataset, seed)
                if self.local_reweighting:
                    sample_weights = calculate_sample_weights(y_original, s)
                    local_model.learn(x, y, sample_weights=sample_weights)
                elif self.boost_factor != 0:
                    local_model.learn(x, y, sample_weights=boost_weights_clients[client_id])
                else:
                    local_model.learn(x, y)
                print("Trained model timestep {} cround {} client {}".format(timestep, cround, client_id))

                for global_model_id, amount in global_model_amounts:
                    global_models.get_model(global_model_id).add_client_data(client_id, x, y, s, amount)
                    local_weights_list[global_model_id].append(local_model.get_weights())
                    local_scales_list[global_model_id].append(len(x) * amount)

            for global_model_id, (local_weights, local_scales) in enumerate(zip(local_weights_list, local_scales_list)):
                if len(local_weights) > 0:
                    new_global_weights = average_weights(local_weights, local_scales)
                    global_models.get_model(global_model_id).model.set_weights(new_global_weights)
                    print("Averaged models on timestep {} cround {} of cluster {}".format(
                        timestep, cround, global_model_id)
                    )
                else:
                    print("Did not average models on timestep {} cround {} of cluster {}".format(
                        timestep, cround, global_model_id)
                    )

        return global_models

    def merge_global_models(self, global_models, dataset, seed):
        size = global_models.current_size
        if size > 25:
            raise Exception("Number of global models > 25")
        all_distances = [[[0 for _ in range(len(self.metrics_clustering))] for _ in range(size)] for _ in range(size)]

        for i in range(len(global_models.models)):
            for j in range(len(global_models.models)):
                id_i = global_models.models[i].id
                id_j = global_models.models[j].id
                if i != j and len(global_models.models[i].clients_data) > 0 and len(
                        global_models.models[j].clients_data) > 0:
                    results_list = []
                    for client_id in global_models.models[j].clients_data.keys():
                        """
                        print("Testing global model {} on data from cluster {} - client {}".format(
                            id_i, id_j, client_id)
                        )"""
                        partial_client_data = global_models.models[j].get_partial_client_data(client_id)
                        results = self.test_client_on_model(
                            self.metrics_clustering, global_models.models[i].model, partial_client_data,
                            dataset.is_image
                        )
                        results_list.append(results)
                    worst_results = self.drift_detector.get_worst_results(results_list)
                    """print("Distance between global model {} and {}: {}".format(id_i, id_j, worst_results))"""
                    all_distances[id_i][id_j] = worst_results

        distances = [[[0 for _ in range(len(self.metrics_clustering))] for _ in range(size)] for _ in range(size)]
        for i in range(len(global_models.models)):
            for j in range(len(global_models.models)):
                id_i = global_models.models[i].id
                id_j = global_models.models[j].id
                results_list = [all_distances[id_i][id_j], all_distances[id_j][id_i]]
                worst_results = self.drift_detector.get_worst_results(results_list)
                """print("Final distance between global model {} and {}: {}".format(id_i, id_j, worst_results))"""
                distances[id_i][id_j] = worst_results

        while True:  # While we can still merge global models
            print_matrix(distances)
            id_0, id_1 = self.drift_detector.get_next_best_results(distances)
            if id_0 and id_1:
                """print("Merging global model {} with global model {}".format(id_0, id_1))"""
                global_models, distances = self.merge_global_models_spec(
                    dataset, seed, global_models, id_0, id_1, distances
                )
            else:
                return global_models

    def merge_global_models_spec(self, dataset, seed, global_models, id_0, id_1, distances):
        global_model_0 = global_models.get_model(id_0)
        global_model_1 = global_models.get_model(id_1)
        scales = [
            get_models_proportions(global_models, id_0),
            get_models_proportions(global_models, id_1)
        ]
        weights = [global_model_0.model.get_weights(), global_model_1.model.get_weights()]
        new_global_model_weights = average_weights(weights, scales)
        new_global_model = get_init_model(dataset, seed)
        new_global_model.set_weights(new_global_model_weights)

        # Create new global Model
        clients = global_model_0.clients
        clients.extend(global_model_1.clients)
        new_global_model_created = global_models.create_new_global_model(new_global_model, clients)
        for client_id, client_data_list in global_model_0.clients_data.items():  # add client data of global_model_0
            for client_data, amount in client_data_list:
                new_global_model_created.add_client_data(
                    client_id, client_data.x, client_data.y, client_data.s, amount
                )
        for client_id, client_data_list in global_model_0.clients_data.items():  # add client data of global_model_1
            for client_data, amount in client_data_list:
                new_global_model_created.add_client_data(
                    client_id, client_data.x, client_data.y, client_data.s, amount
                )
        """
        print("Created global model {} from global models {} and {}".format(
            new_global_model_created.id, id_0, id_1)
        )"""

        # Create new column and row for new model id and update distances
        new_row = []
        for i in range(len(distances)):
            results_list = [distances[id_0][i], distances[id_1][i]]
            worst_results = self.drift_detector.get_worst_results(results_list)
            new_row.append(worst_results)
        for i in range(len(distances)):
            distances[i].append(new_row[i])
        new_row.append([0 for _ in range(len(self.metrics_clustering))])
        distances.append(new_row)

        # Reset Distances of deleted models
        distances[id_0][id_1] = [0 for _ in range(len(self.metrics_clustering))]
        distances[id_1][id_0] = [0 for _ in range(len(self.metrics_clustering))]
        for i in range(len(distances)):
            distances[id_0][i] = [0 for _ in range(len(self.metrics_clustering))]
            distances[id_1][i] = [0 for _ in range(len(self.metrics_clustering))]
            distances[i][id_0] = [0 for _ in range(len(self.metrics_clustering))]
            distances[i][id_1] = [0 for _ in range(len(self.metrics_clustering))]

        global_models.deleted_merged_model(id_0)
        global_models.deleted_merged_model(id_1)

        return global_models, distances

    def update_global_models(self, clients_data_timestep, global_models, dataset, seed, timestep):
        global_models_to_create = []

        for client_id, client_data_raw in enumerate(clients_data_timestep):
            x, y, s, _ = client_data_raw
            client_data = ClientData(x, y, s)

            # Calculate results on all global models
            results_global_models = {}
            #print("Client {} - testing on all global models".format(client_id))
            for global_model in global_models.models:
                #print("Client {} - testing on global model id {}".format(client_id, global_model.id))
                results = self.test_client_on_model(
                    self.metrics_clustering, global_model.model, client_data, dataset.is_image
                )
                results_global_models[global_model] = results

            # Get Model for client given results_global_models and test
            model_weights, model_id_amounts = self.clustering.get_model_weights_for_client(
                results_global_models, timestep
            )
            model = get_init_model(dataset, seed)
            model.set_weights(model_weights)
            #print("Client {} - testing on best model(s)".format(client_id))
            results_model = self.test_client_on_model(self.metrics_clustering, model, client_data, dataset.is_image)

            # Detect Drift
            drift_detected_metrics = self.drift_detector.drift_detected(results_model, timestep)
            if sum(drift_detected_metrics) >= 1:
                print("Drift detected at client {}".format(client_id))
                model = get_init_model(dataset, seed)
                clients = [Client(client_id, 1)]
                global_models_to_create.append([model, clients])
            else:
                print("No drift detected at client {}".format(client_id))
                for global_model_id, amount in model_id_amounts.items():
                    client = Client(client_id, amount)
                    global_models.set_client_model(global_model_id, client)

        for model, clients in global_models_to_create:
            global_models.create_new_global_model(model, clients)

        return global_models

    def test_client_on_model(self, metrics, model, client_data, is_image):
        results = []
        pred = model.predict(client_data.x)
        y_true_original, y_pred_original, y_true, y_pred = super().get_y(client_data.y, pred, is_image)
        for client_metric in metrics:
            res = client_metric.calculate(y_true_original, y_pred_original, y_true, y_pred, client_data.s)
            #print("{} - {:.2f}".format(client_metric.name, res))
            results.append(res)

        return results

    def test_models(self, global_models, clients_data_timestep, clients_metrics, dataset, timestep, seed):
        boost_weights_clients = []
        clients_data_mistakes = []

        for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
            x, y, s, _ = client_data
            model, _ = self.get_model_client(client_id, global_models, dataset, seed)
            pred = model.predict(x)
            y_true_original, y_pred_original, y_true, y_pred = super().get_y(y, pred, dataset.is_image)
            boost_weights_clients.append(self.get_boost_weights(y_true, y_pred, timestep))
            clients_data_mistakes.append(get_clients_data_mistakes(client_data, y_true, y_pred))
            for client_metric in client_metrics:
                res = client_metric.update(y_true_original, y_pred_original, y_true, y_pred, s)
                print(res, client_metric.name)
            for client_metric in self.metrics_clustering:
                client_metric.update(y_true_original, y_pred_original, y_true, y_pred, s)

        return boost_weights_clients, clients_data_mistakes

    def get_model_client(self, client_id: int, global_models: GlobalModels, dataset, seed):
        weights = []
        amounts = []
        global_model_amounts = []
        for global_model in global_models.models:
            for client in global_model.clients:
                if client.id == client_id:
                    weights.append(global_model.model.get_weights())
                    amounts.append(client.amount)
                    global_model_amounts.append([global_model.id, client.amount])

        averaged_weights = average_weights(weights, amounts)
        model = get_init_model(dataset, seed)
        model.set_weights(averaged_weights)

        return model, global_model_amounts

    def get_boost_weights(self, y_true, y_pred, timestep):
        weights = []
        for y_true_i, y_pred_i in zip(y_true, y_pred):
            w = 1
            if timestep == 0:
                weights.append(w)
            else:
                if y_true_i == y_pred_i:
                    weights.append(w * math.exp(-self.boost_factor))
                else:
                    weights.append(w * math.exp(self.boost_factor))

        return np.array(weights)


def get_init_model(dataset, seed):
    model = NN_model(dataset.input_shape, seed, dataset.is_image)
    model.compile(dataset.is_image)

    return model


def update_clients_identities(clients_identities, n_clients, global_models):
    for client_id in range(n_clients):
        timestep_client_identities = []

        for model in global_models.models:
            for client in model.clients:
                if client.id == client_id:
                    timestep_client_identities.append([model.id, client.amount])

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
            for model_id, amount in client_identities_timestep:
                if model_id in identities:
                    identities[model_id].append([client, amount])
                else:
                    identities[model_id] = [[client, amount]]

        for model_id, clients_amounts in identities.items():
            print("Model id: ", model_id, ":", clients_amounts)


def print_matrix(matrix):
    for d in matrix:
        string = ""
        for a in d:
            string += " " + "-".join([str(b) for b in a])
        print(string)


def get_clients_data_mistakes(client_data, y_true, y_pred):  # TODO
    x, y, s, y_original = client_data
    data_x = []
    data_y = []
    data_s = []
    data_y_original = []
    for y_true_i, y_pred_i, x_i, y_i, s_i, yo_i in zip(y_true, y_pred, x, y, s, y_original):
        if y_true_i != y_pred_i:
            data_x.append(x_i)
            data_y.append(y_i)
            data_s.append(s_i)
            data_y_original.append(yo_i)

    data = [data_x, data_y, data_s, data_y_original]

    return data
