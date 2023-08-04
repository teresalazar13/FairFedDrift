from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fair_fed_drift.Client import Client
from federated.algorithms.fair_fed_drift.ClientData import ClientData
from federated.algorithms.fair_fed_drift.GlobalModels import GlobalModels, get_scales_dict, get_client_scales
from federated.algorithms.fair_fed_drift.clustering.ClusteringFactory import get_clustering_by_name
from federated.algorithms.fair_fed_drift.drift_detection.DriftDetectorFactory import get_detector_by_name
from federated.model import NN_model
from metrics.Loss import Loss
from metrics.MetricFactory import get_metrics, get_metrics_by_names


class FairFedDrift(Algorithm):
    # TODO - use loss and fairness loss instead of accuracy and balanced accuracy
    def __init__(self):
        self.metrics_clustering = None
        self.drift_detector = None
        self.clustering = None
        self.delta = 0.8
        name = "fair_fed_drift"
        super().__init__(name)

    def set_specs(self, args):
        self.clustering = get_clustering_by_name(args.clustering)
        self.drift_detector = get_detector_by_name(args.drift_detector)
        self.drift_detector.set_specs(args)
        self.metrics_clustering = get_metrics_by_names(args.metrics)
        metrics_string = "-".join(args.metrics)
        super().set_subfolders("{}/clustering-{}/drift_detector-{}/metrics-{}".format(
            self.name, self.clustering.name, self.drift_detector.name, metrics_string
        ))

    def perform_fl(self, seed, clients_data, dataset):
        global_models = GlobalModels()
        init_model = get_init_model(dataset, seed)
        init_clients = [Client(id, ClientData(x, y, s), 1) for id, [x, y, s, _] in enumerate(clients_data[0])]
        global_models.create_new_global_model(init_model, init_clients)
        clients_metrics = [get_metrics(dataset.is_image) for _ in range(dataset.n_clients)]

        for timestep in range(dataset.n_timesteps):
            # STEP 1 - Test each client's data on previous clustering identities
            self.test_models(global_models, clients_data[timestep], clients_metrics, dataset, seed)
            global_models.reset_clients()

            # STEP 2 - Recalculate Global Models (cluster identities) and detect concept drift
            global_models = self.update_global_models(clients_data[timestep], global_models, dataset, seed, timestep)

            # STEP 3 - Merge Global Models
            global_models = self.merge_global_models(global_models, dataset, seed)

            # STEP 4 - Train and average models
            global_models = self.train_and_average(clients_data[timestep], global_models, dataset, seed, timestep)


    def train_and_average(self, clients_data_timestep, global_models, dataset, seed, timestep):
        for cround in range(dataset.n_rounds):
            local_weights_list = [[] for _ in range(global_models.current_size)]
            local_amounts_list = [[] for _ in range(global_models.current_size)]
            local_sizes_list = [[] for _ in range(global_models.current_size)]
            
            for client_id, client_data in enumerate(clients_data_timestep):
                x, y, s, _ = client_data
                local_model, global_model_amounts = self.get_model_client(client_id, global_models, dataset, seed)
                local_model.learn(x, y)
                print("Trained model timestep {} cround {} client {}".format(timestep, cround, client_id))
                
                for global_model_id, amount in global_model_amounts:
                    local_weights_list[global_model_id].append(local_model.get_weights())
                    local_amounts_list[global_model_id].append(amount)
                    local_sizes_list[global_model_id].append(len(x))
            
            for global_model_id, (local_weights, local_amounts, local_sizes) in \
                    enumerate(zip(local_weights_list, local_amounts_list, local_sizes_list)):
                if len(local_weights) > 0:
                    scales = get_client_scales(local_amounts, local_sizes)
                    new_global_weights = super().average_weights(local_weights, scales)
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
        distances = [[0 for _ in range(size)] for _ in range(size)]
        scales_dict = get_scales_dict(global_models.models)

        for i in range(len(global_models.models)):
            for j in range(len(global_models.models)):
                id_i = global_models.models[i].id
                id_j = global_models.models[j].id
                if i != j and len(global_models.models[i].clients) > 0 and len(global_models.models[j].clients) > 0:
                    results = []
                    for client in global_models.models[j].clients:
                        print("Testing global model {} on data from cluster {} - client {}".format(
                            id_i, id_j, client.id)
                        )
                        partial_client_data = client.client_data.get_partial_data(scales_dict[id_j])
                        result = self.test_client_on_model(
                            [Loss()], global_models.models[i].model, partial_client_data, dataset.is_image
                        )[0]
                        results.append(result)
                    distances[id_i][id_j] = min(results)

        for i in range(len(global_models.models)):
            for j in range(len(global_models.models)):
                id_i = global_models.models[i].id
                id_j = global_models.models[j].id
                distances[id_i][id_j] = min([distances[id_i][id_j], distances[id_j][id_i]])

        while True:  # While we can still merge global models
            max_distance, ids = self.get_max_distance(distances)
            if max_distance > self.delta:
                print("Merging global model {} with global model {}".format(ids[0], ids[1]))
                global_model_0 = global_models.get_model(ids[0])
                global_model_1 = global_models.get_model(ids[1])
                scales_two_models = get_scales_dict([global_model_0, global_model_1])
                scales = [scales_two_models[ids[0]], scales_two_models[ids[1]]]
                weights = [global_model_0.get_weights(), global_model_1.get_weights()]
                new_global_model_weights = super().average_weights(weights, scales)
                new_global_model = get_init_model(dataset, seed)
                new_global_model.set_weights(new_global_model_weights)

                # Reset Distances
                distances[ids[0]][ids[1]] = 0
                distances[ids[1]][ids[0]] = 0

                # Create new global Model
                clients = global_model_0.clients
                clients.extend(global_model_1.clients)
                global_models.create_new_global_model(new_global_model, clients)

                # Reset Client old models
                global_models.reset_clients_merged_models(ids[0], ids[1])

            else:
                return global_models

    def get_max_distance(self, distances):
        max = 0
        ids = [0, 0]
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                if distances[i][j] > max:
                    max = distances[i][j]
                    ids = [i, j]

        return max, ids

    def update_global_models(self, clients_data_timestep, global_models, dataset, seed, timestep):

        for client_id, client_data in enumerate(clients_data_timestep):

            # Calculate results on all global models
            results_global_models = {}
            print("Client {} - testing on all global models".format(client_id))
            for global_model in global_models.models:
                print("Client {} - testing on global model id {}".format(client_id, global_model.id))
                results = self.test_client_on_model(
                    self.metrics_clustering, global_model.model, client_data, dataset.is_image
                )
                results_global_models[global_model] = results

            # Get Model for client given results_global_models and test
            model_weights, model_id_amounts = self.clustering.get_model_weights_for_client(results_global_models)
            model = get_init_model(dataset, seed)
            model.set_weights(model_weights)
            print("Client {} - testing on best model(s)".format(client_id))
            results_model = self.test_client_on_model(self.metrics_clustering, model, client_data, dataset.is_image)

            # Detect Drift
            drift_detected_metrics = self.drift_detector.drift_detected(timestep, results_model)
            x, y, s, _ = client_data
            if sum(drift_detected_metrics) >= 1:
                print("Drift detected at client {}\n".format(client_id))
                model = get_init_model(dataset, seed)
                clients = [Client(client_id, ClientData(x, y, s), 1)]
                global_models.create_new_global_model(model, clients)
            else:
                print("No drift detected at client {}\n".format(client_id))
                for global_model_id, amount in model_id_amounts.items():
                    client = Client(client_id, ClientData(x, y, s), amount)
                    global_models.set_client_model(global_model_id, client)

        return global_models

    def test_client_on_model(self, metrics, model, client_data, is_image):
        results = []
        x, y, s, _ = client_data
        pred = model.predict(x)
        y_true_original, y_pred_original, y_true, y_pred = super().get_y(y, pred, is_image)
        for client_metric in metrics:
            res = client_metric.calculate(y_true_original, y_pred_original, y_true, y_pred, s)
            print("{} - {:.2f}".format(client_metric.name, res))
            results.append(res)

        return results

    def test_models(self, global_models, clients_data_timestep, clients_metrics, dataset, seed):
        for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
            x, y, s, _ = client_data
            model, _ = self.get_model_client(client_id, global_models, dataset, seed)
            pred = model.predict(x)
            y_true_original, y_pred_original, y_true, y_pred = super().get_y(y, pred, dataset.is_image)
            for client_metric in client_metrics:
                res = client_metric.update(y_true_original, y_pred_original, y_true, y_pred, s)
                print(res, client_metric.name)
            for client_metric in self.metrics_clustering:
                client_metric.update(y_true_original, y_pred_original, y_true, y_pred, s)

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

        averaged_weights = super().average_weights(weights, amounts)
        model = get_init_model(dataset, seed)
        model.set_weights(averaged_weights)


        return model, global_model_amounts


def get_init_model(dataset, seed):
    model = NN_model(dataset.n_features, seed, dataset.is_image)
    model.compile(dataset.is_image)

    return model
