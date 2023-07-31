from federated.algorithms.Algorithm import Algorithm
from federated.algorithms.fair_fed_drift.clustering.ClusteringFactory import get_clustering_by_name
from federated.algorithms.fair_fed_drift.drift_detection.DriftDetectorFactory import get_detector_by_name
from federated.model import NN_model
from metrics.MetricFactory import get_metrics, get_metrics_by_names


class FairFedDrift(Algorithm):

    def __init__(self):
        self.metrics_clustering = None
        self.drift_detector = None
        self.clustering = None
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
        initial_global_model = NN_model(dataset.n_features, seed, dataset.is_image)
        global_models = [initial_global_model]
        clients_metrics = [get_metrics(dataset.is_image) for _ in range(dataset.n_clients)]
        client_identities = [[] for _ in range(dataset.n_clients)]
        cluster_identities_clients = [[[0, 1]] for _ in range(dataset.n_clients)]  # first

        for timestep in range(dataset.n_timesteps):
            # STEP 1 - Test each client's data on previous clustering identities
            self.test_models(
                global_models, clients_data[timestep], clients_metrics, seed, dataset, cluster_identities_clients
            )

            # STEP 2 - Determine cluster identities and detect concept drift  # TODO - merge and delete old
            cluster_identities_clients, global_models = self.get_new_cluster_identities(
                clients_data[timestep], global_models, seed, dataset, timestep
            )
            print("Cluster identities timestep {} - {}".format(timestep, cluster_identities_clients))
            [client_identities[i].append(id) for i, id in enumerate(cluster_identities_clients)]

            # STEP 3 - Train and average models
            self.train_and_average(timestep, dataset, clients_data, cluster_identities_clients, global_models, seed)

        return clients_metrics, client_identities

    def test_models(self, global_models, clients_data_timestep, clients_metrics, seed, dataset, cluster_identities_clients):
        for client_id, (client_data, client_metrics) in enumerate(zip(clients_data_timestep, clients_metrics)):
            x, y, s, _ = client_data
            model = self.clustering.get_model_cluster_identities(
                self, global_models, cluster_identities_clients[client_id], seed, dataset
            )
            pred = model.predict(x)
            y, pred = super().get_y(y, pred, dataset.is_image)
            for client_metric in client_metrics:
                res = client_metric.update(y, pred, s)
                print(res, client_metric.name)
            for client_metric in self.metrics_clustering:
                client_metric.update(y, pred, s)

    def get_new_cluster_identities(self, clients_data_timestep, global_models, seed, dataset, timestep):
        cluster_identities_clients = []
        original_size = len(global_models)

        for client_id, client_data in enumerate(clients_data_timestep):
            x, y, s, _ = client_data

            # Calculate metrics on all clusters
            values_clusters = []
            print("Client {} - checking best cluster id".format(client_id))
            for cluster_id, global_model in enumerate(global_models[:original_size]):
                pred = global_model.predict(x)
                y_, pred = super().get_y(y, pred, dataset.is_image)
                values = []
                print("Client {} - testing on cluster id {}".format(client_id, cluster_id))
                for client_metric in self.metrics_clustering:
                    value = client_metric.calculate(y_, pred, s)
                    print("{} - {:.2f}".format(client_metric.name, value))
                    values.append(value)
                values_clusters.append(values)

            # Assign client to best cluster(s)
            cluster_identities = self.clustering.get_cluster_identities(values_clusters)

            # Calculate results on best model(s)
            model = self.clustering.get_model_cluster_identities(self, global_models, cluster_identities, seed, dataset)
            pred = model.predict(x)
            y_, pred = super().get_y(y, pred, dataset.is_image)
            values_best = []
            print("Client {} - testing on best model {}".format(client_id, cluster_identities))
            for client_metric in self.metrics_clustering:
                value = client_metric.calculate(y_, pred, s)
                values_best.append(value)
                print("{} - {:.2f}".format(client_metric.name, value))

            drift_detected_metrics = self.drift_detector.drift_detected(timestep, values_best)
            if sum(drift_detected_metrics) >= 1:
                global_models.append(NN_model(dataset.n_features, seed, dataset.is_image))
                new_cluster_identities = self.clustering.get_new_cluster_identities_drift(
                    global_models, drift_detected_metrics, cluster_identities
                )
                cluster_identities_clients.append(new_cluster_identities)
            else:
                print("No drift detected at client {}".format(client_id))
                cluster_identities_clients.append(cluster_identities)

        return cluster_identities_clients, global_models

    def train_and_average(self, timestep, dataset, clients_data, cluster_identities_clients, global_models, seed):
        for cround in range(dataset.n_rounds):
            local_weights_lists = [[] for _ in range(len(global_models))]
            client_scaling_factors_lists = [[] for _ in range(len(global_models))]

            for client in range(dataset.n_clients):
                x, y, s, _ = clients_data[timestep][client]
                cluster_identities = cluster_identities_clients[client]
                local_model = self.clustering.get_model_cluster_identities(
                    self, global_models, cluster_identities, seed, dataset
                )
                local_model.learn(x, y)
                for [cluster_id, weight] in cluster_identities:
                    local_weights_lists[cluster_id].append(local_model.get_weights())
                    client_scaling_factors_lists[cluster_id].append(weight)
                # K.clear_session()
                print("Trained model timestep {} cround {} client {} with identities {}".format(
                    timestep, cround, client, cluster_identities)
                )

            for i, (weights, scales) in enumerate(zip(local_weights_lists, client_scaling_factors_lists)):
                if len(weights) > 0:
                    new_global_weights = super().average_weights(weights, scales)
                    global_models[i].set_weights(new_global_weights)
                    print("Averaged models on timestep {} cround {} of cluster identity {}".format(timestep, cround, i))
                else:
                    print("Did not average models on timestep {} cround {} of cluster identity {}".format(
                        timestep, cround, i)
                    )
