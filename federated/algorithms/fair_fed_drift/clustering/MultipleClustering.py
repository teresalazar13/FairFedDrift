from federated.algorithms.fair_fed_drift.clustering.Clustering import Clustering
from federated.algorithms.fedavg import average_weights
from federated.model import NN_model

"""
    Identify with M clusters
        For each Metric m - Identify with the cluster with the best value of Metric m
"""
class MultipleClustering(Clustering):

    def __init__(self):
        name = "multiple"
        super().__init__(name)
        self.default_weight = 1

    def get_cluster_identities(self, values_clusters):
        best_values = [0 for _ in range(len(values_clusters[0]))]  # array of size M
        best_cluster_ids = [[0, self.default_weight] for _ in range(len(values_clusters[0]))]  # array of size M

        for i in range(len(values_clusters)):  # For each cluster result
            for j in range(len(values_clusters[i])):  # For each metric m
                if values_clusters[i][j] > best_values[j]:
                    best_values[j] = values_clusters[i][j]
                    best_cluster_ids[j] = [i, self.default_weight]

        return best_cluster_ids

    def get_model_cluster_identities(self, global_models, cluster_identities, seed, dataset):
        weights_list = [global_models[id].get_weights() for [id, _] in cluster_identities]
        scaling_factors = [self.default_weight for _ in range(len(cluster_identities))]  # array of size M
        model_weights = average_weights(weights_list, scaling_factors)
        model = NN_model(dataset.n_features, seed, dataset.is_image)
        model.compile(dataset.is_image)
        model.set_weights(model_weights)

        return model

    def get_new_cluster_identities_drift(self, global_models, drift_detected_metrics, cluster_identities):
        new_cluster_identities = []
        for drift_detected, [id, _] in zip(drift_detected_metrics, cluster_identities):
            if drift_detected:
                new_cluster_identities.append(len(global_models - 1))
            else:
                new_cluster_identities.append(id)

        return new_cluster_identities
