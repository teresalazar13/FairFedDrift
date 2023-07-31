from federated.algorithms.fair_fed_drift.clustering.Clustering import Clustering
from federated.model import NN_model

"""
    Identify partially with each cluster based on value of sum of Metrics chosen (M)
"""
class PartialClustering(Clustering):

    def __init__(self):
        name = "partial"
        super().__init__(name)
        self.default_weight = 1

    def get_cluster_identities(self, values_clusters):
        sum_values = [sum(values) for values in values_clusters]
        cluster_identities = [[id, values/sum(sum_values)] for id, values in enumerate(sum_values)]

        return cluster_identities

    def get_model_cluster_identities(self, alg, global_models, cluster_identities, seed, dataset):
        weights_list = [model.get_weights() for model in global_models]
        scaling_factors = [cluster_identity[1] for cluster_identity in cluster_identities]
        model_weights = alg.average_weights(weights_list, scaling_factors)
        model = NN_model(dataset.n_features, seed, dataset.is_image)
        model.compile(dataset.is_image)
        model.set_weights(model_weights)

        return model

    def get_new_cluster_identities_drift(self, global_models, drift_detected_metrics, cluster_identities):
        return [self.default_weight for _ in range(len(global_models))]
        # identify with each cluster equally (e.g. 1) when drift detected
