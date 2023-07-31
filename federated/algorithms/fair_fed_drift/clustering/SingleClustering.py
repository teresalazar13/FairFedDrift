from federated.algorithms.fair_fed_drift.clustering.Clustering import Clustering
from federated.model import NN_model

"""
    Identity with cluster with the best value of sum of Metrics chosen (M)
"""
class SingleClustering(Clustering):

    def __init__(self):
        name = "single"
        super().__init__(name)
        self.default_weight = 1

    def get_cluster_identities(self, values_clusters):
        best_values = []
        best_cluster_identity = -1

        for cluster_id, values in enumerate(values_clusters):
            if sum(values) > sum(best_values):
                best_values = values
                best_cluster_identity = cluster_id

        return [[best_cluster_identity, self.default_weight]]

    def get_model_cluster_identities(self, _, global_models, cluster_identities, seed, dataset):
        model = NN_model(dataset.n_features, seed, dataset.is_image)
        model.compile(dataset.is_image)
        model.set_weights(global_models[cluster_identities[0][0]].get_weights())

        return model

    def get_new_cluster_identities_drift(self, global_models, _, __):
        return [[len(global_models) - 1, self.default_weight]]
