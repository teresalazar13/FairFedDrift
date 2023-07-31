from abc import abstractmethod


class Clustering:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calculate_cluster_identities(self, values_clusters):
        raise NotImplementedError("Must implement calculate_cluster_identities")

    @abstractmethod
    def get_model_cluster_identities(self, alg, global_models, cluster_identities, seed, dataset):
        raise NotImplementedError("Must implement get_model_cluster_identities")

    @abstractmethod
    def get_new_cluster_identities_drift(self, global_models, drift_detected_metrics, cluster_identities):
        raise NotImplementedError("Must implement get_new_cluster_identities_drift")
