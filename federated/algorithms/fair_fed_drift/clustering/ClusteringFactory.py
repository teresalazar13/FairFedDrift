from federated.algorithms.fair_fed_drift.clustering.PartialClustering import PartialClustering
from federated.algorithms.fair_fed_drift.clustering.SingleClustering import SingleClustering


def get_clusterings():
    return [SingleClustering(), PartialClustering()]


def get_clustering_by_name(name):
    for clustering in get_clusterings():
        if name == clustering.name:
            return clustering

    raise Exception("No Clustering with the name ", name)
