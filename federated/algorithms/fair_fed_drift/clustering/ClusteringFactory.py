from federated.algorithms.fair_fed_drift.clustering.SingleClustering import SingleClustering


def get_clusterings():
    return [SingleClustering()]


def get_clustering_by_name(name):
    for clustering in get_clusterings():
        if name == clustering.name:
            return clustering

    raise Exception("No Clustering with the name ", name)
