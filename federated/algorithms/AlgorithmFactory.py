from federated.algorithms.fair_aggregation import FairAggregation
from federated.algorithms.fairfedavg import FairFedAvg
from federated.algorithms.fedavg import FedAvg
from federated.algorithms.fedmom import FedMom


def get_algorithms():
    return [FedAvg(), FairFedAvg(), FairAggregation(), FedMom()]


def get_algorithm_by_name(name):
    for alg in get_algorithms():
        if name == alg.name:
            return alg

    raise Exception("No Algorithm with the name ", name)
