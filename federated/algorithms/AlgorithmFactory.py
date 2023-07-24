from federated.algorithms.fair_aggregation import FairAggregation
from federated.algorithms.fairfedavg import FairFedAvg
from federated.algorithms.fairfedavg_mom import FairFedAvgMom
from federated.algorithms.fed_lr import FedLR
from federated.algorithms.fedavg import FedAvg
from federated.algorithms.fedmom import FedMom


def get_algorithms():
    return [FedAvg(), FairFedAvg(), FairAggregation(), FedMom(), FairFedAvgMom(), FedLR()]


def get_algorithm_by_name(name):
    for alg in get_algorithms():
        if name == alg.name:
            return alg

    raise Exception("No Algorithm with the name ", name)
