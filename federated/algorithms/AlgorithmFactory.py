from federated.algorithms.fair_fed_drift.fair_fed_drift_2 import FairFedDrift
from federated.algorithms.fedavg import FedAvg
from federated.algorithms.fedmom import FedMom
from federated.algorithms.fair_aggregation import FairAggregation
from federated.algorithms.fed_lr import FedLR


def get_algorithms():
    return [FairFedDrift(), FedAvg(), FedMom(), FairAggregation(), FedLR()]


def get_algorithm_by_name(name):
    for alg in get_algorithms():
        if name == alg.name:
            return alg

    raise Exception("No Algorithm with the name ", name)
