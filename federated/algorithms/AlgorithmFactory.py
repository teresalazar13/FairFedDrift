from federated.algorithms.drift.fair_fed_drift import FairFedDrift
from federated.algorithms.drift.fed_drift import FedDrift
from federated.algorithms.fedavg import FedAvg
from federated.algorithms.oracle import Oracle


def get_algorithms():
    return [FedAvg(), Oracle(), FedDrift(), FairFedDrift()]


def get_algorithm_by_name(name):
    for alg in get_algorithms():
        if name == alg.name:
            return alg

    raise Exception("No Algorithm with the name ", name)
