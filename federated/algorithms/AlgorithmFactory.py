from federated.algorithms.fair_fed_drift.fed_drift import FedDrift
from federated.algorithms.fedval import FedVal
from federated.algorithms.fedavg import FedAvg
from federated.algorithms.fedavg_lr import FedAvgLR
from federated.algorithms.oracle import Oracle


def get_algorithms():
    return [FedDrift(), FedAvg(), FedAvgLR(), FedVal(), Oracle()]


def get_algorithm_by_name(name):
    for alg in get_algorithms():
        if name == alg.name:
            return alg

    raise Exception("No Algorithm with the name ", name)
