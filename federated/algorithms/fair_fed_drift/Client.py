from federated.algorithms.fair_fed_drift.ClientData import ClientData


class Client:

    def __init__(self, id: int, amount: float):
        self.id = id
        self.amount = amount
