from federated.algorithms.fair_fed_drift.ClientData import ClientData


class Client:

    def __init__(self, id: int, client_data: ClientData, amount: float):
        self.id = id
        self.client_data = client_data
        self.amount = amount
