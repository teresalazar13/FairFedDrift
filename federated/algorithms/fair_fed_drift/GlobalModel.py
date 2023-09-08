import random

from federated.algorithms.fair_fed_drift.ClientData import ClientData


class GlobalModel:

    def __init__(self, model, id: int):
        self.model = model
        self.id = id
        self.n_points = 0
        self.clients = {}   # {client_id: ClientData}

    def set_client(self, client_id, client_data):
        self.clients[client_id] = client_data
        self.n_points += len(client_data.x)

    def reset_clients(self):
        self.clients = {}

    def get_partial_client_data(self, client_id):
        x = self.clients[client_id].x
        y = self.clients[client_id].y
        s = self.clients[client_id].s
        total_data = sum([len(data.x) for data in self.clients.values()])
        proportion = len(x) / total_data
        size = int(len(x) * proportion)
        perm = list(range(0, size))
        random.shuffle(perm)

        return ClientData(x[perm], y[perm], s[perm])
