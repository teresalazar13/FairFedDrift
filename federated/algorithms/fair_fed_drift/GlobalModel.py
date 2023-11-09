import random
import numpy as np

from federated.algorithms.fair_fed_drift.ClientData import ClientData


class GlobalModel:

    def __init__(self, model, id, name):
        self.model = model
        self.id = id
        self.name = name
        self.n_points = 0
        self.clients = {}   # {client_id: client_data}

    def set_client(self, client_id, client_data):
        if client_id not in self.clients:
            self.clients[client_id] = client_data
        else:
            self.clients[client_id].x = np.concatenate((self.clients[client_id].x, client_data.x))
            self.clients[client_id].y = np.concatenate((self.clients[client_id].y, client_data.y))
            self.clients[client_id].s = np.concatenate((self.clients[client_id].s, client_data.s))

        self.n_points += len(client_data.x)

    def get_partial_client_data(self, client_id):
        x = self.clients[client_id].x
        y = self.clients[client_id].y
        s = self.clients[client_id].s

        proportion = len(x) / self.n_points
        size = int(len(x) * proportion)
        idx = np.random.choice(np.arange(len(x)), size, replace=False)

        return ClientData(x[idx], y[idx], s[idx])
