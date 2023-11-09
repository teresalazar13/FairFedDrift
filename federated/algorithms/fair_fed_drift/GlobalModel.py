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
            x = self.clients[client_id].x.tolist()
            x.extend(client_data.x.tolist())
            y = self.clients[client_id].y.tolist()
            y.extend(client_data.y.tolist())
            s = self.clients[client_id].s.tolist()
            s.extend(client_data.s.tolist())
            self.clients[client_id] = ClientData(np.array(x), np.array(y), np.array(s))

        self.n_points += len(client_data.x)

    def get_partial_client_data(self, client_id):
        x = self.clients[client_id].x
        y = self.clients[client_id].y
        s = self.clients[client_id].s

        proportion = len(x) / self.n_points
        size = int(len(x) * proportion)
        perm = list(range(0, size))
        random.shuffle(perm)

        return ClientData(np.array(x)[perm], np.array(y)[perm], np.array(s)[perm])
