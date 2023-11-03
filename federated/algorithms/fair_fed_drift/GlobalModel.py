import random

import numpy as np

from federated.algorithms.fair_fed_drift.ClientData import ClientData


class GlobalModel:

    def __init__(self, model, id, name):
        self.model = model
        self.id = id
        self.name = name
        self.n_points = 0
        self.clients = {}   # {client_id: ClientData} # {client_id: {timestep: ClientData}} -> when has client identity

    def set_client(self, client_id, client_data, timestep):
        if client_id in self.clients:
            self.clients[client_id][timestep] = client_data
        else:
            self.clients[client_id] = {timestep: client_data}
        self.n_points += len(client_data.x)
    def get_partial_client_data(self, client_id, limit_timestep):
        x = []
        y = []
        s = []
        for t, client_data in self.clients[client_id].items():
            if t <= limit_timestep:
                x.extend(client_data.x)
                y.extend(client_data.y)
                s.extend(client_data.s)
        proportion = len(x) / self.n_points
        size = int(len(x) * proportion)
        perm = list(range(0, size))
        random.shuffle(perm)

        if x:
            return ClientData(np.array(x)[perm], np.array(y)[perm], np.array(s)[perm])
        return None
