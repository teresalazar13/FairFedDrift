from typing import List
import random
import numpy as np

from federated.algorithms.fair_fed_drift.Client import Client
from federated.algorithms.fair_fed_drift.ClientData import ClientData


class GlobalModel:

    def __init__(self, model, id: int, clients: List[Client]):
        self.model = model
        self.id = id
        self.clients = clients
        self.clients_data = {}   # {client_id: [[ClientData, amount]] (client_data_list), ...}
        # clients_data is data the model was "federatively trained" on

    def add_client_data(self, client_id, x, y, s, amount):
        if client_id in self.clients_data:
            self.clients_data[client_id].append([ClientData(x, y, s), amount])
        else:
            self.clients_data[client_id] = [[ClientData(x, y, s), amount]]

    def get_partial_client_data(self, client_id):
        total_data = 0
        total_client_data = 0
        x = []
        y = []
        s = []

        for client_id_2, client_data_list in self.clients_data.items():
            for client_data, amount in client_data_list:
                total_data += len(client_data.x) * amount

                if client_id == client_id_2:
                    total_client_data += len(client_data.x) * amount
                    x.append(client_data.x)
                    y.append(client_data.y)
                    s.append(client_data.s)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        s = np.concatenate(s, axis=0)
        proportion = total_client_data / total_data

        size = int(len(x) * proportion)
        perm = list(range(0, size))
        random.shuffle(perm)

        return ClientData(x[perm], y[perm], s[perm])

    def reset_clients(self):
        self.clients = []

    def set_client(self, client):
        self.clients.append(client)
