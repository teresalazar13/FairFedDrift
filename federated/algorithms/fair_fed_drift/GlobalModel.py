from typing import List

from federated.algorithms.fair_fed_drift.Client import Client


class GlobalModel:

    def __init__(self, model, id: int, clients: List[Client]):
        self.model = model
        self.id = id
        self.clients = clients

    def reset_clients(self):
        self.clients = []

    def set_client(self, client):
        self.clients.append(client)

    def get_clients_sum_sizes_amounts(self):
        sizes = []
        amounts = []

        for client in self.clients:
            sizes.append(len(client.client_data.x))
            amounts.append(client.amount)

        return sum(sizes), sum(amounts)
