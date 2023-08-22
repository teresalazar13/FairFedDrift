from typing import List

from federated.algorithms.fair_fed_drift.Client import Client
from federated.algorithms.fair_fed_drift.GlobalModel import GlobalModel


class GlobalModels:

    def __init__(self):
        self.models = []
        self.current_size = 0

    def create_new_global_model(self, model, clients: List[Client]):
        new_global_model = GlobalModel(model, self.current_size, clients)
        self.current_size = self.current_size + 1
        self.models.append(new_global_model)
        return new_global_model

    def get_model(self, global_model_id):
        for model in self.models:
            if model.id == global_model_id:
                return model
        raise Exception("No model with id ", global_model_id)

    def reset_clients(self):
        for model in self.models:
            model.reset_clients()

    def deleted_merged_model(self, global_model_id):
        for model in self.models:
            if model.id == global_model_id:
                self.models.remove(model)

    def set_client_model(self, global_model_id, client):
        for model in self.models:
            if model.id == global_model_id:
                model.set_client(client)


def get_models_proportions(global_models, global_model_id):
    total_data = 0

    for client_id, client_data_list in global_models.get_model(global_model_id).clients_data.items():
        for client_data, amount in client_data_list:
            total_data += len(client_data.x) * amount

    return total_data
