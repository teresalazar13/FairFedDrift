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

    def get_model(self, global_model_id):
        for model in self.models:
            if model.id == global_model_id:
                return model
        raise Exception("No model with id ", global_model_id)

    def reset_clients(self):
        for model in self.models:
            model.reset_clients()

    def reset_clients_merged_models(self, global_model_id_0, global_model_id_1):
        for model in self.models:
            if global_model_id_0 == model.id or global_model_id_1 == model.id:
                model.reset_clients()

    def set_client_model(self, global_model_id, client):
        for model in self.models:
            if model.id == global_model_id:
                model.set_client(client)

def get_models_proportions(global_models, global_model_id):
    total_data = 0

    for client_id, client_data_list in global_models.get_model(global_model_id).clients_data:
        for client_data, amount in client_data_list:
            total_data += len(client_data.x) * amount

    return total_data


def get_client_scales(local_amounts, local_sizes):
    scales = []

    for amount, size in zip(local_amounts, local_sizes):
        size_scale = size / sum(local_sizes)
        amount_scale = amount / sum(local_amounts)
        scale = calculate_scale(size_scale, amount_scale)
        scales.append(scale)

    return scales

