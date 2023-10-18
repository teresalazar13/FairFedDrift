from federated.algorithms.fair_fed_drift.GlobalModel import GlobalModel


class GlobalModels:

    def __init__(self):
        self.models = []
        self.new_models = 0
        self.current_size = 0

    def create_new_global_model(self, model, name_merge_a=None, name_merge_b=None):
        if name_merge_a and name_merge_b:  # if created from merge
            name = "{}-{}".format(name_merge_a, name_merge_b)
        else:
            name = str(self.new_models)
            self.new_models = self.new_models + 1
        id = self.current_size
        new_global_model = GlobalModel(model, id, name)
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

    def set_client_model(self, global_model_id, client, client_data):
        for model in self.models:
            if model.id == global_model_id:
                model.set_client(client, client_data)
