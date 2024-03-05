import logging

from federated.algorithms.Identity import Identity
from federated.algorithms.drift.GlobalModel import GlobalModel


class GlobalModels:

    def __init__(self):
        self.models = []
        self.n_models = 0

    def create_new_global_model(self, model, name_merge_a=None, name_merge_b=None):
        if name_merge_a and name_merge_b:  # if created from merge
            name = "{}-{}".format(name_merge_a, name_merge_b)
        else:
            name = str(self.n_models)
        id = self.n_models
        new_global_model = GlobalModel(model, Identity(id, name))
        self.models.append(new_global_model)
        self.n_models = self.n_models + 1
        logging.info("Created new Global Model with id: {}, name: {}".format(id, name))

        return new_global_model

    def get_model(self, global_model_id):
        for model in self.models:
            if model.identity.id == global_model_id:
                return model
        raise Exception("No model with id ", global_model_id)

    def deleted_merged_model(self, global_model_id):
        for model in self.models:
            if model.identity.id == global_model_id:
                self.models.remove(model)
