from abc import abstractmethod


class Clustering:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_model_weights_for_client(self, results_global_models):
        raise NotImplementedError("Must implement get_model_weights_for_client")
