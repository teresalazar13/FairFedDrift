from abc import abstractmethod


class Algorithm:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def perform_fl(self, n_timesteps, n_crounds, n_clients, n_features, clients_data, seed, is_image):
        raise NotImplementedError("Must implement perform_fl")
