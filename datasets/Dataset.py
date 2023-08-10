import glob
import os

from federated.algorithms.AlgorithmFactory import get_algorithm_by_name


class Dataset:

    def __init__(self, name, n_features):
        self.name = name
        self.n_features = n_features
        self.n_rounds = 10  # number of rounds per timestep
        drift_ids = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # timestep 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # timestep 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # timestep 3  # TODO - update image (remove 10, add 0 in 3rd)
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],  # timestep 4
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2],  # timestep 5
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],  # timestep 6
            [2, 2, 0, 1, 1, 1, 2, 2, 2, 1],  # timestep 7
            [2, 2, 0, 0, 2, 0, 0, 1, 1, 1],  # timestep 8
            [0, 2, 0, 0, 2, 2, 0, 1, 1, 1],  # timestep 9
            [0, 1, 2, 1, 2, 2, 2, 2, 0, 0],  # timestep 10
        ]
        self.drift_ids = drift_ids
        self.drift_ids_col, self.n_clients, self.n_drifts, self.n_timesteps = self.get_drift_ids_col(drift_ids)

    def get_drift_ids_col(self, drift_ids):
        n_clients = len(drift_ids[0])
        drifts = set()
        drift_ids_col = [[] for _ in range(n_clients)]
        for timestep in range(len(drift_ids)):
            for client in range(len(drift_ids[timestep])):
                drift_id = drift_ids[timestep][client]
                drift_ids_col[client].append(drift_id)
                drifts.add(drift_id)

        return drift_ids_col, n_clients, len(drifts), len(drift_ids)

    def get_folder(self, algorithm_subfolders, varying_disc):
        return "./results/{}/disc_{}/{}".format(self.name, varying_disc, algorithm_subfolders)

    def get_all_folders(self, varying_disc):
        folder = "./results/{}/disc_{}".format(self.name, varying_disc)
        folders = []
        for x in os.walk(folder):
            if len(x) > 1 and "client_1" in x[1]:
                folders.append(x[0])

        return folder, folders, folders  # TODO
