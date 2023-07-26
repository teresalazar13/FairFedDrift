import os


class Dataset:

    def __init__(self, name, n_features):
        self.name = name
        self.n_features = n_features
        self.n_rounds = 10
        drift_ids = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 1, 2],
            [2, 2, 1, 1],
            [2, 2, 2, 1],
            [2, 2, 2, 0],
            [1, 2, 2, 0],
            [2, 0, 2, 2]
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

    def get_folder(self, alg, varying_disc):
        return "./results/{}/disc_{}/{}".format(self.name, varying_disc, alg)

    def get_all_folders(self, n_drifts, varying_disc):
        folder = "./results/{}/n-drifts_{}/disc_{}".format(self.name, n_drifts, varying_disc)
        algs = [x for x in os.listdir(folder) if not x.startswith('.') and "." not in x]
        folders = ["{}/{}".format(folder, x) for x in algs]

        return folder, folders, algs
