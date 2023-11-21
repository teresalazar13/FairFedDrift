import os


class Dataset:

    def __init__(self, name, input_shape, is_large, is_binary_target):
        self.name = name
        self.input_shape = input_shape
        self.is_large = is_large
        self.is_binary_target = is_binary_target
        self.n_rounds = 10  # number of rounds per timestep

        if self.is_binary_target:
            drift_ids = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # timestep 0
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # timestep 1
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # timestep 2 -> CONCEPT DRIFT
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # timestep 3
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # timestep 4
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # timestep 5 -> CONCEPT DRIFT
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # timestep 6
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # timestep 7
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # timestep 8 -> CONCEPT DRIFT
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # timestep 9
            ]
        else:
            drift_ids = [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # timestep 0
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # timestep 1
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # timestep 2 -> CONCEPT DRIFT
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # timestep 3
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # timestep 4
                [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],  # timestep 5 -> CONCEPT DRIFT
                [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],  # timestep 6
                [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],  # timestep 7
                [2, 2, 2, 0, 0, 0, 0, 0, 1, 1],  # timestep 8 -> CONCEPT DRIFT
                [2, 2, 2, 0, 0, 0, 0, 0, 1, 1],  # timestep 9
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
        algs = []
        for x in os.walk(folder):
            if len(x) > 1 and "client_1" in x[0] and "results.csv" in x[-1]:
                f = "/".join(x[0].split("/")[:-1])
                a = ";".join(x[0].split("/")[4:-1])
                if f not in folders:
                    folders.append(f)
                    algs.append(a)

        return folder, folders, algs
