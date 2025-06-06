import os


class Dataset:

    def __init__(self, name, input_shape, is_pt, n_classes, is_image=True):
        self.name = name
        self.input_shape = input_shape
        self.is_pt = is_pt
        self.n_classes = n_classes
        self.n_rounds = 10  # number of rounds per timestep
        self.is_image = is_image

    def set_drifts(self, scenario):
        drift_ids = get_drift_ids(scenario)
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

    def get_folder(self, scenario, algorithm_subfolders, varying_disc):
        return "./results/scenario-{}/{}/disc_{}/{}".format(scenario, self.name, varying_disc, algorithm_subfolders)


def get_drift_ids(scenario):
    f = open("./datasets/scenarios/{}.csv".format(scenario))
    drift_ids = f.read().split("\n")
    drift_ids = [d.split(",") for d in drift_ids]
    drift_ids = [[int(a) for a in b] for b in drift_ids]
    f.close()

    return drift_ids
