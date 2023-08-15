import random
import numpy as np
from scipy.stats import multivariate_normal

from datasets.Dataset import Dataset
from plot.plot import plot_synthetic_data


class Synthetic(Dataset):

    def __init__(self):
        name = "synthetic"
        n_features = 3
        super().__init__(name, n_features)
        self.n_samples = 5000
        self.is_image = False

    def create_batched_data(self, algorithm_subfolders, varying_disc):
        drift_ids = self.drift_ids
        n_drifts = self.n_drifts
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        n_samples = self.get_n_samples_per_drift(drift_ids, n_drifts)
        drift_data = self.generate_drift_data(algorithm_subfolders, n_drifts, varying_disc, n_samples)

        batched_data = []
        for i in range(n_timesteps):
            batched_data_round = []
            for j in range(n_clients):
                drift_data_id = drift_data[drift_ids[i][j]]
                drift_data_id_client_round = [
                    drift_data_id[0][:self.n_samples],
                    drift_data_id[1][:self.n_samples],
                    drift_data_id[2][:self.n_samples],
                    drift_data_id[1][:self.n_samples]
                ]
                batched_data_round.append(drift_data_id_client_round)  # add data
                drift_data_id[0] = drift_data_id[0][self.n_samples:]  # remove added data
                drift_data_id[1] = drift_data_id[1][self.n_samples:]
                drift_data_id[2] = drift_data_id[2][self.n_samples:]
            batched_data.append(batched_data_round)

        return batched_data

    def get_n_samples_per_drift(self, drift_ids, n_drifts):
        n_samples = [0 for _ in range(n_drifts)]
        for drift_ids_timestep in drift_ids:
            for drift_id_client in drift_ids_timestep:
                n_samples[drift_id_client] += self.n_samples

        return n_samples

    def generate_drift_data(self, algorithm_subfolders, n_drifts, varying_disc, n_samples):
        drift_data = []
        up = 0
        right = 0

        for i in range(n_drifts):
            print("disc {} | up: {} | right: {} | n_samples: {}".format(varying_disc, up, right, n_samples[i]))
            X_client, y_client, s_client = generate_synthetic_data(n_samples[i], varying_disc, right, up)
            X_client = np.append(X_client, s_client.reshape((len(s_client), 1)), axis=1)
            drift_data.append([X_client, y_client, s_client, varying_disc, right, up])
            up += 0.2
            right += 0.2
        filename = "{}/data.png".format(self.get_folder(algorithm_subfolders, varying_disc))
        plot_synthetic_data(drift_data, n_drifts, n_samples, filename)

        return drift_data


def generate_synthetic_data(n_samples, varying_disc, right, up):
    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and
        disc_0.5 means it's in non-protected group (e.g., male).
    """
    SEED = 1122334455
    random.seed(SEED)
    np.random.seed(SEED)

    def gen_gaussian(mean_in, cov_in, class_label, n_samples):
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv, X, y

    n_up = int((n_samples // 4) * varying_disc)
    n_others = (n_samples - n_up) // 3
    if n_others*3 + n_up != n_samples:
        n_up += n_samples - (n_others*3 + n_up)

    mu_pp, sigma_pp = [2.5, 2.5], [[3, 1], [1, 3]]  # privileged positive
    nv_pp, X_pp, y_pp = gen_gaussian(mu_pp, sigma_pp, 1, n_others)

    mu_pn, sigma_pn = [-2.5, -2.5], [[3, 1], [1, 3]]  # privileged negative
    nv_pn, X_pn, y_pn = gen_gaussian(mu_pn, sigma_pn, 0, n_others)

    mu_up, sigma_up = [2 - right, 2 - up], [[3, 1], [1, 3]]  # unprivileged positive
    nv_up, X_up, y_up = gen_gaussian(mu_up, sigma_up, 1, n_up)

    mu_un, sigma_un = [0 + right, 0 + up], [[3, 1], [1, 3]]  # unprivileged negative
    nv_un, X_un, y_un = gen_gaussian(mu_un, sigma_un, 0, n_others)

    X = np.vstack((X_pp, X_pn, X_up, X_un))
    y = np.hstack((y_pp, y_pn, y_up, y_un))
    x_s = [0 for _ in range(len(X_pp) + len(X_pn))]
    x_s.extend([1 for _ in range(len(X_up) + len(X_un))])

    # shuffle the data
    perm = list(range(0, n_samples))
    random.shuffle(perm)
    X = X[perm]
    y = y[perm]
    x_s = np.array(x_s)[perm]

    print("DI Train: {:.2f}".format((len(X_up)/(len(X_up) + len(X_un)))/(len(X_pp)/(len(X_pp) + len(X_pn)))))

    return X, y, x_s
