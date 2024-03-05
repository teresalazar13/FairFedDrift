import random
import numpy as np
from scipy.stats import multivariate_normal

from datasets.Dataset import Dataset


class Synthetic(Dataset):

    def __init__(self):
        name = "synthetic"
        input_shape = 3
        is_large = False
        is_binary_target = True
        is_image = False
        super().__init__(name, input_shape, is_large, is_binary_target, is_image)
        self.n_samples = 600

    def create_batched_data(self, varying_disc):
        drift_ids = self.drift_ids
        n_drifts = self.n_drifts
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        n_samples = self.get_n_samples_per_drift(drift_ids, n_drifts)
        drift_data = self.generate_drift_data(n_drifts, varying_disc, n_samples)

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

    def generate_drift_data(self, n_drifts, varying_disc, n_samples):
        drift_data = []

        for i in range(n_drifts):
            if i == 0:
                shift = 0
            elif i == 1:
                shift = 2
            elif i == 2:
                shift = -2
            else:
                raise Exception("Invalid drift")
            X_client, y_client, s_client = generate_synthetic_data(n_samples[i], varying_disc, shift)
            X_client = np.append(X_client, s_client.reshape((len(s_client), 1)), axis=1)
            drift_data.append([X_client, y_client, s_client, varying_disc])

        return drift_data


def generate_synthetic_data(n_samples, varying_disc, shift):
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

    n_privileged = int(n_samples / (2 + 2 * varying_disc))
    n_unprivileged = int(n_privileged * varying_disc)

    mu_pp, sigma_pp = [1, 1.5], [[2, 1], [1, 2]]  # privileged positive
    nv_pp, X_pp, y_pp = gen_gaussian(mu_pp, sigma_pp, 1, n_privileged)

    mu_pn, sigma_pn = [1, -1.5], [[2, 1], [1, 2]]  # privileged negative
    nv_pn, X_pn, y_pn = gen_gaussian(mu_pn, sigma_pn, 0, n_privileged)

    mu_up, sigma_up = [-1, 1 - shift], [[2, 1], [1, 2]]  # unprivileged positive
    nv_up, X_up, y_up = gen_gaussian(mu_up, sigma_up, 1, n_unprivileged)

    mu_un, sigma_un = [-1, -1 - shift], [[2, 1], [1, 2]]  # unprivileged negative
    nv_un, X_un, y_un = gen_gaussian(mu_un, sigma_un, 0, n_unprivileged)

    X = np.vstack((X_pp, X_pn, X_up, X_un))
    y = np.hstack((y_pp, y_pn, y_up, y_un))
    x_s = [1 for _ in range(len(X_pp) + len(X_pn))]
    x_s.extend([0 for _ in range(len(X_up) + len(X_un))])

    # shuffle the data
    perm = list(range(0, n_unprivileged*2 + n_privileged*2))
    random.shuffle(perm)
    X = X[perm]
    y = y[perm]
    x_s = np.array(x_s)[perm]
    #logging.info("DI Train: {:.2f}".format((len(X_up)/(len(X_up) + len(X_un)))/(len(X_pp)/(len(X_pp) + len(X_pn)))))

    return X, y, x_s
