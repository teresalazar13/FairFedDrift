import numpy as np
from scipy.stats import multivariate_normal
import random

from datasets.Feature import Feature
from plot.plot import plot_synthetic_data


class SyntheticDataset:

    def __init__(self):
        self.name = "synthetic"
        self.sensitive_attribute = Feature("S", [1], [0])
        self.target = Feature("Y", 1, 0)
        self.n_samples = 2000
        self.is_image = False

    def get_folder(self, alg, n_drifts, varying_disc):
        return "./results_online/{}/{}/n-drifts_{}/disc_{}".format(self.name, alg, n_drifts, varying_disc)

    def create_batched_data(self, alg, n_drifts, varying_disc, n_clients, n_rounds):
        drift_ids, n_samples = self.generate_drift_ids(n_clients, n_rounds, n_drifts)
        drift_data = self.generate_drift_data(alg, n_drifts, varying_disc, n_samples)
        drift_ids_col = [[] for _ in range(n_clients)]

        batched_data = []
        for i in range(n_rounds):
            batched_data_round = []
            for j in range(n_clients):
                drift_data_id = drift_data[drift_ids[i][j]]
                drift_data_id_client_round = [
                    drift_data_id[0][:self.n_samples],
                    drift_data_id[1][:self.n_samples],
                    drift_data_id[2][:self.n_samples]
                ]
                batched_data_round.append(drift_data_id_client_round)  # add data
                drift_data_id[0] = drift_data_id[0][self.n_samples:]  # remove added data
                drift_data_id[1] = drift_data_id[1][self.n_samples:]
                drift_data_id[2] = drift_data_id[2][self.n_samples:]
                for _ in range(self.n_samples):
                    drift_ids_col[j].append(drift_ids[i][j])
            batched_data.append(batched_data_round)

        return batched_data, drift_ids_col, 3

    def generate_drift_ids(self, n_clients, n_rounds, n_drifts):
        """
        drift_ids = [
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)]
        ]  # start with the same concept (0)
        n_samples = [self.n_samples * n_clients * 5]
        n_samples.extend(
            [0 for _ in range(n_drifts - 1)]
        )

        for i in range(5, n_rounds):
            drift_id_round = []
            for j in range(n_clients):
                if random.random() > 0.5:  # 50% chance of changing concept
                    choices = list(range(n_drifts))
                    choices.remove(drift_ids[i - 1][j])
                    drift_id = random.choice(choices)
                    print("Drift change at round", i, "client", j)
                else:
                    drift_id = drift_ids[i - 1][j]  # get previous drift id
                drift_id_round.append(drift_id)
                n_samples[drift_id] += self.n_samples
            drift_ids.append(drift_id_round)

        print(drift_ids)"""

        drift_ids = [
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]  # start with the same concept (0)
        n_samples = [self.n_samples * 26, self.n_samples * 14]
        print(drift_ids)

        return drift_ids, n_samples

    def generate_drift_data(self, alg, n_drifts, varying_disc, n_samples):
        drift_data = []
        up = 1.0
        right = 1.0

        for i in range(n_drifts):
            if varying_disc:
                disc = random.random()
            else:
                disc = 0.8
            print("disc {} | up: {} | right: {} | n_samples: {}".format(disc, up, right, n_samples[i]))
            X_client, y_client, s_client = generate_synthetic_data(n_samples[i], disc, right, up)
            X_client = np.append(X_client, s_client.reshape((len(s_client), 1)), axis=1)
            drift_data.append([X_client, y_client, s_client, disc, right, up])
            up += 0.5
            right += 0.5
        filename = "{}/data.png".format(self.get_folder(alg, n_drifts, varying_disc))
        plot_synthetic_data(drift_data, n_drifts, n_samples, filename)

        return drift_data


def generate_synthetic_data(n_samples, disc, right, up):
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

    n_up = int((n_samples // 4) * disc)
    n_others = (n_samples - n_up) // 3
    if n_others*3 + n_up != n_samples:
        n_up += n_samples - (n_others*3 + n_up)

    mu_pp, sigma_pp = [2.5 + right, 2.5 + up], [[3, 1], [1, 3]]  # privileged positive
    nv_pp, X_pp, y_pp = gen_gaussian(mu_pp, sigma_pp, 1, n_others)

    mu_pn, sigma_pn = [-2 + right, -2 + up], [[3, 1], [1, 3]]  # privileged negative
    nv_pn, X_pn, y_pn = gen_gaussian(mu_pn, sigma_pn, 0, n_others)

    mu_up, sigma_up = [2 + right, 2 + up], [[3, 1], [1, 3]]  # unprivileged positive
    nv_up, X_up, y_up = gen_gaussian(mu_up, sigma_up, 1, n_up)

    mu_un, sigma_un = [0.5 + right, 0.5 + up], [[3, 3], [1, 3]]  # unprivileged negative
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
