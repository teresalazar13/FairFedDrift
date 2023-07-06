import random
import os
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical


class FairMNIST:

    def __init__(self):
        self.name = "fairMNIST"
        self.size = 60000
        self.is_image = True

    def get_folder(self, alg, n_drifts, varying_disc):
        return "./results/{}/n-drifts_{}/disc_{}/{}".format(self.name, n_drifts, varying_disc, alg)

    def get_all_folders(self, n_drifts, varying_disc):
        folder = "./results/{}/n-drifts_{}/disc_{}".format(self.name, n_drifts, varying_disc)
        algs = [x for x in os.listdir(folder) if not x.startswith('.') and "." not in x]
        folders = ["{}/{}".format(folder, x) for x in algs]

        return folder, folders, algs

    def create_batched_data(self, _, n_drifts, varying_disc, n_clients, n_timesteps):
        drift_ids = self.generate_drift_ids(n_clients, n_timesteps, n_drifts)
        (train_X_priv, train_y_priv), (__, _) = mnist.load_data()
        batched_data = []
        drift_ids_col = [[] for _ in range(n_clients)]
        train_X_priv_rounds = np.array_split(train_X_priv, n_timesteps)
        train_y_priv_rounds = np.array_split(train_y_priv, n_timesteps)

        for i in range(n_timesteps):
            batched_data_round = []
            train_X_priv_round_clients = np.array_split(train_X_priv_rounds[i], n_clients)
            train_y_priv_round_clients = np.array_split(train_y_priv_rounds[i], n_clients)
            for j in range(n_clients):
                drift_id = drift_ids[i][j]
                train_X_priv_round_client = train_X_priv_round_clients[j]
                train_y_priv_round_client = train_y_priv_round_clients[j]
                size_unpriv = round(len(train_X_priv_round_client) * varying_disc)
                train_X_unpriv_round_client = np.rot90(train_X_priv_round_client.copy() * -1, axes=(-2, -1))
                for n in range(len(train_y_priv_round_client)):
                    if train_y_priv_round_client[n] == 7:
                        plt.imshow(train_X_unpriv_round_client[n])
                        plt.show()
                        plt.imshow(train_X_priv_round_client[n])
                        plt.show()
                        exit()
                train_y_unpriv_round_client = train_y_priv_round_client.copy()
                if drift_id != 0:
                    train_y_unpriv_round_client[train_y_unpriv_round_client == drift_id] = 100
                    train_y_unpriv_round_client[train_y_unpriv_round_client == drift_id + 1] = 101
                    train_y_unpriv_round_client[train_y_unpriv_round_client == 100] = drift_id + 1
                    train_y_unpriv_round_client[train_y_unpriv_round_client == 101] = drift_id
                train_X = np.concatenate((train_X_priv_round_client[size_unpriv:], train_X_unpriv_round_client[:size_unpriv]), axis=0)
                train_y = np.concatenate((train_y_priv_round_client[size_unpriv:], train_y_unpriv_round_client[:size_unpriv]), axis=0)
                train_s = [1] * (len(train_X_priv_round_client) - size_unpriv) + [0] * size_unpriv
                train_X = train_X.astype('float32') / 255.0
                train_y = to_categorical(train_y)
                perm = list(range(0, len(train_X)))
                random.shuffle(perm)
                train_X = train_X[perm]
                train_y = train_y[perm]
                train_s = np.array(train_s)[perm]
                batched_data_round.append([train_X, train_y, train_s])
                drift_ids_col[j].append(drift_ids[i][j])
            batched_data.append(batched_data_round)

        return batched_data, drift_ids_col, 3

    def generate_drift_ids(self, n_clients, n_timesteps, n_drifts):
        drift_ids = [
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)],
            [0 for _ in range(n_clients)]
        ]

        for i in range(3, n_timesteps):
            drift_id_round = []
            for j in range(n_clients):
                if n_drifts > 0 and random.random() > 0.5:
                    choices = list(range(n_drifts))
                    choices.remove(drift_ids[i - 1][j])
                    drift_id = random.choice(choices)
                    print("Drift change at round", i, "client", j)
                else:
                    drift_id = drift_ids[i - 1][j]  # get previous drift id
                drift_id_round.append(drift_id)
            drift_ids.append(drift_id_round)
        print(drift_ids)

        return drift_ids


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)

    return arr
