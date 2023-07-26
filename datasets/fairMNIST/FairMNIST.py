import random
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from datasets.Dataset import Dataset


class FairMNIST(Dataset):

    def __init__(self):
        name = "fairMNIST"
        n_features = None
        super().__init__(name, n_features)
        self.size = 60000
        self.is_image = True

    def create_batched_data(self, _, varying_disc):
        drift_ids = self.drift_ids
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        (train_X_priv, train_y_priv), (__, _) = mnist.load_data()
        batched_data = []
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
                train_y_unpriv_round_client = train_y_priv_round_client.copy()
                if drift_id != 0:
                    train_y_unpriv_round_client[train_y_unpriv_round_client == drift_id] = 100
                    train_y_unpriv_round_client[train_y_unpriv_round_client == drift_id + 1] = 101
                    train_y_unpriv_round_client[train_y_unpriv_round_client == 100] = drift_id + 1
                    train_y_unpriv_round_client[train_y_unpriv_round_client == 101] = drift_id
                train_X = np.concatenate((train_X_priv_round_client[size_unpriv:], train_X_unpriv_round_client[:size_unpriv]), axis=0)
                train_y_original = np.concatenate((train_y_priv_round_client[size_unpriv:], train_y_unpriv_round_client[:size_unpriv]), axis=0)
                train_s = [1] * (len(train_X_priv_round_client) - size_unpriv) + [0] * size_unpriv
                train_X = train_X.astype('float32') / 255.0
                train_y = to_categorical(train_y_original)
                perm = list(range(0, len(train_X)))
                random.shuffle(perm)
                train_X = train_X[perm]
                train_y = train_y[perm]
                train_s = np.array(train_s)[perm]
                batched_data_round.append([train_X, train_y, train_s, train_y_original])
            batched_data.append(batched_data_round)

        return batched_data
