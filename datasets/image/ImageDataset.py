import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from datasets.Dataset import Dataset


class ImageDataset(Dataset):

    def __init__(self, name, input_shape, is_pt, n_classes, X, y):
        super().__init__(name, input_shape, is_pt, n_classes)
        self.X = X
        self.y = y

    def create_batched_data(self, varying_disc):
        if self.is_pt:
            return self.create_batched_data_pt(varying_disc)
        else:
            return self.create_batched_data_tf(varying_disc)

    def create_batched_data_tf(self, varying_disc):
        drift_ids = self.drift_ids
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        X_priv, y_priv = self.X, self.y
        batched_data = []
        X_priv_rounds = np.array_split(X_priv, n_timesteps)
        y_priv_rounds = np.array_split(y_priv, n_timesteps)

        for i in range(n_timesteps):
            batched_data_round = []
            X_priv_round_clients = np.array_split(X_priv_rounds[i], n_clients)
            y_priv_round_clients = np.array_split(y_priv_rounds[i], n_clients)
            for j in range(n_clients):
                drift_id = drift_ids[i][j]
                X_priv_round_client = X_priv_round_clients[j]
                y_priv_round_client = y_priv_round_clients[j]
                size_unpriv = round(len(X_priv_round_client) * varying_disc)
                X_unpriv_round_client = np.rot90(X_priv_round_client.copy() * -1, axes=(-2, -1))
                y_unpriv_round_client = y_priv_round_client.copy()
                if drift_id != 0:
                    y_unpriv_round_client[y_unpriv_round_client == drift_id] = 100
                    y_unpriv_round_client[y_unpriv_round_client == drift_id + 1] = 101
                    y_unpriv_round_client[y_unpriv_round_client == 100] = drift_id + 1
                    y_unpriv_round_client[y_unpriv_round_client == 101] = drift_id
                X = np.concatenate((X_priv_round_client[size_unpriv:], X_unpriv_round_client[:size_unpriv]), axis=0)
                y_original = np.concatenate((y_priv_round_client[size_unpriv:], y_unpriv_round_client[:size_unpriv]), axis=0)
                s = [1] * (len(X_priv_round_client) - size_unpriv) + [0] * size_unpriv
                X = X.astype('float32') / 255.0
                y = to_categorical(y_original)
                perm = list(range(0, len(X)))
                random.shuffle(perm)
                X = X[perm]
                y = y[perm]
                s = np.array(s)[perm]
                batched_data_round.append([X, y, s, y_original])
            batched_data.append(batched_data_round)

        return batched_data

    def create_batched_data_pt(self, varying_disc):
        drift_ids = self.drift_ids
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        X_priv, y_priv = self.get_classes([5, 6, 7, 8, 9], None)
        X_unpriv, y_unpriv = self.get_classes([0, 1, 2, 3, 4], round(len(X_priv) * varying_disc))
        X_priv_rounds = np.array_split(X_priv, n_timesteps)
        y_priv_rounds = np.array_split(y_priv, n_timesteps)
        X_unpriv_rounds = np.array_split(X_unpriv, n_timesteps)
        y_unpriv_rounds = np.array_split(y_unpriv, n_timesteps)
        batched_data = []

        for i in range(n_timesteps):
            batched_data_round = []
            X_priv_round_clients = np.array_split(X_priv_rounds[i], n_clients)
            y_priv_round_clients = np.array_split(y_priv_rounds[i], n_clients)
            X_unpriv_round_clients = np.array_split(X_unpriv_rounds[i], n_clients)
            y_unpriv_round_clients = np.array_split(y_unpriv_rounds[i], n_clients)
            for j in range(n_clients):
                drift_id = drift_ids[i][j]
                X_priv_round_client = X_priv_round_clients[j]
                y_priv_round_client = y_priv_round_clients[j]
                X_unpriv_round_client = X_unpriv_round_clients[j]
                y_unpriv_round_client = y_unpriv_round_clients[j]
                if drift_id != 0:
                    class_a = drift_id - 1
                    class_b = drift_id
                    y_unpriv_round_client[y_unpriv_round_client == class_a] = 100
                    y_unpriv_round_client[y_unpriv_round_client == class_b] = 101
                    y_unpriv_round_client[y_unpriv_round_client == 100] = class_b
                    y_unpriv_round_client[y_unpriv_round_client == 101] = class_a
                X = np.concatenate((X_priv_round_client, X_unpriv_round_client), axis=0)
                y_original = np.concatenate((y_priv_round_client, y_unpriv_round_client), axis=0)
                s = [1] * len(X_priv_round_client) + [0] * len(X_unpriv_round_client)
                X = X.astype('float32') / 255.0
                y = to_categorical(y_original)
                perm = list(range(0, len(X)))
                random.shuffle(perm)
                X = X[perm]
                y = y[perm]
                s = np.array(s)[perm]
                batched_data_round.append([X, y, s, y_original])
            batched_data.append(batched_data_round)

        return batched_data

    def get_classes(self, classes, size):
        mask = np.isin(self.y, classes)
        mask = mask[:, np.newaxis]
        if size is None:
            X_filtered = self.X[mask.squeeze()]
            y_filtered = self.y[mask.squeeze()]
        else:
            X_filtered = self.X[mask.squeeze()][:size]
            y_filtered = self.y[mask.squeeze()][:size]

        return X_filtered, y_filtered
