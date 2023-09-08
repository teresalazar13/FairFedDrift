import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from datasets.Dataset import Dataset


class ImageDataset(Dataset):

    def __init__(self, name, input_shape, X, y):
        is_image = True
        super().__init__(name, is_image, input_shape)
        self.X = X
        self.y = y

    def create_batched_data(self, varying_disc):
        drift_ids = self.drift_ids
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        X_priv = self.X
        y_priv = self.y

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
                X_unpriv_round_client = self.rotate(X_priv_round_client)
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

    def rotate(self, X_priv_round_client):
        if self.name != "fairCIFAR":
            return np.rot90(X_priv_round_client.copy() * -1, axes=(-2, -1))
        else:
            X_train_rotated_negated = np.empty_like(X_priv_round_client)
            for i in range(len(X_priv_round_client)):
                for channel in range(3):  # 3 channels for RGB
                    rotated_channel = np.rot90(X_priv_round_client[i, :, :, channel], k=1)  # Rotate 90 degrees
                    X_train_rotated_negated[i, :, :, channel] = rotated_channel * -1

            return X_train_rotated_negated

