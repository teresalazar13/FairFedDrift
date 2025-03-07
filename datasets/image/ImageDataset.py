import random
import numpy as np
from tensorflow.keras.utils import to_categorical
from datasets.Dataset import Dataset
from scipy.ndimage import rotate


class ImageDataset(Dataset):

    def __init__(self, name, input_shape, is_pt, is_binary_target, X, y):
        super().__init__(name, input_shape, is_pt, is_binary_target)
        self.X = X
        self.y = y

    def create_batched_data(self, varying_disc):
        drift_ids = self.drift_ids
        n_clients = self.n_clients
        n_timesteps = self.n_timesteps
        X_priv, y_priv = self.X, self.y
        #X_priv, y_priv = self.augment(self.X, self.y)
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
                X_unpriv_round_client = self.negate(X_priv_round_client)
                y_unpriv_round_client = y_priv_round_client.copy()
                if drift_id != 0:
                    if self.is_pt:  # if is CIFAR-100 (nothing to do with PyTorch)
                        for k in range(3):
                            y_unpriv_round_client[y_unpriv_round_client == drift_id + k] = 100
                            y_unpriv_round_client[y_unpriv_round_client == drift_id + k + 10] = 101
                            y_unpriv_round_client[y_unpriv_round_client == 100] = drift_id + k + 10
                            y_unpriv_round_client[y_unpriv_round_client == 101] = drift_id + k
                    else:
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

    def negate(self, X_priv_round_client):
        if not self.is_pt:  # MNIST and FashionMNIST
            return np.rot90(X_priv_round_client.copy() * -1, axes=(-2, -1))
        else:  # CIFAR-100 and CIFAR-10
            inverted = X_priv_round_client.copy()
            inverted[..., :3] = 255 - inverted[..., :3]
            return inverted

    def augment(self, X, Y):
        X_augmented = []
        Y_augmented = []

        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            angles = np.linspace(-20, 20, 3)
            for angle in angles:
                new_image = rotate(x, angle, reshape=False, mode='nearest')  # Rotate image
                X_augmented.append(new_image)
                Y_augmented.append(y)

        X_augmented = np.array(X_augmented)
        Y_augmented = np.array(Y_augmented)

        indices = np.arange(len(X_augmented))
        np.random.shuffle(indices)

        return X_augmented[indices], Y_augmented[indices]
