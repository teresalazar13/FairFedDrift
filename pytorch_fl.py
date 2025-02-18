import time
import logging
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import sys
import copy


from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from federated.algorithms.Algorithm import get_y
from federated.algorithms.Identity import Identity
from metrics.MetricFactory import get_metrics

class NNPT:
    def __init__(self):
        self.batch_size = 32
        self.n_epochs = 5
        self.model = NNPT_()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def compile(self):
        pass

    def learn(self, x_, y_):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        x_tensor = torch.tensor(x_, dtype=torch.float32).unsqueeze(1)  # Add channel dim  # TODO - check if for cifar100 this works
        y_tensor = torch.tensor(np.argmax(y_, axis=-1), dtype=torch.long)  # Convert from one-hot to class indices. Ensure y is Long type for CrossEntropyLoss
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            for batch, (X, y) in enumerate(dataloader):
                self.model.zero_grad()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
    def predict(self, x):
        self.model.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # Add channel dim  # TODO - check if for cifar100 this works

        return self.model(x_tensor).detach().numpy()


class NNPT_(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = input_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calculate the flattened size after conv + pool
        # Input: (1, 28, 28) → Conv(3x3) → (32, 26, 26) → Pool(2x2) → (32, 13, 13)
        #After convolution: (32, 26, 26) (since 28 - 3 + 1 = 26)
        #After pooling: (32, 13, 13) (dividing by 2)
        #Flattened size: 32 × 13 × 13 = 5408
        self.fc1 = nn.Linear(32 * 13 * 13, 100)  # Fully connected layer
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))  # Convolution + ReLU
        x = self.pool(x)  # Max pooling
        x = torch.flatten(x, start_dim=1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))  # Fully connected layer + ReLU
        x = torch.nn.functional.softmax(self.fc2(x), dim=1)  # Output layer with softmax
        return x


class NNTF:
    def __init__(self):
        self.batch_size = 32
        self.n_epochs = 5
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.model.add(tf.keras.layers.Dense(10, activation='softmax'))

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return copy.deepcopy(self.model.get_weights())

    def compile(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        loss = 'categorical_crossentropy'
        self.model.compile(loss=loss, optimizer=optimizer)

    def learn(self, x, y):
        tf.keras.utils.disable_interactive_logging()
        self.model.fit(x=x, y=y, batch_size=self.batch_size, epochs=self.n_epochs, verbose=0)

    def predict(self, x):
        return self.model.predict(x)


def test(clients_data_timestep, clients_metrics, global_model):
    for client_data, client_metrics in zip(clients_data_timestep, clients_metrics):
        x, y_true_raw, s, _ = client_data
        y_pred_raw = global_model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, False)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            logging.info("{}-{}".format(res, client_metric.name))


def perform_fl(clients_data):
    if is_tf:
        global_model = NNTF()
    else:
        global_model = NNPT()
    clients_metrics = [get_metrics(False) for _ in range(n_clients)]
    # Train with data from first timestep
    global_model = train_and_average(global_model, clients_data, 0)

    for timestep in range(1, n_timesteps):
        test(clients_data[timestep], clients_metrics, global_model)
        global_model = train_and_average(global_model, clients_data, timestep)

    # Clients identities are always 0 (only one global model)
    clients_identities = [[] for _ in range(n_clients)]
    for i in range(n_clients):
        for _ in range(1, n_timesteps):
            clients_identities[i].append(Identity(0, 0))

    return clients_metrics, clients_identities


def average_weights(weights_list, scaling_factors):
    if is_tf:
        scaled_local_weights_list = []
        global_count = sum(scaling_factors)

        for local_weights, local_count in zip(weights_list, scaling_factors):
            scale = local_count / global_count
            scaled_local_weights = []
            for i in range(len(local_weights)):
                scaled_local_weights.append(scale * local_weights[i])

            scaled_local_weights_list.append(scaled_local_weights)

        global_weights = []
        for grad_list_tuple in zip(*scaled_local_weights_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            global_weights.append(layer_mean)

    else:  # TODO - check if it weighted average
        global_weights = copy.deepcopy(weights_list[0])
        for key in global_weights.keys():
            for i in range(1, len(weights_list)):
                global_weights[key] += weights_list[i][key]
            global_weights[key] = torch.div(global_weights[key], len(weights_list))

    return global_weights


def train_and_average(global_model, clients_data, timestep):
    for cround in range(n_rounds):
        local_weights_list = []
        client_scaling_factors_list = []
        for client in range(n_clients):
            x, y, s, _ = clients_data[timestep][client]
            global_weights = global_model.get_weights()
            if is_tf:
                local_model = NNTF()
            else:
                local_model = NNPT()
            local_model.compile()
            local_model.set_weights(global_weights)
            local_model.learn(x, y)
            local_weights_list.append(local_model.get_weights())
            client_scaling_factors_list.append(len(x))
            # K.clear_session()
            logging.info("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

        new_global_weights = average_weights(local_weights_list, client_scaling_factors_list)
        global_model.set_weights(new_global_weights)
        logging.info("Averaged models on timestep {} cround {}".format(timestep, cround))

    return global_model

logging.basicConfig(
    level=logging.DEBUG
)
logging.getLogger().addHandler(logging.StreamHandler())
if sys.argv[1] == "tf":
    is_tf = True
else:
    is_tf = False

if is_tf:
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    input_shape = (28, 28, 1)
    X_priv = np.concatenate([train_X, test_X], axis=0)
    y_priv = np.concatenate([train_y, test_y], axis=0)
else:
    train_dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_dataset = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    X_priv = torch.cat([train_dataset.data, test_dataset.data], dim=0).float()
    y_priv = torch.cat([train_dataset.targets, test_dataset.targets], dim=0)
    input_shape = (1, 28, 28)  # PyTorch uses (C, H, W) format

varying_disc = 0.1
n_clients = 10
n_timesteps = 10
n_rounds = 10
drift_ids = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 1, 1, 1, 1, 1],
    [2, 2, 2, 0, 0, 0, 0, 0, 1, 1],
    [2, 2, 2, 0, 0, 0, 0, 0, 1, 1]
]


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
        if is_tf:
            X_unpriv_round_client = np.rot90(X_priv_round_client.copy() * -1, axes=(-2, -1))
            y_unpriv_round_client = y_priv_round_client.copy()
        else:
            X_unpriv_round_client = np.rot90(X_priv_round_client.clone() * -1, axes=(-2, -1))
            y_unpriv_round_client = y_priv_round_client.clone()
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

print(len(batched_data))
print(len(batched_data[0]))
print(len(batched_data[0][0]))
print(len(batched_data[0][0][0]))
print(batched_data[0][0][0].shape)

perform_fl(batched_data)
