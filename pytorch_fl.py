import logging
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import sys
import copy
import time
from keras.datasets import fashion_mnist, cifar100
from tensorflow.keras.utils import to_categorical
from federated.algorithms.Algorithm import get_y
from federated.algorithms.Identity import Identity
from metrics.MetricFactory import get_metrics
import torchvision.transforms as transforms



class NNPT:
    def __init__(self, is_large):
        if not is_large:
            self.batch_size = 32
            self.n_epochs = 5
            self.model = NNPTSmall()
            self.model = self.model.to('cuda')
        else:
            self.batch_size = 64
            self.n_epochs = 15
            self.model = NNPTLarge()
            self.model = self.model.to('cuda')

        print("hey", next(self.model.parameters()).device)


    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()

    def compile(self):
        pass

    def learn(self, x_, y_):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        if not is_large:
            x_tensor = torch.tensor(x_, dtype=torch.float32).unsqueeze(1)  # Add channel dim
        else:
            x_tensor = torch.tensor(x_, dtype=torch.float32)
            x_tensor = x_tensor.permute(0, 3, 1, 2)
        x_tensor = x_tensor.to('cuda')
        print("oi", x_tensor.device)
        print("hey", next(self.model.parameters()).device)
        y_tensor = torch.tensor(np.argmax(y_, axis=-1), dtype=torch.long)  # Convert from one-hot to class indices. Ensure y is Long type for CrossEntropyLoss
        y_tensor = y_tensor.to('cuda')
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.n_epochs):
            for batch, (X, y) in enumerate(dataloader):
                self.model.zero_grad()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                if not is_large:
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
                else:
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
                optimizer.step()

    def predict(self, x):
        self.model.eval()
        if not is_large:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # Add channel dim
            x_tensor = x_tensor.to('cuda')
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
            x_tensor = x_tensor.permute(0, 3, 1, 2)
            x_tensor = x_tensor.to('cuda')
        return self.model(x_tensor).detach().numpy()


class NNPTSmall(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = input_shape[0]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 13 * 13, 100)  # Fully connected layer
        self.fc2 = nn.Linear(100, 10)
        # Calculate the flattened size after conv + pool
        # Input: (1, 28, 28) → Conv(3x3) → (32, 26, 26) → Pool(2x2) → (32, 13, 13)
        #After convolution: (32, 26, 26) (since 28 - 3 + 1 = 26)
        #After pooling: (32, 13, 13) (dividing by 2)
        #Flattened size: 32 × 13 × 13 = 5408

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))  # Convolution + ReLU
        x = self.pool(x)  # Max pooling
        x = torch.flatten(x, start_dim=1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))  # Fully connected layer + ReLU
        x = self.fc2(x)  # Output layer with softmax
        return x


class NNPTLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = transforms.Resize((224, 224))
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for layer in self.resnet50.children():
            if isinstance(layer, nn.BatchNorm2d):
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False
        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.resnet50.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(256),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.resize(x)  # Apply resizing transformation
        x = self.resnet50(x)
        return x


def test(clients_data_timestep, clients_metrics, global_model):
    for client_data, client_metrics in zip(clients_data_timestep, clients_metrics):
        x, y_true_raw, s, _ = client_data
        y_pred_raw = global_model.predict(x)
        y_true, y_pred = get_y(y_true_raw, y_pred_raw, False)
        for client_metric in client_metrics:
            res = client_metric.update(y_true, y_pred, y_true_raw, y_pred_raw, s)
            logging.info("{}-{}".format(res, client_metric.name))


def perform_fl(clients_data, is_large):
    global_model = NNPT(is_large)
    clients_metrics = [get_metrics(False) for _ in range(n_clients)]
    # Train with data from first timestep
    global_model = train_and_average(global_model, clients_data, 0, is_large)

    for timestep in range(1, n_timesteps):
        test(clients_data[timestep], clients_metrics, global_model)
        global_model = train_and_average(global_model, clients_data, timestep, is_large)

    # Clients identities are always 0 (only one global model)
    clients_identities = [[] for _ in range(n_clients)]
    for i in range(n_clients):
        for _ in range(1, n_timesteps):
            clients_identities[i].append(Identity(0, 0))

    return clients_metrics, clients_identities


def average_weights(weights_list, scaling_factors):
    global_weights = copy.deepcopy(weights_list[0])
    global_count = sum(scaling_factors)
    for key in global_weights.keys():
        global_weights[key] *= scaling_factors[0]  # Scale first model's weights
        for i in range(1, len(weights_list)):
            global_weights[key] += weights_list[i][key] * scaling_factors[i]  # Scale other models' weights
        global_weights[key] = torch.div(global_weights[key], global_count)  # Normalize by total samples

    return global_weights


def train_and_average(global_model, clients_data, timestep, is_large):
    for cround in range(n_rounds):
        local_weights_list = []
        client_scaling_factors_list = []
        for client in range(n_clients):
            start = time.time()
            x, y, s, _ = clients_data[timestep][client]
            global_weights = global_model.get_weights()
            local_model = NNPT(is_large)
            local_model.compile()
            local_model.set_weights(global_weights)
            local_model.learn(x, y)
            local_weights_list.append(local_model.get_weights())
            client_scaling_factors_list.append(len(x))
            # K.clear_session()
            logging.info("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

            end = time.time()
            print(end-start)

        new_global_weights = average_weights(local_weights_list, client_scaling_factors_list)
        global_model.set_weights(new_global_weights)
        logging.info("Averaged models on timestep {} cround {}".format(timestep, cround))

    return global_model


def negate(X_priv_round_client):
    dim = input_shape[0]
    if dim == 1:
        return np.rot90(X_priv_round_client.copy().astype(np.int16) * -1, axes=(-2, -1))
    elif dim == 3:
        return np.rot90(X_priv_round_client.copy().astype(np.int16) * -1, axes=(-3, -2))
    else:
        raise Exception("Can't rotate for shape ", input_shape)


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    logging.basicConfig(
        level=logging.DEBUG
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    if sys.argv[1] == "large":
        is_large = True
    else:
        is_large = False

    if is_large:
        (train_X, train_y), (test_X, test_y) = cifar100.load_data()
        input_shape = (3, 224, 224)  # PyTorch uses (C, H, W) format
    else:
        (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
        input_shape = (1, 28, 28)  # PyTorch uses (C, H, W) format
    X_priv = np.concatenate([train_X, test_X], axis=0)
    y_priv = np.concatenate([train_y, test_y], axis=0)

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
            X_unpriv_round_client = negate(X_priv_round_client)
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

    print(len(batched_data))
    print(len(batched_data[0]))
    print(len(batched_data[0][0]))
    print(len(batched_data[0][0][0]))
    print(batched_data[0][0][0].shape)

    perform_fl(batched_data, is_large)
