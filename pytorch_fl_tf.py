import time
import logging
import random
import numpy as np

import sys
import copy
import time

from keras.datasets import fashion_mnist, cifar100
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from federated.algorithms.Algorithm import get_y
from federated.algorithms.Identity import Identity
from metrics.Accuracy import Accuracy
from metrics.MetricFactory import get_metrics




class NNTF:
    def __init__(self, is_large):
        self.dataset = is_large
        if not is_large:
            self.batch_size = 32
            self.n_epochs = 5
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            self.model.add(tf.keras.layers.Flatten())
            self.model.add(tf.keras.layers.Dense(100, activation='relu'))
            self.model.add(tf.keras.layers.Dense(10, activation='softmax'))
        else:
            self.batch_size = 64
            self.n_epochs = 15
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Resizing(224, 224, interpolation='bilinear'))
            resnet_model50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            for layer in resnet_model50.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
                else:
                    layer.trainable = False
            self.model.add(resnet_model50)
            self.model.add(tf.keras.layers.GlobalAveragePooling2D())
            self.model.add(tf.keras.layers.Dense(256, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(.25))
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Dense(100, activation='softmax'))
            dummy_input = tf.random.normal([1] + list(input_shape))
            dummy_labels = tf.zeros([1, 100])
            self.compile()
            self.model.fit(dummy_input, dummy_labels, epochs=1, batch_size=1, verbose=0)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return copy.deepcopy(self.model.get_weights())

    def compile(self):
        if self.dataset == "small":
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
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


def perform_fl(clients_data, is_large):
    global_model = NNTF(is_large)
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

    return global_weights


def train_and_average(global_model, clients_data, timestep, is_large):
    for cround in range(n_rounds):
        local_weights_list = []
        client_scaling_factors_list = []
        for client in range(n_clients):
            start = time.time()
            x, y, s, _ = clients_data[timestep][client]
            global_weights = global_model.get_weights()
            local_model = NNTF(is_large)
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
    dim = input_shape[2]
    if dim == 1:
        return np.rot90(X_priv_round_client.copy().astype(np.int16) * -1, axes=(-2, -1))
    elif dim == 3:
        return np.rot90(X_priv_round_client.copy().astype(np.int16) * -1, axes=(-3, -2))
    else:
        raise Exception("Can't rotate for shape ", input_shape)


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
    input_shape = (32, 32, 3)
else:
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    input_shape = (28, 28, 1)

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
