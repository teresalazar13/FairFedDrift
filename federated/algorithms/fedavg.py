from federated.algorithms.Algorithm import Algorithm, average_weights, test, sum_weights, divide_weights, scale_weights
from federated.algorithms.Identity import Identity
from federated.model import NN_model
from federated.modelPT import NNModel
from metrics.MetricFactory import get_metrics
import logging
import time
import gc
import tensorflow as tf
import psutil


class FedAvg(Algorithm):

    def __init__(self):
        name = "FedAvg"
        color = "blue"
        marker = "+"
        super().__init__(name, color, marker)

    def perform_fl(self, seed, clients_data, dataset):
        global_model = NNModel(dataset, seed)
        clients_metrics = [get_metrics(dataset.is_binary_target) for _ in range(dataset.n_clients)]
        # Train with data from first timestep
        global_model = train_and_average(global_model, dataset, clients_data, 0, seed)

        for timestep in range(1, dataset.n_timesteps):
            test(clients_data[timestep], clients_metrics, global_model, dataset)
            global_model = train_and_average(global_model, dataset, clients_data, timestep, seed)

        # Clients identities are always 0 (only one global model)
        clients_identities = [[] for _ in range(dataset.n_clients)]
        for i in range(dataset.n_clients):
            for _ in range(1, dataset.n_timesteps):
                clients_identities[i].append(Identity(0, 0))

        return clients_metrics, clients_identities


def train_and_average(global_model, dataset, clients_data, timestep, seed):
    for cround in range(dataset.n_rounds):
        start = time.time()
        total_count = 0
        global_weights_summed = None

        for client in range(dataset.n_clients):
            print_memory_usage("Before Training")
            print_gpu_memory()

            x, y, s, _ = clients_data[timestep][client]
            local_model = NNModel(dataset, seed)
            local_model.compile()
            local_model.set_weights(global_model.get_weights())
            local_model.learn(x, y)
            local_count = len(x)
            local_weights = local_model.get_weights()
            if global_weights_summed is None:
                global_weights_summed = scale_weights(local_weights, local_count)
            else:
                global_weights_summed = sum_weights(global_weights_summed, local_weights, local_count)
            total_count += local_count
            logging.info("Trained model timestep {} cround {} client {}".format(timestep, cround, client))

        new_global_weights = divide_weights(global_weights_summed, total_count)
        global_model.set_weights(new_global_weights)
        logging.info("Averaged models on timestep {} cround {}".format(timestep, cround))
        end = time.time()
        logging.info(end - start)

    return global_model


def print_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("[INFO] No GPU detected.")
        return
    for gpu in gpus:
        try:
            device_name = gpu.name.replace("physical_device:", "")  # Extract device name
            details = tf.config.experimental.get_memory_info(device_name)  # Pass as string
            print(f"GPU Memory (Used): {details['current'] / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Could not retrieve GPU memory info: {e}")


def print_memory_usage(label):
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    logging.info(f"[{label}] Memory Usage: {mem:.2f} MB")
