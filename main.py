from datasets.DatasetFactory import get_dataset_by_name
from federated.algorithms.AlgorithmFactory import get_algorithm_by_name
from plot.plot import save_results
import shutil
import os
import argparse
import sys
import numpy as np
import random
import tensorflow as tf


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fl', required=True, help='algorithm')
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--n_timesteps', required=True, help='n_timesteps')
    parser.add_argument('--n_rounds', required=True, help='n_rounds')
    parser.add_argument('--n_clients', required=True, help='n_clients')
    parser.add_argument('--n_drifts', required=True, help='n_drifts')
    parser.add_argument('--varying_disc', required=True, help='varying_disc')
    args = parser.parse_args(sys.argv[1:])

    return get_algorithm_by_name(args.fl), get_dataset_by_name(args.dataset), \
           int(args.n_timesteps), int(args.n_rounds), int(args.n_clients), int(args.n_drifts), float(args.varying_disc)


def generate_directories(dataset, alg, n_clients, n_drifts, varying_disc):
    folder = dataset.get_folder(alg, n_drifts, varying_disc)
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)
    for i in range(n_clients):
        os.mkdir("{}/client_{}".format(folder, i+1))


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == '__main__':
    seed = 10
    set_seeds(seed)
    algorithm, dataset, n_timesteps, n_rounds, n_clients, n_drifts, varying_disc = get_arguments()
    generate_directories(dataset, algorithm.name, n_clients, n_drifts, varying_disc)
    clients_data, drift_ids_col, n_features = dataset.create_batched_data(
        algorithm.name, n_drifts, varying_disc, n_clients, n_timesteps
    )
    clients_metrics, client_gm_ids_col = algorithm.perform_fl(
        n_timesteps, n_rounds, n_clients, n_features, clients_data, seed, dataset.is_image
    )

    for i in range(len(clients_metrics)):
        save_results(
            clients_metrics[i], drift_ids_col[i], client_gm_ids_col[i],
            "{}/client_{}/results.csv".format(dataset.get_folder(algorithm.name, n_drifts, varying_disc), i+1)
        )
