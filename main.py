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
    parser.add_argument('--varying_disc', required=True, help='varying_disc')

    parser.add_argument('--clustering', required=False, help='clustering')
    parser.add_argument('--metrics', nargs='+', required=False, help='metrics')
    parser.add_argument('--drift_detector', required=False, help='drift_detector')
    parser.add_argument('--thresholds', nargs='+', required=False, help='thresholds')

    args = parser.parse_args(sys.argv[1:])
    algorithm = get_algorithm_by_name(args.fl)
    dataset = get_dataset_by_name(args.dataset)
    if args.clustering and args.metrics and args.drift_detector:  # FairFedDrift
        algorithm.set_specs(args)
    varying_disc = float(args.varying_disc)

    return algorithm, dataset, varying_disc


def generate_directories(dataset, algorithm_subfolders, varying_disc):
    folder = dataset.get_folder(algorithm_subfolders, varying_disc)
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)
    for i in range(dataset.n_clients):
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
    algorithm, dataset, varying_disc = get_arguments()
    generate_directories(dataset, algorithm.subfolders, varying_disc)
    clients_data = dataset.create_batched_data(algorithm.subfolders, varying_disc)
    clients_metrics, client_gm_ids_col = algorithm.perform_fl(seed, clients_data, dataset)

    for i in range(len(clients_metrics)):
        save_results(
            clients_metrics[i], dataset.drift_ids_col[i], client_gm_ids_col[i],
            "{}/client_{}/results.csv".format(dataset.get_folder(algorithm.subfolders, varying_disc), i+1)
        )
