import pandas as pd

from datasets.DatasetFactory import get_dataset_by_name
from federated.algorithms.AlgorithmFactory import get_algorithm_by_name
import shutil
import os
import argparse
import sys
import numpy as np
import random
import tensorflow as tf
import logging


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True, help='scenario')
    parser.add_argument('--fl', required=True, help='algorithm')
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--varying_disc', required=True, help='varying_disc')
    parser.add_argument('--thresholds', nargs='+', required=False, help='thresholds')
    parser.add_argument('--window', required=False, help='window')

    args = parser.parse_args(sys.argv[1:])
    scenario = int(args.scenario)
    algorithm = get_algorithm_by_name(args.fl)
    dataset = get_dataset_by_name(args.dataset)
    dataset.set_drifts(scenario)
    if args.thresholds:
        algorithm.set_specs(args)
    varying_disc = float(args.varying_disc)

    # TODO - remove
    parser.add_argument('--net', required=True, help='net')
    parser.add_argument('--bs', required=True, help='batch_size')
    parser.add_argument('--lr', required=True, help='learning_rate')
    parser.add_argument('--ne', required=True, help='n_epochs')
    parser.add_argument('--nr', required=True, help='n_rounds')
    dataset.set_args(args.net, int(args.bs), float(args.lr), int(args.ne), int(args.nr))

    return scenario, algorithm, dataset, varying_disc


def generate_directories(scenario, dataset, algorithm_subfolders, varying_disc):
    folder = dataset.get_folder(scenario, algorithm_subfolders, varying_disc)
    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder, exist_ok=True)
    for i in range(dataset.n_clients):
        os.mkdir("{}/client_{}".format(folder, i+1))


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    #tf.keras.utils.set_random_seed(seed)
    #tf.compat.v1.set_random_seed(seed)
    #session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #tf.compat.v1.keras.backend.set_session(sess)


def save_results(metrics, drift_ids, clients_identities, filename):
    df = pd.DataFrame()
    df = df.astype('object')
    for metric in metrics:
        df[metric.name] = metric.res
    df["drift-id"] = drift_ids[1:]
    df["gm-id"] = [ci.id for ci in clients_identities]
    df["gm-name"] = [ci.name for ci in clients_identities]

    df.to_csv(filename, index=False)


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    scenario, algorithm, dataset, varying_disc = get_arguments()
    seed = scenario
    set_seeds(seed)
    generate_directories(scenario, dataset, algorithm.subfolders, varying_disc)
    logging.basicConfig(
        filename="{}/output.txt".format(dataset.get_folder(scenario, algorithm.subfolders, varying_disc)),
        level=logging.DEBUG
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    clients_data = dataset.create_batched_data(varying_disc)
    clients_metrics, clients_identities = algorithm.perform_fl(seed, clients_data, dataset)

    for i in range(len(clients_metrics)):
        save_results(
            clients_metrics[i], dataset.drift_ids_col[i], clients_identities[i],
            "{}/client_{}/results.csv".format(dataset.get_folder(scenario, algorithm.subfolders, varying_disc), i+1)
        )
