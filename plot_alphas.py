from matplotlib import pyplot as plt

from datasets.DatasetFactory import get_dataset_by_name
from federated.algorithms.AlgorithmFactory import get_algorithm_by_name
from metrics.MetricFactory import get_metrics
from plot.plot import read_results
import argparse
import sys
import json
import statistics


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True, help='scenario')
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--varying_discs', nargs='+', required=True, help='varying_disc')

    args = parser.parse_args(sys.argv[1:])
    scenario = int(args.scenario)
    dataset = get_dataset_by_name(args.dataset)
    dataset.set_drifts(scenario)
    varying_discs = [float(a) for a in args.varying_discs]

    return scenario, dataset, varying_discs


def avg_results(all_results_dict, res_clients_list, algs, metric):
    for res_clients, alg in zip(res_clients_list, algs):
        avg = []
        for i in range(len(res_clients[0][metric].values)):
            values = []
            for j in range(len(res_clients)):
                if i == 0:
                    values.append(res_clients[j][metric].values[i])
                else:
                    previous_drift_id = res_clients[j]["drift-id"][i - 1]
                    current_drift_id = res_clients[j]["drift-id"][i]
                    if current_drift_id == previous_drift_id:
                        values.append(res_clients[j][metric].values[i])
            avg.append(sum(values) / len(values))
        average = sum(avg)/len(avg)
        logging.info("{} - {}: {:.2f}".format(alg, metric, average))

        alg_main = alg.split(";")[1]
        if alg_main not in all_results_dict:
            all_results_dict[alg_main] = {alg: {metric: average}}
        elif alg not in all_results_dict[alg_main]:
            all_results_dict[alg_main][alg] = {metric:average}
        else:
            all_results_dict[alg_main][alg][metric] = average

    return all_results_dict


def get_best_results_dict(all_results_dict):
    best_results_dict = {}
    for alg, algs_results_dict in all_results_dict.items():
        best_bacc = 0
        best_results = None
        for alg_spec, results_dict in algs_results_dict.items():
            bacc = results_dict["BalancedACC"]
            if bacc > best_bacc:
                best_bacc = bacc
                best_results = results_dict
        best_results_dict[alg] = best_results

    return best_results_dict


def print_average_results(best_results_dict):
    for alg, scenarios_results in best_results_dict.items():
        avg_dict = {}
        for scenario, results_dict in scenarios_results.items():
            for metric, res in results_dict.items():
                if metric not in avg_dict:
                    avg_dict[metric] = [res]
                else:
                    avg_dict[
                        metric].append(res)
        logging.info(alg, avg_dict)
        for metric, res_list in avg_dict.items():
            logging.info("{} - {}: {:.2f}+-{:.2f}".format(alg, metric, sum(res_list)/len(res_list), statistics.stdev(res_list)))


def plot_all(varying_discs, best_results_dict_list, dataset, scenario):
    results_dict = {}
    for varying_disc, algs_best_results_dict in zip(varying_discs, best_results_dict_list):
        for alg, alg_best_results_dict in algs_best_results_dict.items():
            for metric, res in alg_best_results_dict.items():
                if metric not in results_dict:
                    results_dict[metric] = {alg: [res]}
                elif alg not in results_dict[metric]:
                    results_dict[metric][alg] = [res]
                else:
                    results_dict[metric][alg].append(res)

    for metric, results in results_dict.items():
        plt.figure(figsize=(5,3.5))
        for alg, values in results.items():
            if len(values) == len(varying_discs):
                plt.scatter(
                    varying_discs, values, label=alg, s=20,
                    color=get_algorithm_by_name(alg).color, marker=get_algorithm_by_name(alg).marker
                )
        filename = "./results/scenario-{}/{}/results-{}-{}.png".format(scenario, dataset.name, dataset.name, metric)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.xlabel("$\\alpha$")
        metric = metric.replace("BalancedACC", "B-AAC")
        plt.ylabel(metric)
        plt.xticks(varying_discs, varying_discs)
        plt.legend(loc="lower right")
        plt.savefig(filename)
        plt.close()


if __name__ == '__main__':
    scenario, dataset, varying_discs = get_arguments()
    best_results_dict_list = []

    for varying_disc in varying_discs:
        all_results_dict = {}
        main_folder, all_folders, algs = dataset.get_all_folders(scenario, varying_disc)
        res_clients_list = []
        for folder_ in all_folders:
            res_clients = []
            for i in range(dataset.n_clients):
                res_client = read_results(get_metrics(dataset.is_binary_target), "{}/client_{}/results.csv".format(folder_, i+1))
                res_clients.append(res_client)
            res_clients_list.append(res_clients)

        for metric in get_metrics(dataset.is_binary_target):
            all_results_dict = avg_results(all_results_dict, res_clients_list, algs, metric.name)

        logging.info("\n\n\nVarying DISC", varying_disc)
        logging.info(json.dumps((all_results_dict), sort_keys=True, indent=4))
        best_results_dict = get_best_results_dict(all_results_dict)
        logging.info(json.dumps((best_results_dict), sort_keys=True, indent=4))
        best_results_dict_list.append(best_results_dict)

    for metric in get_metrics(dataset.is_binary_target):
        plot_all(varying_discs, best_results_dict_list, dataset, scenario)
