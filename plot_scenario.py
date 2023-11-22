from matplotlib import pyplot as plt

from datasets.DatasetFactory import get_dataset_by_name
from federated.algorithms.AlgorithmFactory import get_algorithm_by_name
from metrics.MetricFactory import get_metrics
from plot.plot import read_results
import argparse
import sys
import json


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True, help='scenario')
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--varying_disc', required=True, help='varying_disc')

    args = parser.parse_args(sys.argv[1:])
    scenario = int(args.scenario)
    dataset = get_dataset_by_name(args.dataset)
    dataset.set_drifts(scenario)
    varying_disc = float(args.varying_disc)

    return scenario, dataset, varying_disc


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
        print("{} - {}: {:.2f}".format(alg, metric, average))

        alg_main = alg.split(";")[1]
        if alg_main not in all_results_dict:
            all_results_dict[alg_main] = {alg: {metric: [average, avg]}}
        elif alg not in all_results_dict[alg_main]:
            all_results_dict[alg_main][alg] = {metric: [average, avg]}
        else:
            all_results_dict[alg_main][alg][metric] = [average, avg]

    return all_results_dict


def get_best_results_dict(all_results_dict):
    best_results_dict = {}
    for alg, algs_dict in all_results_dict.items():
        best_bacc = 0
        best_results = None
        for alg_spec, results_dict in algs_dict.items():
            bacc = results_dict["BalancedACC"][0]
            if bacc > best_bacc:
                best_bacc = bacc
                best_results = results_dict
        print(alg)
        print(json.dumps((best_results), sort_keys=True, indent=4))
        best_results_dict[alg] = best_results

    return best_results_dict


def plot_algorithms(best_results_dict, filename, metric, title):
    plt.figure(figsize=(5, 3.5))
    for alg, res_clients in best_results_dict.items():
        if alg == "FedAvg":
            plt.plot(
                range(0, len(res_clients[metric][1])), res_clients[metric][1], label=alg,
                color=get_algorithm_by_name(alg).color
            )
    plt.title(title)
    plt.xticks(range(0, 10))
    plt.xlabel("timestep")
    plt.ylim([0, 1])
    plt.ylabel(metric)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    scenario, dataset, varying_disc = get_arguments()
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

    #print(json.dumps((all_results_dict), sort_keys=True, indent=4))
    #print("\n\n\nBEST Results")
    best_results_dict = get_best_results_dict(all_results_dict)

    for metric in get_metrics(dataset.is_binary_target):
        title = r'{} ($\alpha$={})'.format(dataset.name, str(varying_disc))
        filename = "{}/results_{}-{}-{}.png".format(main_folder, dataset.name, str(varying_disc), metric.name)
        plot_algorithms(best_results_dict, filename, metric.name, title)
