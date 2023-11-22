from datasets.DatasetFactory import get_dataset_by_name
from metrics.MetricFactory import get_metrics
from plot.plot import read_results
import argparse
import sys
import json
import statistics


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', required=True, help='scenario')
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--varying_disc', required=True, help='varying_disc')

    args = parser.parse_args(sys.argv[1:])
    scenarios = [int(a) for a in args.scenarios]
    datasets = [get_dataset_by_name(args.dataset) for _ in range(len(scenarios))]
    for d, scenario in zip(datasets, scenarios):
        d.set_drifts(scenario)
    varying_disc = float(args.varying_disc)

    return scenarios, datasets, varying_disc


def avg_results(all_results_dict, scenario, res_clients_list, algs, metric):
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
            all_results_dict[alg_main] = {scenario: {alg: {metric: average}}}
        elif scenario not in all_results_dict[alg_main]:
            all_results_dict[alg_main][scenario] = {alg: {metric: average}}
        elif alg not in all_results_dict[alg_main][scenario]:
            all_results_dict[alg_main][scenario][alg] = {metric:average}
        else:
            all_results_dict[alg_main][scenario][alg][metric] = average

    return all_results_dict


def get_best_results_dict(all_results_dict):
    best_results_dict = {}
    for alg, scenarios_dict in all_results_dict.items():
        for scenario, algs_results_dict in scenarios_dict.items():
            best_bacc = 0
            best_results = None
            for alg_spec, results_dict in algs_results_dict.items():
                bacc = results_dict["BalancedACC"]
                if bacc > best_bacc:
                    best_bacc = bacc
                    best_results = results_dict
            print(alg, scenario, best_results)
            if alg not in best_results_dict:
                best_results_dict[alg] = {scenario: best_results}
            best_results_dict[alg][scenario] = best_results


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
        print(alg, avg_dict)
        for metric, res_list in avg_dict.items():
            print("{} - {}: {:.2f}+-{:.2f}".format(alg, metric, sum(res_list)/len(res_list), statistics.stdev(res_list)))

if __name__ == '__main__':
    scenarios, datasets, varying_disc = get_arguments()
    all_results_dict = {}

    for dataset, scenario in zip(datasets, scenarios):
        print("\n\nScenario ", scenario)
        main_folder, all_folders, algs = dataset.get_all_folders(scenario, varying_disc)

        res_clients_list = []
        for folder_ in all_folders:
            res_clients = []
            for i in range(dataset.n_clients):
                res_client = read_results(get_metrics(dataset.is_binary_target), "{}/client_{}/results.csv".format(folder_, i+1))
                res_clients.append(res_client)
            res_clients_list.append(res_clients)

        for metric in get_metrics(dataset.is_binary_target):
            all_results_dict = avg_results(all_results_dict, scenario, res_clients_list, algs, metric.name)

    print(json.dumps((all_results_dict), sort_keys=True, indent=4))

    best_results_dict = get_best_results_dict(all_results_dict)
    print(json.dumps((best_results_dict), sort_keys=True, indent=4))

    print_average_results(best_results_dict)
