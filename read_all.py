from datasets.DatasetFactory import get_dataset_by_name
from metrics.MetricFactory import get_metrics

import math
import argparse
import sys
import json
import statistics
import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', required=True, help='scenario')
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--varying_disc', required=True, help='varying_disc')
    parser.add_argument('--window', required=False, help='window')

    args = parser.parse_args(sys.argv[1:])
    scenarios = [int(a) for a in args.scenarios]
    datasets = [get_dataset_by_name(args.dataset) for _ in range(len(scenarios))]
    for d, scenario in zip(datasets, scenarios):
        d.set_drifts(scenario)
    varying_disc = float(args.varying_disc)
    window = math.inf
    if args.window:
        window = args.window

    return scenarios, datasets, varying_disc, window


def avg_results(all_results_dict, scenario, window, res_clients_list, algs, metric):
    for res_clients, alg in zip(res_clients_list, algs):
        avg = []
        stds = []
        for timestep in range(len(res_clients[0][metric].values)):
            values = []
            for client_id in range(len(res_clients)):
                if timestep == 0:
                    values.append(res_clients[client_id][metric].values[timestep])
                else:
                    previous_drift_id = res_clients[client_id]["drift-id"][timestep - 1]
                    current_drift_id = res_clients[client_id]["drift-id"][timestep]
                    if current_drift_id == previous_drift_id:
                        values.append(res_clients[client_id][metric].values[timestep])
            avg.append(sum(values) / len(values))
            if len(values) > 1:
                stds.append(statistics.stdev(values))
            else:
                stds.append(0)
        average = sum(avg)/len(avg)
        print("{} - {}: {:.2f}".format(alg, metric, average))
        print("avg - {}".format(avg))
        print("stds - {}".format(stds))

        alg_main = None
        if "FedDrift" not in alg and "FairFedDrift" not in alg:
            alg_main = alg.split(";")[1]
        elif "window" in alg and alg.split(";")[2] == "window-{}".format(window):  # check if window matches
            alg_main = "{}-{}".format(alg.split(";")[1], alg.split(";")[2])

        if alg_main:
            if alg_main not in all_results_dict:
                all_results_dict[alg_main] = {scenario: {alg: {metric: [average, avg]}}}
            elif scenario not in all_results_dict[alg_main]:
                all_results_dict[alg_main][scenario] = {alg: {metric: [average, avg]}}
            elif alg not in all_results_dict[alg_main][scenario]:
                all_results_dict[alg_main][scenario][alg] = {metric: [average, avg]}
            else:
                all_results_dict[alg_main][scenario][alg][metric] = [average, avg]

    return all_results_dict


def get_best_results_dict(all_results_dict):
    best_results_dict = {}
    for alg, scenarios_dict in all_results_dict.items():
        for scenario, algs_results_dict in scenarios_dict.items():  # For each scenario, find best results
            best_value = 0
            best_results = None
            for alg_spec, results_dict in algs_results_dict.items():
                print(alg, alg_spec, results_dict)
                print("\n\n\n")
                aeq = results_dict["AEQ"][0]
                if aeq > best_value:
                    best_value = aeq
                    best_results = results_dict
            if alg not in best_results_dict:
                best_results_dict[alg] = {scenario: best_results}
            best_results_dict[alg][scenario] = best_results

    return best_results_dict


def print_results_dict(all_results_dict, n_scenarios):
    res = {}
    for alg, scenarios_dict in all_results_dict.items():
        for scenario, algs_results_dict in scenarios_dict.items():  # For each scenario, find best results
            for alg_spec, results_dict in algs_results_dict.items():
                if alg_spec not in res:
                    res[alg_spec] = {scenario: results_dict}
                else:
                    res[alg_spec][scenario] = results_dict

    print("\n\nALL RESULTS\n\n")
    print_average_results(res, n_scenarios)
    print("\n\nFINISH ALL RESULTS\n\n")


def print_average_results(best_results_dict, n_scenarios):
    for alg, scenarios_results in best_results_dict.items():
        avg_dict = {}
        for scenario, results_dict in scenarios_results.items():
            for metric, res in results_dict.items():
                if metric not in avg_dict:
                    avg_dict[metric] = [res]
                else:
                    avg_dict[metric].append(res)
        for metric, res_list in avg_dict.items():
            all_results = []
            for r in res_list:
                all_results.extend(r[1])
            if n_scenarios == len(res_list):
                print("{} - {}: {:.2f}+-{:.2f}".format(
                    alg, metric, sum(all_results)/len(all_results), statistics.stdev(all_results))
                )
            else:
                print("Not all scenarios for", alg)

def read_results(metrics, filename):
    df = pd.read_csv(filename)
    res = {}
    for metric in metrics:
        res[metric.name] = df[metric.name]
    res["drift-id"] = df["drift-id"]
    res["gm-id"] = df["gm-id"]

    return res


if __name__ == '__main__':
    scenarios, datasets, varying_disc, window = get_arguments()
    all_results_dict = {}

    for dataset, scenario in zip(datasets, scenarios):
        print("\n\nScenario {}".format(scenario))
        main_folder, all_folders, algs = dataset.get_all_folders(scenario, varying_disc)

        res_clients_list = []
        for folder_ in all_folders:
            res_clients = []
            for i in range(dataset.n_clients):
                res_client = read_results(get_metrics(dataset.is_binary_target), "{}/client_{}/results.csv".format(folder_, i+1))
                res_clients.append(res_client)
            res_clients_list.append(res_clients)

        for metric in get_metrics(dataset.is_binary_target):
            all_results_dict = avg_results(all_results_dict, scenario, window, res_clients_list, algs, metric.name)

    #print("all results dict")
    #print(json.dumps((all_results_dict), sort_keys=True, indent=4))

    best_results_dict = get_best_results_dict(all_results_dict)
    #print("best results dict")
    #print(json.dumps((best_results_dict), sort_keys=True, indent=4))

    print_results_dict(all_results_dict, len(scenarios))  # for delta plot

    print("Best Results:")
    print_average_results(best_results_dict, len(scenarios))
