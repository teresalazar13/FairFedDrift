import argparse
import sys
import os
import statistics
import pandas as pd


# TODO - delta = threshold
# TODO - alpha = varying_disc

# python3 read.py --scenarios 1 --dataset MNIST-GDrift --alpha 0.1 --algorithm FedAvg
# python3 read.py --scenarios 1 2 3 4 5 --dataset MNIST-GDrift --alpha 0.1 --algorithm Oracle
# python3 read.py --scenarios 1 --dataset MNIST-GDrift --alpha 0.1 --algorithm FedDrift --window inf --delta 0.1
# python3 read.py --scenarios 1 --dataset MNIST-GDrift --alpha 0.1 --algorithm FairFedDrift --window inf --deltas 0.01 0.05 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0


def get_client_path(main_path, client_id, algorithm, window, delta):
    if algorithm in ["FedAvg", "Oracle"]:
        client_path = "{}/client_{}/results.csv".format(main_path, client_id)
    elif algorithm == "FedDrift":
        client_path = "{}/window-{}/loss-{}/client_{}/results.csv".format(
            main_path, window, delta, client_id
        )
    elif algorithm == "FairFedDrift":
        client_path = "{}/window-{}/loss_p-{}/loss_up-{}/client_{}/results.csv".format(
            main_path, window, delta, delta, client_id
        )
    else:
        raise Exception("No algorithm with the name: {}", algorithm)

    return client_path


def avg(l):
    return sum(l)/len(l)


def std(l):
    return statistics.stdev(l)


def get_clients_results_scenario(main_path, algorithm, window, deltas):
    if algorithm in ["FedAvg", "Oracle"]:
        deltas = [None]

    best_accs = []
    best_aeqs = []
    best_oeqs = []
    best_opps = []
    best_f1 = []
    best_f1eq = []
    best_delta = None

    for delta in deltas:
        accs = []
        aeqs = []
        oeqs = []
        opps = []
        f1 = []
        f1eq = []
        for client_id in range(1, 11):
            client_path = get_client_path(main_path, client_id, algorithm, window, delta)
            if os.path.exists(client_path):
                df = pd.read_csv(client_path)
                previous_drift = 0
                for timestep in range(len(df["drift-id"])):
                    current_drift = df["drift-id"][timestep]
                    if previous_drift == current_drift:
                        accs.append(df["ACC"][timestep])
                        aeqs.append(df["AEQ"][timestep])
                        oeqs.append(df["OEQ"][timestep])
                        opps.append(df["OPP"][timestep])
                        oeqs.append(df["OEQ"][timestep])
                        f1.append(df["F1Score"][timestep])
                        f1eq.append(df["F1 Score Equality"][timestep])
                    previous_drift = current_drift
        if len(accs) == 0:
            print("No results for delta {}".format(delta))
        elif len(best_accs) == 0 or (avg(accs) + avg(aeqs) + avg(oeqs) + avg(opps)) > (avg(best_accs) + avg(best_aeqs) + avg(best_oeqs) + avg(best_opps)):
            best_accs = accs
            best_aeqs = aeqs
            best_oeqs = oeqs
            best_opps = opps
            best_f1 = f1
            best_f1eq = f1eq
            best_delta = delta

    return best_accs, best_aeqs, best_oeqs, best_opps, best_f1, best_f1eq, best_delta


def get_results(scenarios, dataset, alpha, algorithm, window, deltas):
    accs = []
    aeqs = []
    oeqs = []
    opps = []
    f1s = []
    f1eqs = []
    for scenario in scenarios:
        main_path = "./results/scenario-{}/{}/disc_{}/{}".format(scenario, dataset, alpha, algorithm)
        accs_s, aeqs_s, oeqs_s, opps_s, f1s_s, f1eqs_s, delta_s = get_clients_results_scenario(main_path, algorithm, window, deltas)
        if len(accs_s) == 0:
            print("No results for scenario {} deltas {}".format(scenario, deltas))
            return None
        accs.extend(accs_s)
        aeqs.extend(aeqs_s)
        oeqs.extend(oeqs_s)
        opps.extend(opps_s)
        f1s.extend(f1s_s)
        f1eqs.extend(f1eqs_s)

    return accs, aeqs, oeqs, opps, f1s, f1eqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', required=True, nargs='+', help='scenarios')
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--alpha', required=True, help='alpha')
    parser.add_argument('--algorithm', required=True, help='algorithm')
    parser.add_argument('--window', required=False, help='window')
    parser.add_argument('--deltas', required=False, nargs='+', help='deltas')
    args = parser.parse_args(sys.argv[1:])
    scenarios = args.scenarios
    dataset = args.dataset
    alpha = args.alpha
    algorithm = args.algorithm
    window = args.window
    deltas = args.deltas

    res = get_results(scenarios, dataset, alpha, algorithm, window, deltas)
    if res is not None:
        accs, aeqs, oeqs, opps, f1s, f1eqs = res
        print("AEQ - {:.2f}+-{:.2f}".format(avg(aeqs), std(aeqs)))
        print("OEQ - {:.2f}+-{:.2f}".format(avg(oeqs), std(oeqs)))
        print("OPP - {:.2f}+-{:.2f}".format(avg(opps), std(opps)))
        print("ACC - {:.2f}+-{:.2f}".format(avg(accs), std(accs)))
        print("F1 - {:.2f}+-{:.2f}".format(avg(f1s), std(f1s)))
        print("F1Eq - {:.2f}+-{:.2f}".format(avg(f1s), std(f1s)))
