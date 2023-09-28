from matplotlib import pyplot as plt

from datasets.DatasetFactory import get_dataset_by_name
from metrics.MetricFactory import get_metrics
import argparse
import sys

from plot.plot import read_results


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--varying_discs', nargs='+', required=True, help='varying_discs array')

    args = parser.parse_args(sys.argv[1:])
    dataset = get_dataset_by_name(args.dataset)
    varying_discs = [float(f) for f in args.varying_discs]

    return dataset, varying_discs


def get_mean_value(res_clients, metric):
    avg = []
    for round in range(1, len(res_clients[0][metric].values)):
        values = []
        for client in range(len(res_clients)):
            previous_drift_id = res_clients[client]["drift-id"][round - 1]
            current_drift_id = res_clients[client]["drift-id"][round]
            if current_drift_id == previous_drift_id:
                values.append(res_clients[client][metric].values[round])
        avg.append(sum(values) / len(values))

    return sum(avg) / len(avg)  # average of all rounds


def plot_all(title, filename, varying_discs, metric, values_fedavg, values_oracle):
    fig = plt.figure()

    plt.scatter(varying_discs, values_fedavg, label="fedavg", color="blue", s=20)
    plt.scatter(varying_discs, values_oracle, label="oracle", color="red", s=20)
    plt.title(title)
    plt.xlabel("$\\alpha$")
    plt.ylabel(metric)
    plt.xticks(varying_discs, varying_discs)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(filename)
    plt.close()


def get_metrics_values(dataset, varying_discs, alg):
    acc_values = []
    balanced_acc_values = []

    for varying_disc in varying_discs:
        folder = "./results/{}/disc_{}/{}".format(dataset.name, varying_disc, alg)
        res_clients = []
        for i in range(dataset.n_clients):
            res_client = read_results(get_metrics(dataset.is_image),"{}/client_{}/results.csv".format(folder, i + 1))
            res_clients.append(res_client)
        mean_value_acc = get_mean_value(res_clients, "ACC")
        mean_value_balanced_acc = get_mean_value(res_clients, "BalancedACC")
        acc_values.append(mean_value_acc)
        balanced_acc_values.append(mean_value_balanced_acc)

    return acc_values, balanced_acc_values


if __name__ == '__main__':
    dataset, varying_discs = get_arguments()
    acc_values_fedavg, balanced_acc_values_fedavg = get_metrics_values(dataset, varying_discs, "fedavg")
    acc_values_oracle, balanced_acc_values_oracle = get_metrics_values(dataset, varying_discs, "oracle")

    title = "{} - ACC results across varying $\\alpha$".format(dataset.name)
    filename = "./results/{}/results-ACC.png".format(dataset.name)
    plot_all(
        title, filename, varying_discs, "ACC",
        acc_values_fedavg, acc_values_oracle
    )

    title = "{} - BalancedACC results across varying $\\alpha$".format(dataset.name)
    filename = "./results/{}/results-BalancedACC.png".format(dataset.name)
    plot_all(
        title, filename, varying_discs, "BalancedACC",
        balanced_acc_values_fedavg, balanced_acc_values_oracle
    )
