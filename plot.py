from datasets.DatasetFactory import get_dataset_by_name
from metrics.MetricFactory import get_metrics
from plot.plot import read_results, plot_algorithms
import argparse
import sys


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--varying_disc', required=True, help='varying_disc')

    args = parser.parse_args(sys.argv[1:])
    dataset = get_dataset_by_name(args.dataset)
    varying_disc = float(args.varying_disc)

    return dataset, varying_disc


if __name__ == '__main__':
    dataset, varying_disc = get_arguments()
    main_folder, all_folders, algs = dataset.get_all_folders(varying_disc)

    res_clients_list = []
    for folder_ in all_folders:
        res_clients = []
        for i in range(dataset.n_clients):
            res_client = read_results(get_metrics(dataset.is_image), "{}/client_{}/results.csv".format(folder_, i+1))
            res_clients.append(res_client)
        res_clients_list.append(res_clients)

    for metric in get_metrics(dataset.is_image):
        title = "{}-{}".format(dataset.name, str(varying_disc))
        plot_algorithms(
            res_clients_list, algs, "{}/results_{}-{}.png".format(main_folder, title, metric.name), metric.name, title
        )
