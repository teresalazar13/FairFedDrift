from main_online import get_arguments
from metrics_online.MetricFactory import get_metrics
from plot.plot import read_results, plot, plot_each_client

if __name__ == '__main__':
    algorithm, dataset, n_rounds, n_clients, n_drifts, varying_disc = get_arguments()
    folder = dataset.get_folder(algorithm.name, n_drifts, varying_disc)
    res_clients = []

    for i in range(n_clients):
        res_client = read_results(get_metrics(dataset.is_image), "{}/client_{}/results_online.csv".format(folder, i+1))
        res_clients.append(res_client)

    for metric in get_metrics(dataset.is_image):
        plot(res_clients, n_rounds, "{}/results_{}.png".format(folder, metric.name), metric.name)
        plot_each_client(res_clients, n_rounds, "{}/results_{}.png".format(folder, metric.name), metric.name)

