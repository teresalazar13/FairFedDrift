from main import get_arguments
from metrics.MetricFactory import get_metrics
from plot.plot import read_results, plot, plot_each_client, plot_avg, plot_algorithms

if __name__ == '__main__':
    algorithm, dataset, varying_disc = get_arguments()
    folder = dataset.get_folder(algorithm.subfolders, varying_disc)
    res_clients = []

    for i in range(dataset.n_clients):
        res_client = read_results(
            get_metrics(dataset.is_image),
            "{}/client_{}/results.csv".format(folder, i+1)
        )
        res_clients.append(res_client)

    for metric in get_metrics(dataset.is_image):
        plot(res_clients, "{}/results_{}.png".format(folder, metric.name), metric.name)
        plot_avg(res_clients, "{}/results_avg_{}.png".format(folder, metric.name), metric.name)
        plot_each_client(res_clients, "{}/results_{}.png".format(folder, metric.name), metric.name)

    main_folder, all_folders, algs = dataset.get_all_folders(varying_disc)
    res_clients_list = []
    print(all_folders)
    for folder_ in all_folders:
        res_clients = []
        for i in range(dataset.n_clients):
            print(all_folders)
            print(folder_)
            res_client = read_results(get_metrics(dataset.is_image), "{}/client_{}/results.csv".format(folder_, i+1))
            res_clients.append(res_client)
        res_clients_list.append(res_clients)
    for metric in get_metrics(dataset.is_image):
        plot_algorithms(res_clients_list, algs, "{}/results_{}.png".format(main_folder, metric.name), metric.name)
