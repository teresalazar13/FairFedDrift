import matplotlib.pyplot as plt
import pandas as pd
import statistics


def save_results(metrics, drift_ids, clients_identities, filename):
    df = pd.DataFrame()
    df = df.astype('object')
    for metric in metrics:
        df[metric.name] = metric.res
    df["drift-id"] = drift_ids
    df["gm-id"] = clients_identities

    df.to_csv(filename, index=False)


def read_results(metrics, filename):
    df = pd.read_csv(filename)
    res = {}
    for metric in metrics:
        res[metric.name] = df[metric.name]
    res["drift-id"] = df["drift-id"]
    res["gm-id"] = df["gm-id"]

    return res


def plot_algorithms(res_clients_list, algs, filename, metric, title):
    fig = plt.figure()

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
        print("{} - {}: {:.2f}+-{:.2f}".format(alg, metric, sum(avg[1:])/len(avg[1:]), statistics.stdev(avg[1:])))
        plt.plot(range(1, len(res_clients[0][metric].values)), avg[1:], label=alg.split(";")[0])
    plt.title(title)
    plt.xticks(range(1, 11))
    plt.xlabel("time")
    plt.ylabel(metric)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
    fig.subplots_adjust(bottom=0.25)
    plt.savefig(filename)
    plt.close()
