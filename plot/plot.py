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
        print("{} - {}: {:.2f}+-{:.2f}".format(alg, metric, sum(avg)/len(avg), statistics.stdev(avg)))
        if "ignore" not in alg:
            plt.plot(range(0, len(res_clients[0][metric].values)), avg, label=alg.split(";")[0])
    plt.title(title)
    plt.xticks(range(0, 10))
    plt.xlabel("time")
    plt.ylim([0, 1])
    plt.ylabel(metric)
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.close()

def save_clients_identities(clients_identities_string, folder):
    if clients_identities_string:
        f = open("{}/clients_identities.txt".format(folder), "w+")
        f.write(clients_identities_string)
        f.close()
