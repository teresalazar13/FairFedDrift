import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import distinctipy


def plot_synthetic_data(drift_data, n_drifts, n_samples, filename):
    fig, axs = plt.subplots(nrows=n_drifts, ncols=1, figsize=(10, 5 * n_drifts))

    for i in range(len(drift_data)):
        X, y, s, disc, right, up = drift_data[i]
        if n_drifts > 1:
            plot_synthetic_data_drift(X, y, s, disc, right, up, axs[i], n_samples[i])
        else:
            plot_synthetic_data_drift(X, y, s, disc, right, up, axs, n_samples[i])
    plt.savefig(filename)


def plot_synthetic_data_drift(X, y, s, disc, right, up, ax, n_samples):
    perm = list(range(0, n_samples))
    random.Random(10).shuffle(perm)  # shuffle data to plot under specific seed
    num_to_draw = 200  # we will only draw a small number of points to avoid clutter
    x_draw = X[perm][:num_to_draw]
    y_draw = y[perm][:num_to_draw]
    s_draw = s[perm][:num_to_draw]

    X_s_0 = x_draw[s_draw == 0.0]
    X_s_1 = x_draw[s_draw == 1.0]
    y_s_0 = y_draw[s_draw == 0.0]
    y_s_1 = y_draw[s_draw == 1.0]
    ax.scatter(
        X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5,
        label="Prot. +ve"
    )
    ax.scatter(
        X_s_0[y_s_0 == 0][:, 0], X_s_0[y_s_0 == 0][:, 1], color='red', marker='x', s=30, linewidth=1.5,
        label="Prot. -ve"
    )
    ax.scatter(
        X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], color='green', marker='o', facecolors='none', s=30,
        label="Non-prot. +ve"
    )
    ax.scatter(
        X_s_1[y_s_1 == 0][:, 0], X_s_1[y_s_1 == 0][:, 1], color='red', marker='o', facecolors='none', s=30,
        label="Non-prot. -ve"
    )
    ax.tick_params(
        axis='x', which='both', bottom='off', top='off', labelbottom='off'
    )  # dont need the ticks to see the data distribution
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
    ax.set_title("up: {} | right: {} | disc: {}".format(up, right, disc))
    ax.legend(fontsize=15, loc="lower right")
    ax.set_xlim((-15, 15))
    ax.set_ylim((-15, 15))


def save_results(metrics, drift_ids, gm_ids_col, filename):
    df = pd.DataFrame()
    df = df.astype('object')
    for metric in metrics:
        df[metric.name] = metric.res
    df["drift-id"] = drift_ids
    df["gm-id"] = gm_ids_col

    df.to_csv(filename, index=False)


def read_results(metrics, filename):
    df = pd.read_csv(filename)
    res = {}
    for metric in metrics:
        res[metric.name] = df[metric.name]
    res["drift-id"] = df["drift-id"]
    res["gm-id"] = df["gm-id"]

    return res


def plot(res_clients, filename, metric):
    plt.figure()
    markers = [".", "+", "*", 1, 2]
    edgecolors = distinctipy.get_colors(20)

    for i in range(len(res_clients)):
        unique_drifts_ids = res_clients[i]["drift-id"].unique()
        for drift_id in unique_drifts_ids:
            unique_gm_ids = res_clients[i]["gm-id"].unique()
            for gm_id in unique_gm_ids:
                indexes = np.where(
                    (res_clients[i]["drift-id"].values == drift_id) & (res_clients[i]["gm-id"].values == gm_id)
                )[0]
                plt.scatter(
                    indexes, res_clients[i][metric].values[indexes],
                    marker=markers[drift_id], color=edgecolors[i],
                    label="client-{} drift-{} gm={}".format(i + 1, drift_id, gm_id)
                )

    plt.ylim([0, 1])
    plt.xlabel("time")
    plt.ylabel(metric)
    plt.savefig(filename)
    plt.close()


def plot_avg(res_clients, filename, metric):
    plt.figure()
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

    plt.scatter(range(0, len(res_clients[0][metric].values)), avg)
    plt.ylim([0, 1])
    plt.xlabel("time")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_algorithms(res_clients_list, algs, filename, metric):
    plt.figure()
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
                    print(previous_drift_id, current_drift_id)
                    if current_drift_id == previous_drift_id:
                        values.append(res_clients[j][metric].values[i])
            avg.append(sum(values) / len(values))
        plt.scatter(range(0, len(res_clients[0][metric].values)), avg, label=alg)
    plt.ylim([0, 1])
    plt.xlabel("time")
    plt.ylabel(metric)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


def plot_each_client(res_clients, filename, metric):
    markers = [".", "+", "*", 1, 2]
    colors = distinctipy.get_colors(20)

    for i in range(len(res_clients)):
        plt.figure()
        unique_drifts_ids = res_clients[i]["drift-id"].unique()
        for drift_id in unique_drifts_ids:
            unique_gm_ids = res_clients[i]["gm-id"].unique()
            for j, gm_id in enumerate(unique_gm_ids):
                indexes = np.where(
                    (res_clients[i]["drift-id"].values == drift_id) & (res_clients[i]["gm-id"].values == gm_id)
                )[0]
                plt.scatter(
                    indexes, res_clients[i][metric].values[indexes],
                    label="drift-{} gm={}".format(drift_id, gm_id), marker=markers[drift_id], color=colors[j]
                )

        plt.xlabel("time")
        plt.ylabel(metric)
        plt.legend()
        plt.ylim([0, 1])
        print(filename.replace(".png", "_client_{}.png".format(i)))
        plt.savefig(filename.replace(".png", "_client_{}.png".format(i)))
        plt.close()
