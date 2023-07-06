import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import distinctipy


def plot_synthetic_data(drift_data, n_drifts, n_samples, filename):
    fig, axs = plt.subplots(nrows=n_drifts, ncols=1, figsize=(10, 25))

    for i in range(len(drift_data)):
        X, y, s, disc, right, up = drift_data[i]
        plot_synthetic_data_drift(X, y, s, disc, right, up, axs[i], n_samples[i])

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
    ax.legend(fontsize=15)
    ax.set_xlim((-15, 15))
    ax.set_ylim((-15, 15))


def save_results(metrics, drift_ids, gm_ids_col, filename):
    df = pd.DataFrame()
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


def plot(res_clients, n_rounds, filename, metric):
    plt.figure()
    markers = [".", "+", "*", 1, 2]
    edgecolors = distinctipy.get_colors(20, rng=5)

    for i in range(len(res_clients)):
        unique_drifts_ids = res_clients[i]["drift-id"].unique()
        for drift_id in unique_drifts_ids:
            unique_gm_ids = res_clients[i]["gm-id"].unique()
            for gm_id in unique_gm_ids:
                indexes = np.where(
                    (res_clients[i]["drift-id"].values == drift_id) & (res_clients[i]["gm-id"].values == gm_id)
                )[0]
                indexes = [indexes[i] for i in range(0, len(indexes), 200)]  # only show one out of 1000
                indexes = [i for i in indexes if i > 100]  # skip first 100
                plt.scatter(
                    indexes, res_clients[i][metric].values[indexes],
                    marker=markers[drift_id], color=edgecolors[i],
                    #color=colors[drift_id], marker=markers[drift_id],
                    label="client-{} drift-{} gm={}".format(i + 1, drift_id, gm_id)
                )

    #round_size = len(res_clients[0][metric]) // n_rounds
    #for i in range(len(res_clients[0][metric]) // round_size):
        #plt.axvline(x=round_size*i, color="grey")

    plt.ylim([0, 1])
    plt.xlabel("time")
    plt.ylabel(metric)
    #plt.legend()
    plt.savefig(filename)
    plt.close()


def plot_each_client(res_clients, n_rounds, filename, metric):
    markers = [".", "+", "*", 1, 2]
    colors = distinctipy.get_colors(20, rng=5)

    for i in range(len(res_clients)):
        plt.figure()
        unique_drifts_ids = res_clients[i]["drift-id"].unique()
        for drift_id in unique_drifts_ids:
            unique_gm_ids = res_clients[i]["gm-id"].unique()
            for gm_id in unique_gm_ids:
                indexes = np.where(
                    (res_clients[i]["drift-id"].values == drift_id) & (res_clients[i]["gm-id"].values == gm_id)
                )[0]
                indexes = [indexes[i] for i in range(0, len(indexes), 200)]  # only show one out of 1000
                indexes = [i for i in indexes if i > 100]  # skip first 100
                plt.scatter(
                    indexes, res_clients[i][metric].values[indexes],
                    label="drift-{} gm={}".format(drift_id, gm_id), marker=markers[drift_id], color=colors[gm_id]
                )
        #round_size = len(res_clients[0][metric]) // n_rounds
        #for j in range(len(res_clients[0][metric]) // round_size):
            #plt.axvline(x=round_size * j)
        #plt.ylim([0, disc_1])
        plt.xlabel("time")
        plt.ylabel(metric)
        plt.legend()
        plt.ylim([0, 1])
        print(filename.replace(".png", "_client_{}.png".format(i)))
        plt.savefig(filename.replace(".png", "_client_{}.png".format(i)))
        plt.close()
