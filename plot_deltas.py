import argparse
import sys
import matplotlib.pyplot as plt
from read import get_results, avg, std

# python3 plot_deltas.py --scenarios 1 2 3 4 5 --dataset MNIST-GDrift --alpha 0.1 --algorithm FairFedDrift --window inf --deltas 0.01 0.05 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0
# python3 plot_deltas.py --scenarios 1 2 3 4 5 --algorithm FairFedDrift --window inf --deltas 0.01 0.05 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0

def plot(title, x, xlabel, aeqs_avg, aeqs_std, oeqs_avg, oeqs_std, opps_avg, opps_std, accs_avg, accs_std, axs):
    axs.plot(x, aeqs_avg, marker='o', color='b', label='AEQ', linestyle="-")
    axs.fill_between(x, [v - s for v, s in zip(aeqs_avg, aeqs_std)], [v + s for v, s in zip(aeqs_avg, aeqs_std)],
                     color='blue', alpha=0.2)
    axs.plot(x, oeqs_avg, marker='o', color='g', label='OEQ', linestyle="--")
    axs.fill_between(x, [v - s for v, s in zip(oeqs_avg, oeqs_std)], [v + s for v, s in zip(oeqs_avg, oeqs_std)],
                     color='green', alpha=0.2)
    axs.plot(x, opps_avg, marker='o', color='r', label='OPP', linestyle=":")
    axs.fill_between(x, [v - s for v, s in zip(opps_avg, opps_std)], [v + s for v, s in zip(opps_avg, opps_std)],
                     color='red', alpha=0.2)
    axs.plot(x, accs_avg, marker='o', color='c', label='ACC', linestyle="-.")
    axs.fill_between(x, [v - s for v, s in zip(accs_avg, accs_std)], [v + s for v, s in zip(accs_avg, accs_std)],
                     color='cyan', alpha=0.2)
    axs.set_xlabel(xlabel)
    axs.set_ylabel('metric values')
    axs.set_title(title)
    axs.legend(fontsize=8, loc="lower right")
    axs.grid(True)


def plot_dataset_alpha(scenarios, dataset, alpha, algorithm, window, deltas_args, axs):
    deltas = []
    accs_avg = []
    accs_std = []
    aeqs_avg = []
    aeqs_std = []
    oeqs_avg = []
    oeqs_std = []
    opps_avg = []
    opps_std = []
    for delta in deltas_args:
        res = get_results(scenarios, dataset, alpha, algorithm, window, [delta])
        if res is not None:
            accs, aeqs, oeqs, opps = res
            accs_avg.append(avg(accs))
            accs_std.append(std(accs))
            aeqs_avg.append(avg(aeqs))
            aeqs_std.append(std(aeqs))
            oeqs_avg.append(avg(oeqs))
            oeqs_std.append(std(oeqs))
            opps_avg.append(avg(opps))
            opps_std.append(std(opps))
            deltas.append(delta)
            print("Delta {}".format(delta))
            print("AEQ - {:.2f}+-{:.2f}".format(avg(aeqs), std(aeqs)))
            print("OEQ - {:.2f}+-{:.2f}".format(avg(oeqs), std(oeqs)))
            print("OPP - {:.2f}+-{:.2f}".format(avg(opps), std(opps)))
            print("ACC - {:.2f}+-{:.2f}".format(avg(accs), std(accs)))

    title = r'{}: $\alpha={}$'.format(dataset.replace("Fashion", "FE").replace("CIFAR10", "CIFAR-10"), alpha)
    plot(
        title, deltas, "$\delta_s$", aeqs_avg, aeqs_std, oeqs_avg, oeqs_std, opps_avg, opps_std, accs_avg, accs_std, axs
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', required=True, nargs='+', help='scenarios')
    parser.add_argument('--dataset', required=False, help='dataset')
    parser.add_argument('--alpha', required=False, help='alpha')
    parser.add_argument('--algorithm', required=True, help='algorithm')
    parser.add_argument('--window', required=False, help='window')
    parser.add_argument('--deltas', required=False, nargs='+', help='deltas')
    args = parser.parse_args(sys.argv[1:])
    scenarios = args.scenarios
    dataset = args.dataset
    alpha = args.alpha
    algorithm = args.algorithm
    window = args.window
    deltas_args = args.deltas

    if dataset and alpha:
        fig, axs = plt.subplots(1)
        plot_dataset_alpha(scenarios, dataset, alpha, algorithm, window, deltas_args, axs)
    else:
        fig, axs = plt.subplots(2, 4)
        for i, alpha in enumerate(["0.05", "0.1"]):
            for j, dataset in enumerate(["MNIST-GDrift", "FashionMNIST-GDrift", "Adult-GDrift"]):
                plot_dataset_alpha(scenarios, dataset, alpha, algorithm, window, deltas_args, axs[i, j])
        plot_dataset_alpha(scenarios, "CIFAR10-GDrift", "0.25", algorithm, window, deltas_args, axs[0, 3])
        plot_dataset_alpha(scenarios, "CIFAR10-GDrift", "0.5", algorithm, window, deltas_args, axs[1, 3])
    fig.set_figheight(7)
    fig.set_figwidth(16)
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    #fig.suptitle(r'Effect of $\delta_s$ on Fairness and Performance')
    plt.show()
