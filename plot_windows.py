import argparse
import sys
import matplotlib.pyplot as plt

from plot_deltas import plot
from read import get_results, avg, std

# python3 plot_windows.py --scenario 1 --dataset MNIST-GDrift --alpha 0.1 --algorithm FairFedDrift --windows 1 2 3 4 5 6 7 8 9 --deltas 0.01 0.05 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0


def plot_dataset_alpha(scenarios, dataset, alpha, algorithm, windows_arg, deltas, axs):
    windows = []
    accs_avg = []
    accs_std = []
    aeqs_avg = []
    aeqs_std = []
    oeqs_avg = []
    oeqs_std = []
    opps_avg = []
    opps_std = []
    for window in windows_arg:
        res = get_results(scenarios, dataset, alpha, algorithm, window, deltas)
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
            windows.append(window)
            print("Window {}".format(window))
            print("AEQ - {:.2f}+-{:.2f}".format(avg(aeqs), std(aeqs)))
            print("OEQ - {:.2f}+-{:.2f}".format(avg(oeqs), std(oeqs)))
            print("OPP - {:.2f}+-{:.2f}".format(avg(opps), std(opps)))
            print("ACC - {:.2f}+-{:.2f}".format(avg(accs), std(accs)))

    title = r'{}: $\alpha={}$'.format(dataset.replace("Fashion", "FE"), alpha)
    plot(
        title, windows, "window", aeqs_avg, aeqs_std, oeqs_avg, oeqs_std, opps_avg, opps_std, accs_avg, accs_std, axs
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True, help='scenario')
    parser.add_argument('--dataset', required=True, help='dataset')
    parser.add_argument('--alpha', required=True, help='alpha')
    parser.add_argument('--algorithm', required=True, help='algorithm')
    parser.add_argument('--windows', required=True, nargs='+', help='window')
    parser.add_argument('--deltas', required=True, nargs='+', help='deltas')
    args = parser.parse_args(sys.argv[1:])
    scenario = args.scenario
    dataset = args.dataset
    alpha = args.alpha
    algorithm = args.algorithm
    windows_args = args.windows
    deltas = args.deltas

    fig, axs = plt.subplots(1)
    plot_dataset_alpha([scenario], dataset, alpha, algorithm, windows_args, deltas, axs)
    fig.set_figheight(4)
    fig.set_figwidth(6)
    axs.set_title(r'Effect of window on Fairness and Performance')
    plt.show()
