import matplotlib.pyplot as plt


def plot_main(delta_values, acc_values, acc_stds, aeq_values, aeq_stds, oeq_values, oeq_stds, opp_values, opp_stds):
    plt.figure()
    plot_each(delta_values, acc_values, acc_stds, "ACC", color='b')
    plot_each(delta_values, aeq_values, aeq_stds, "AEQ", color='g')
    plot_each(delta_values, oeq_values, oeq_stds, "OEQ", color='r')
    plot_each(delta_values, opp_values, opp_stds, "OPP", color='m')

    plt.xticks([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    plt.xlabel('Delta')
    plt.ylabel('Metric Values')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_each(delta_values, values, stds, name, color):
    filtered_delta_values = [d for d, v in zip(delta_values, values) if v is not None]
    filtered_values = [v for v in values if v is not None]
    filtered_stds = [s for v, s in zip(values, stds) if v is not None]
    plt.plot(filtered_delta_values, filtered_values, marker='o', color=color, label=name, linestyle="-")
    plt.fill_between(filtered_delta_values,
                     [v - s for v, s in zip(filtered_values, filtered_stds)],
                     [v + s for v, s in zip(filtered_values, filtered_stds)],
                     color=color, alpha=0.2)


if __name__ == '__main__':
    delta_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    # FairFedDrift - MNIST 0.05 scenario 1
    acc_values = [None, None, None, 0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95]
    acc_stds = [None, None, None, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    aeq_values = [None, None, None, 0.77, 0.81, 0.84, 0.85, 0.87, 0.87, 0.83, 0.83]
    aeq_stds = [None, None, None, 0.08, 0.06, 0.06, 0.05, 0.05, 0.05, 0.06, 0.05]
    oeq_values = [None, None, None, 0.71, 0.74, 0.77, 0.78, 0.80, 0.80, 0.79, 0.78]
    oeq_stds = [None, None, None, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06]
    opp_values = [None, None, None, 0.70, 0.75, 0.77, 0.78, 0.80, 0.80, 0.79, 0.79]
    opp_stds = [None, None, None, 0.09, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06]
    plot_main(delta_values, acc_values, acc_stds, aeq_values, aeq_stds, oeq_values, oeq_stds, opp_values, opp_stds)

    # FairFedDrift - MNIST 0.1 scenario 1
    acc_values = [None, None, 0.93, 0.94, 0.93, 0.93, 0.94, 0.94, 0.94, 0.94, 0.94]
    acc_stds = [None, None, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    aeq_values = [None, None, 0.90, 0.92, 0.90, 0.91, 0.92, 0.91, 0.92, 0.92, 0.92]
    aeq_stds = [None, None, 0.04, 0.04, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
    oeq_values = [None, None, 0.86, 0.88, 0.85, 0.87, 0.87, 0.87, 0.88, 0.88, 0.87]
    oeq_stds = [None, None, 0.04, 0.05, 0.04, 0.04, 0.05, 0.04, 0.05, 0.04, 0.05]
    opp_values = [None, None, 0.86, 0.88, 0.86, 0.87, 0.88, 0.87, 0.88, 0.88, 0.88]
    opp_stds = [None, None, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
