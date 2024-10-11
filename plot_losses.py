import pandas as pd
import matplotlib.pyplot as plt


from read import avg, std


def read(main_path):
    overall_losses = [[] for _ in range(9)]  # 9 is number of timesteps
    unprivileged_losses = [[] for _ in range(9)]  # 9 is number of timesteps
    for client_id in range(1, 11):
        client_path = "{}/client_{}/results.csv".format(main_path, client_id)
        df = pd.read_csv(client_path)
        previous_drift = 0
        for timestep in range(len(df["drift-id"])):
            current_drift = df["drift-id"][timestep]
            if previous_drift == current_drift:
                overall_losses[timestep].append(df["Loss"][timestep])
                unprivileged_losses[timestep].append(df["LossUnprivileged"][timestep])
            previous_drift = current_drift
    
    avgs_overall_losses = [avg(l) for l in overall_losses]
    stds_overall_losses = [std(l) for l in overall_losses]
    avgs_unprivileged_losses = [avg(l) for l in overall_losses]
    stds_unprivileged_losses = [std(l) for l in overall_losses]
    
    return avgs_overall_losses, stds_overall_losses, avgs_unprivileged_losses, stds_unprivileged_losses


def plot_helper(avg_values_s, std_values_s, avg_values_b, std_values_b, ylabel, title, label_s, label_b):
    x = range(1, 10)
    plt.plot(x, avg_values_s, marker='o', color='b', label=label_s, linestyle="-")
    plt.fill_between(x, [v - s for v, s in zip(avg_values_s, std_values_s)],
                     [v + s for v, s in zip(avg_values_s, std_values_s)], color='blue', alpha=0.2)
    plt.plot(x, avg_values_b, marker='o', color='r', label=label_b, linestyle="--")
    plt.fill_between(x, [v - s for v, s in zip(avg_values_b, std_values_b)],
                     [v + s for v, s in zip(avg_values_b, std_values_b)], color='red', alpha=0.2)
    plt.xlabel(r'$t$')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot(
        avgs_overall_losses_s, stds_overall_losses_s, avgs_unprivileged_losses_s, stds_unprivileged_losses_s,
        avgs_overall_losses_b, stds_overall_losses_b, avgs_unprivileged_losses_b, stds_unprivileged_losses_b
):
    # Plot overall losses
    plot_helper(
        avgs_overall_losses_s, stds_overall_losses_s,
        avgs_overall_losses_b, stds_overall_losses_b,
        r'$\ell$',
        r'Overall loss ($\ell$) values across distinct $\alpha$ values',
        r'FedAvg ($\alpha$=0.1)',
        r'FedAvg ($\alpha$=0.5)'
    )

    # Plot unprivileged group-specific losses
    plot_helper(
        avgs_unprivileged_losses_s, stds_unprivileged_losses_s,
        avgs_unprivileged_losses_b, stds_unprivileged_losses_b,
        r'$\ell_{S=0}$',
        r'Group-specific loss ($\ell_{S=0}$) values across distinct $\alpha$ values',
        r'FedAvg ($\alpha$=0.1)',
        r'FedAvg ($\alpha$=0.5)'
    )


if __name__ == '__main__':
    main_path_1 = "./results/scenario-1/MNIST-GDrift/disc_0.1/FedAvg"
    main_path_2 = "./results/scenario-1/MNIST-GDrift/disc_0.5/FedAvg"
    avgs_overall_losses_s, stds_overall_losses_s, avgs_unprivileged_losses_s, stds_unprivileged_losses_s = read(main_path_1)
    avgs_overall_losses_b, stds_overall_losses_b, avgs_unprivileged_losses_b, stds_unprivileged_losses_b = read(main_path_2)
    plot(
        avgs_overall_losses_s, stds_overall_losses_s, avgs_unprivileged_losses_s, stds_unprivileged_losses_s,
        avgs_overall_losses_b, stds_overall_losses_b, avgs_unprivileged_losses_b, stds_unprivileged_losses_b
    )

    
    
