from matplotlib import pyplot as plt


def plot_loss():
    plt.figure(figsize=(5, 3.5))
    metric = r'$\ell$'
    filename = "./Loss.png"
    fedavg_1 = [0.232004105, 0.16022246480000002, 0.155830392, 0.1313788094, 0.14532913399999997, 0.14840156200000001, 0.133090378, 0.12486841000000001, 0.0937390061, 0.0704806768]
    plt.plot(range(0, len(fedavg_1)), fedavg_1, label=r'FedAvg ($\alpha$=0.1)', color="slateblue", linestyle='-')
    fedavg_8 = [0.247859275, 0.17188574999999998, 0.338034466, 0.28943438, 0.27914743, 0.39950378, 0.370329521, 0.347829872, 0.42559499900000003, 0.17999327199999998]
    plt.plot(range(0, len(fedavg_8)), fedavg_8, label=r'FedAvg ($\alpha$=0.8)', color="dodgerblue", linestyle="--")
    plt.xticks(range(0, 10))
    plt.xlabel(r'$t$')
    plt.ylim([0, 1])
    plt.ylabel(metric)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_loss_s0():
    plt.figure(figsize=(5, 3.5))
    metric = r'$\ell_{S=0}$'
    filename = "./Loss_S0.png"
    fedavg_1 = [0.46530501399999996, 0.422957306, 0.6321275319999999, 0.435066442, 0.5410406240000001, 0.693914735, 0.6035494349999999, 0.541623856, 0.536577139, 0.37849725999999995]
    plt.plot(range(0, len(fedavg_1)), fedavg_1, label=r'FedAvg ($\alpha$=0.1)', color="slateblue", linestyle="-")
    fedavg_8 = [0.23772795300000005, 0.14169542799999998, 0.357989731, 0.3051637, 0.284505344, 0.4485552790000001, 0.41374011, 0.38984839399999993, 0.501485168, 0.190891296]
    plt.plot(range(0, len(fedavg_8)), fedavg_8, label=r'FedAvg ($\alpha$=0.8)', color="dodgerblue", linestyle="--")
    plt.xticks(range(0, 10))
    plt.xlabel(r'$t$')
    plt.ylim([0, 1])
    plt.ylabel(metric)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    plot_loss()
    plot_loss_s0()
