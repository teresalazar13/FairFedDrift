from matplotlib import pyplot as plt


def plot_loss():
    plt.figure(figsize=(5, 3.5))
    metric = r'$\ell$'
    filename = "./Loss.png"
    fedavg_1 = [0.232004105, 0.16022246480000002, 0.155830392, 0.1313788094, 0.14532913399999997, 0.14840156200000001, 0.133090378, 0.12486841000000001, 0.0937390061, 0.0704806768]
    plt.plot(range(0, len(fedavg_1)), fedavg_1, label=r'FedAvg ($\alpha$=0.1)', color="slateblue", linestyle='-')
    fedavg_9 = [0.232759832, 0.16546678600000003, 0.32253460599999995, 0.285315957, 0.307970396, 0.43402331400000005, 0.381531238, 0.35622206, 0.39812241950000005, 0.169374242]
    plt.plot(range(0, len(fedavg_9)), fedavg_9, label=r'FedAvg ($\alpha$=0.9)', color="dodgerblue", linestyle="--")
    plt.xticks(range(0, 10))
    plt.xlabel("timestep")
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
    fedavg_9 = [0.21156001, 0.131930678, 0.326965282, 0.28839296000000003, 0.31063570399999996, 0.44681889900000005, 0.39122278600000004, 0.36094981200000004, 0.420569249, 0.167220078]
    plt.plot(range(0, len(fedavg_9)), fedavg_9, label=r'FedAvg ($\alpha$=0.9)', color="dodgerblue", linestyle="--")
    plt.xticks(range(0, 10))
    plt.xlabel("timestep")
    plt.ylim([0, 1])
    plt.ylabel(metric)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    plot_loss()
    plot_loss_s0()
