import matplotlib.pyplot as plt

values_aeq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
stds_aeq = [0.1, 0.2, 0.1, 0.3, 0.15, 0.25, 0.1, 0.2, 0.15, 0.3]

values_oeq = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
stds_oeq = [0.15, 0.3, 0.2, 0.25, 0.1, 0.3, 0.2, 0.15, 0.25, 0.35]

values_opp = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
stds_opp = [0.2, 0.25, 0.15, 0.35, 0.2, 0.25, 0.15, 0.3, 0.2, 0.3]

values_acc = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
stds_acc = [0.25, 0.2, 0.3, 0.1, 0.25, 0.35, 0.3, 0.2, 0.25, 0.2]

x = range(1, 11)
plt.plot(x, values_aeq, marker='o', color='b', label='AEQ', linestyle="-")
plt.fill_between(x, [v - s for v, s in zip(values_aeq, stds_aeq)], [v + s for v, s in zip(values_aeq, stds_aeq)], color='blue', alpha=0.2)
plt.plot(x, values_oeq, marker='o', color='g', label='OEQ', linestyle="--")
plt.fill_between(x, [v - s for v, s in zip(values_oeq, stds_oeq)], [v + s for v, s in zip(values_oeq, stds_oeq)], color='green', alpha=0.2)
plt.plot(x, values_opp, marker='o', color='r', label='OPP', linestyle=":")
plt.fill_between(x, [v - s for v, s in zip(values_opp, stds_opp)], [v + s for v, s in zip(values_opp, stds_opp)], color='red', alpha=0.2)
plt.plot(x, values_acc, marker='o', color='c', label='ACC', linestyle="-.")
plt.fill_between(x, [v - s for v, s in zip(values_acc, stds_acc)], [v + s for v, s in zip(values_acc, stds_acc)], color='cyan', alpha=0.2)
plt.xlabel('window size')
plt.ylabel('metric values')
plt.title('Effect of Window Size on Fairness and Performance')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()
