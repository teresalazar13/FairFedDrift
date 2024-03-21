import matplotlib.pyplot as plt

values_aeq = [0.90, 0.91, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92]
stds_aeq = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.02, 0.04, 0.04, 0.04]

values_oeq = [0.86, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 0.88, 0.88, 0.88]
stds_oeq = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]

values_opp = [0.86, 0.87, 0.87, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88]
stds_opp = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.04, 0.04, 0.04]

values_acc = [0.93, 0.93, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94]
stds_acc = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

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
