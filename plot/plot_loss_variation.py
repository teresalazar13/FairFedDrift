import matplotlib.pyplot as plt


values_loss_small_alpha = [0.43773401199999995, 0.41899527999999997, 0.4475290009999999, 0.42731604999999995, 0.44205220799999995, 0.444688949, 0.41397825299999996, 0.406223066, 0.42279992]
stds_loss_small_alpha = [0.43773401199999995, 0.41899527999999997, 0.4475290009999999, 0.42731604999999995, 0.44205220799999995, 0.444688949, 0.41397825299999996, 0.406223066, 0.42279992]

values_loss_big_alpha = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
stds_loss_big_alpha = [0.15, 0.3, 0.2, 0.25, 0.1, 0.3, 0.2, 0.15, 0.25, 0.35]

x = range(1, 10)
plt.plot(x, values_loss_small_alpha, marker='o', color='b', label=r'FedAvg ($\alpha$=0.1)', linestyle="-")
plt.fill_between(x, [v - s for v, s in zip(values_loss_small_alpha, stds_loss_small_alpha)], [v + s for v, s in zip(values_loss_small_alpha, stds_loss_small_alpha)], color='blue', alpha=0.2)
plt.plot(x, values_loss_big_alpha, marker='o', color='r', label=r'FedAvg ($\alpha$=0.5)', linestyle="--")
plt.fill_between(x, [v - s for v, s in zip(values_loss_big_alpha, stds_loss_big_alpha)], [v + s for v, s in zip(values_loss_big_alpha, stds_loss_big_alpha)], color='red', alpha=0.2)
plt.xlabel(r'$t$')
plt.ylabel(r'$\ell$')
plt.title(r'Overall loss ($\ell$) values spanning distinct $\alpha$ values')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()



values_loss_small_alpha = [0.335367123, 0.3164652, 0.5400063530000001, 0.5228019560000001, 0.660042324, 0.6754732299999999, 0.5886080199999999, 0.57810888, 0.786024596]
stds_loss_small_alpha = [0.335367123, 0.3164652, 0.5400063530000001, 0.5228019560000001, 0.660042324, 0.6754732299999999, 0.5886080199999999, 0.57810888, 0.786024596]

values_loss_big_alpha = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
stds_loss_big_alpha = [0.15, 0.3, 0.2, 0.25, 0.1, 0.3, 0.2, 0.15, 0.25, 0.35]

x = range(1, 10)
plt.plot(x, values_loss_small_alpha, marker='o', color='b', label=r'FedAvg ($\alpha$=0.1)', linestyle="-")
plt.fill_between(x, [v - s for v, s in zip(values_loss_small_alpha, stds_loss_small_alpha)], [v + s for v, s in zip(values_loss_small_alpha, stds_loss_small_alpha)], color='blue', alpha=0.2)
plt.plot(x, values_loss_big_alpha, marker='o', color='r', label=r'FedAvg ($\alpha$=0.5)', linestyle="--")
plt.fill_between(x, [v - s for v, s in zip(values_loss_big_alpha, stds_loss_big_alpha)], [v + s for v, s in zip(values_loss_big_alpha, stds_loss_big_alpha)], color='red', alpha=0.2)
plt.xlabel(r'$t$')
plt.ylabel(r'$\ell_{S=0}$')
plt.title(r'Group-specific loss ($\ell_{S=0}$) values spanning distinct $\alpha$ values')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.show()