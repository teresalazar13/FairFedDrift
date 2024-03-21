import matplotlib.pyplot as plt


values_loss_small_alpha = [0.258359287, 0.211736094, 0.20599601699999998, 0.17740566700000002, 0.19526174000000002, 0.20383742, 0.182039949, 0.18179192319999998, 0.132890123]
stds_loss_small_alpha = [0.06856437173956992, 0.02229414428396233, 0.05213164139460432, 0.04771846072900229, 0.014646266150524511, 0.027626719925122414, 0.04473184067519752, 0.07895297126968377, 0.06398457699283061]

values_loss_big_alpha = [0.296115471, 0.231899466, 0.32038739299999996, 0.293578644, 0.27438804, 0.386134683, 0.338167025, 0.31246532799999993, 0.34718675499999996]
stds_loss_big_alpha = [0.07193280971756606, 0.015047539438173272, 0.0408309909596661, 0.06101727669920494, 0.047555969334688156, 0.04691475525679482, 0.07219906656568233, 0.11434052418088249, 0.16961093433047603]

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



values_loss_small_alpha = [0.46806581300000005, 0.50004836, 0.652607422, 0.47753813, 0.728412898, 0.704488607, 0.582523496, 0.5681368480000001, 0.5183965140000002]
stds_loss_small_alpha = [0.2373503879730765, 0.23048829063294604, 0.2560852971441986, 0.23806497346171496, 0.2328013544956805, 0.19900562837735988, 0.2669970538413237, 0.09059631946750002, 0.26075045601810576]

values_loss_big_alpha = [0.291106001, 0.226125926, 0.461170849, 0.384375486, 0.394118348, 0.6080267920000001, 0.5055520450000002, 0.47279984199999997, 0.578500263]
stds_loss_big_alpha = [0.09306020490900793, 0.05216386206621908, 0.09341785413392946, 0.09444786926245165, 0.09969416356347105, 0.0548281419604851, 0.0971243007476161, 0.11879799730489621, 0.28358944324869184]

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