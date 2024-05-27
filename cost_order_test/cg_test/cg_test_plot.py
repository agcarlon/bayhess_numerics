import matplotlib.pyplot as plt
import numpy as np

data = np.load("cg_data.npy", allow_pickle=True).tolist()
style = {
    'linestyle': '-',
    'marker': 'o',
    'c': 'C0',
    'ms': 2.
}

rate=5
max_time = np.max(data["time"])
max_dim = np.max(data['dimension'])

fig, axs = plt.subplots(2, 1, figsize=(6, 5))
axs[0].loglog(data["dimension"], data["time"], **style, label="time")
axs[0].loglog([max_dim/2, max_dim], [max_time/2**rate, max_time], '--k', label=rf"$O(d^{{{rate}}})$")
axs[0].legend()
axs[0].set_ylabel("time (s)")
axs[0].grid()


axs[1].loglog(data["dimension"], data["residue"], **style, label="residue")
# axs[1].loglog([200, 400], [100, 100*2**rate], '--k', label=rf"$O(d^{{{rate}}})$")
# axs[1].loglog([200, 400], [100, 800], '-.k', label=r"$O(d^3)$")
axs[1].legend()
axs[1].set_ylabel("Residue")
axs[1].set_xlabel("Dimension")
axs[1].grid()


fig.tight_layout()
fig.savefig("cg_data.pdf")
