import matplotlib.pyplot as plt
import numpy as np

data = np.load("data_cg_newton_direction.npy", allow_pickle=True).tolist()
style = {
    'linestyle': '-',
    'marker': 'o',
    'c': 'C0',
    'ms': 2.
}

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

rate=2
max_time = np.max(data["time"])*.7
max_dim = np.max(data['dims'])

ax.loglog(data["dims"], data["time"], **style, label="Finding Newton direction with CG")
ax.loglog([max_dim/4, max_dim], [max_time/4**rate, max_time], '--k', label=rf"$O(d^{{{rate}}})$")
# ax.loglog([200, 400], [100, 800], '-.k', label=r"$O(d^3)$")
ax.legend()
ax.set_ylabel("time (s)")
ax.grid()

fig.tight_layout()
fig.savefig("Newton_direction_times.pdf")
