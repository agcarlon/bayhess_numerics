import matplotlib.pyplot as plt
import numpy as np

data = np.load("data.npy", allow_pickle=True).tolist()
style = {
    'linestyle': '-',
    'marker': 'o',
    'c': 'C0',
    'ms': 2.
}

fig, axs = plt.subplots(3, 1, figsize=(6, 7))
axs[0].plot(data["dims"], data["obtaining curv. pairs"], **style,
            label="obtaining curv. pairs")
axs[0].legend()
axs[0].set_ylabel("time (s)")
axs[0].grid()

rate = 2
max_time = np.max(data["finding Hessian"]) * .7
max_dim = np.max(data['dims'])

axs[1].loglog(data["dims"], [*map(lambda x: x[8], data["finding Hessian"])],
              **style, label="finding Hessian")
axs[1].loglog([max_dim / 4, max_dim], [max_time / 4 ** rate, max_time], '--k',
              label=rf"$O(d^{{{rate}}})$")
# axs[1].loglog([200, 400], [100, 800], '-.k', label=r"$O(d^3)$")
axs[1].legend()
axs[1].set_ylabel("time (s)")
axs[1].grid()

rate = 2

inv_time = np.array([*map(lambda x: x[6], data["inverting Hessian"])])

cs = np.polyfit(data['dims'], np.log(inv_time), 1)


def regression(d):
    return np.exp(cs[1])*np.exp(cs[0])**d


max_time = np.max(data["inverting Hessian"]) * .2
max_dim = np.max(data['dims'])

# axs[2].semilogy(data["dims"], inv_time, **style,
#               label="inverting Hessian")
# axs[2].semilogy([max_dim / 2, max_dim], [regression(max_dim/2) , regression(max_dim)], '--k',
#               label=rf"$O(c^d)$")
axs[2].loglog(data["dims"], inv_time, **style,
              label="inverting Hessian")
axs[2].loglog([max_dim / 2, max_dim], [max_time/2**rate, max_time], '--k',
              label=rf"$O(d^{{{rate}}})$")
axs[2].legend()
axs[2].set_ylabel("time (s)")
axs[2].grid()

fig.tight_layout()
fig.savefig("data.pdf")
