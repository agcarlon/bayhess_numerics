import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
from mice import MICE, plot_mice


def plot_newton_cg(folder='sgd_mice_bay', method='sgd_mice_bay'):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        'axes.grid': True
    })

    colors = [f'C{i}' for i in range(12)]

    with open(f'{folder}/{method}') as f:
        lines = f.readlines()

    lines = np.array(lines)

    iterations = []
    times = []
    residues = []  # run, homotopy, newton iteration
    tols = []
    cg_iters = []  # run, homotopy, newton_iteration
    num_curv_pairs = []

    for i, line in enumerate(lines):
        if re.search('homotopy', line):
            ll = re.split(',', line[:-1])
            out = [float(re.split(':', ee)[1]) for ee in ll]
            tols[-1].append(out[2])
            residues[-1].append([])
            cg_iters[-1].append([])
        if re.search('CG - Ended', line):
            ll = re.split(',', line[:-1])
            out = int(re.split(':', ll[2])[1])
            cg_iters[-1][-1].append(out)
        if re.search('Decreased grad. norm from', line):
            ll = re.split(',', line[:-1])
            items = re.search('.* Decreased grad. norm from (.*) to (.*)',
                              ll[0])
            outputs = items.groups()
            outs = [float(item) for item in outputs]
            if len(residues[-1][-1]) == 0:
                residues[-1][-1].extend(outs)
            else:
                residues[-1][-1].append(outs[1])
        if re.search('Time spent in Bay-Hessian: (.*)', line):
            times.append(float(
                re.search('Time spent in Bay-Hessian: (.*)', line).groups()[
                    0]))
        if re.search(f'Starting Bay-Hess algorithm: (.*) curv. pairs', line):
            iterations.append(0)
            residues.append([])
            cg_iters.append([])
            tols.append([])
            num_curv_pairs.append(int(
                re.search('Starting Bay-Hess algorithm: (.*) curv. pairs',
                          line).groups()[0]))

    print(len(iterations))
    fig, axs = plt.subplots(len(iterations), 1,
                            figsize=(6, 1 + 2 * len(iterations)),
                            squeeze=False)
    axs = axs[:, 0]
    for ax, pairs, residues_run, cg, tol in zip(axs, num_curv_pairs, residues,
                                                            cg_iters, tols):
        ax.set_title(
            f'Curvature pairs: {pairs}')
        k = 0
        for res_homotopy, tol_, cg_, color in zip(residues_run, tol, cg,
                                                  colors):
            ax.semilogy(np.arange(k, k + len(res_homotopy)), res_homotopy,
                        'o-', c=color)
            ax.semilogy([k, k + len(res_homotopy) - 1], [tol_, tol_], '-.',
                        c=color, alpha=.5)
            for i, cg__ in enumerate(cg_):
                ax.text(k + i, res_homotopy[i] * 1.5, cg__, ha='left',
                        va='bottom')
            k += max(0, len(res_homotopy) - 1)
        lims = ax.get_ylim()
        ax.set_ylim(top=lims[1] * 10)
        ax.set_ylabel('Residue')
        ax.set_xlabel('Newton iteration')
    tol_lines = axs[0].plot([], [], '-.', alpha=.5, c='k',
                            label='Tolerance for central-path step')
    axs[0].legend()

    fig.tight_layout()
    fig.savefig(f'{folder}/newton_cg_{method}.pdf')


def get_data_from_log(dim, folder='sgd_mice_bay'):
    with open(f'{folder}/{dim}') as f:
        lines = f.readlines()

    lines = np.array(lines)

    iterations = []
    times = []
    residues = []  # run, homotopy, newton iteration
    tols = []
    cg_iters = []  # run, homotopy, newton_iteration
    num_curv_pairs = []

    for i, line in enumerate(lines):
        if re.search('homotopy', line):
            ll = re.split(',', line[:-1])
            out = [float(re.split(':', ee)[1]) for ee in ll]
            tols[-1].append(out[2])
            residues[-1].append([])
            cg_iters[-1].append([])
        if re.search('CG - Ended', line):
            ll = re.split(',', line[:-1])
            out = int(re.split(':', ll[2])[1])
            cg_iters[-1][-1].append(out)
        if re.search('Decreased grad. norm from', line):
            ll = re.split(',', line[:-1])
            items = re.search('.* Decreased grad. norm from (.*) to (.*)',
                              ll[0])
            outputs = items.groups()
            outs = [float(item) for item in outputs]
            if len(residues[-1][-1]) == 0:
                residues[-1][-1].extend(outs)
            else:
                residues[-1][-1].append(outs[1])
        if re.search('Time spent in Bay-Hessian: (.*)', line):
            times.append(float(
                re.search('Time spent in Bay-Hessian: (.*)', line).groups()[
                    0]))
        if re.search(f'Starting Bay-Hess algorithm: (.*) curv. pairs', line):
            iterations.append(0)
            residues.append([])
            cg_iters.append([])
            tols.append([])
            num_curv_pairs.append(int(
                re.search('Starting Bay-Hess algorithm: (.*) curv. pairs',
                          line).groups()[0]))
    return [iterations, residues, cg_iters, tols, num_curv_pairs]


def plot_num_cg():
    dims = np.arange(2, 697, 5)
    cg_iters = []
    for dim in dims:
        data = get_data_from_log(dim, folder="logs")
        cg_iters.append(np.sum(np.sum(np.sum(data[2][7]))))

    rate=1
    max_time = np.max(cg_iters)*1.
    max_dim = np.max(dims)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.semilogy(dims, cg_iters)
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('CG iterations')
    ax.semilogy([max_dim/4, max_dim], [max_time/4**rate, max_time], '--k', label=rf"$O(d^{{{rate}}})$")
    ax.legend()

    # ax.grid()
    fig.tight_layout()
    fig.savefig("CG_iterations.pdf")


if __name__ == '__main__':
    # plot_newton_cg(folder='logs', method='242')
    plot_num_cg()