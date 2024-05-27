import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib


def plot_newton_cg(dataset, method='sgd_mice_bay'):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        'axes.grid': True
    })

    colors = [f'C{i}' for i in range(12)]

    with open(f'{dataset}/{method}.txt') as f:
        lines = f.readlines()

    lines = np.array(lines)

    iterations = []
    times = []
    residues = []  # run, homotopy, newton iteration
    tols = []
    cg_iters = []  # run, homotopy, newton_iteration
    num_curv_pairs = []

    for i, line in enumerate(lines):
        if re.search('Bay iteration', line):
            iterations.append(int(
                re.search('.*-Bay iteration: (.*)', line).groups()[0]))
            residues.append([])
            cg_iters.append([])
            tols.append([])
        if re.search('homotopy', line):
            ll = re.split(',', line[:-1])
            out = [float(re.split(':', ee)[1]) for ee in ll]
            tols[-1].append(out[2])
            residues[-1].append([])
            cg_iters[-1].append([])
        if re.search('CG - Ended', line):
            ll = re.split(',', line[:-1])
            out = int(re.split(':', ll[1])[1])
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
            num_curv_pairs.append(int(
                re.search('Starting Bay-Hess algorithm: (.*) curv. pairs',
                          line).groups()[0]))

    fig, axs = plt.subplots(len(iterations), 1,
                            figsize=(6, 1 + 2 * len(iterations)),
                            squeeze=False)
    axs = axs[:, 0]
    for ax, iter_newton, time, pairs, residues_run, cg, tol in zip(axs, iterations,
                                                            times, num_curv_pairs, residues,
                                                            cg_iters, tols):
        ax.set_title(
            f'Iteration: {iter_newton}, curvature pairs: {pairs}, time spent: {time:.2f}s')
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
    fig.savefig(f'{dataset}/newton_cg_{method}.pdf')
