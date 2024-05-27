import numpy as np
from mice import plot_mice
import matplotlib.pyplot as plt
import matplotlib
from plot_newton_cg import plot_newton_cg

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

methods = ['sgd_mice_bay_full', 'sgd_mice_bay_diag', 'sgd_mice_bay_trace']
names = [r'$P_{\ell} = \Sigma_{\ell}^{-1}$', r'$P_{\ell} = (\text{diag}(\Sigma_{\ell}))^{-1}$', r'$P_{\ell} = 1/\text{tr}(\Sigma_{\ell})\,I$']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
for method, name, color in zip(methods, names, colors):
    data = np.load(f'test_full/{method}.npy', allow_pickle=True)
    print(len(data[0]))
    if len(data) > 1:
        [opt_gap, update_iters, op_err_norm] = data[:3]
        update_iters = np.array(update_iters)
        ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, label=name, c=color)
        ax.semilogy(update_iters+1, np.array(opt_gap)[update_iters], 'o', ms=3., c=color, markeredgecolor='k')
    else:
        [opt_gap] = data
        ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, label=name, c=color)
ax.plot([],[], 'wo', label='Hessian update', markeredgecolor='k', ms=3.)
ax.legend()
ax.set_title('Quadratic function, dim.: 10, cond. number: 1000')
ax.set_xlabel('Iteration')
ax.set_ylabel('Optimality gap')
fig.tight_layout()
fig.savefig('test_full/convergence_full.pdf')

plot_newton_cg('test_full', 'sgd_mice_bay_full')
plot_newton_cg('test_full', 'sgd_mice_bay_diag')
plot_newton_cg('test_full', 'sgd_mice_bay_trace')
