import numpy as np
from mice import plot_mice
import matplotlib.pyplot as plt
import matplotlib


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# methods = ['sgd', 'sgd_mice', 'sgd_bay_fixed', 'sgd_bay_decr', 'sgd_mice_bay']
# names = ['SGD', 'SGD-MICE', 'SGD-Bay fixed step', 'SGD-Bay decr. step', 'SGD-MICE-Bay']
# colors = ['C0', 'C1', 'C2', 'C3', 'C4']
#
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# for method, name, color in zip(methods, names, colors):
#     data = np.load(f'{method}/{method}.npy', allow_pickle=True)
#     print(len(data[0]))
#     if len(data) > 1:
#         [opt_gap, update_iters, op_err_norm] = data[:3]
#         update_iters = np.array(update_iters)
#         ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, label=name, c=color)
#         ax.loglog(update_iters+1, np.array(opt_gap)[update_iters], 'o', ms=3., c=color, markeredgecolor='k')
#     else:
#         [opt_gap] = data
#         ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, label=name, c=color)
# ax.plot([],[], 'wo', label='Hessian update', markeredgecolor='k', ms=3.)
# ax.legend()
# ax.set_title('Quadratic function, dim.: 10, cond. number: 1000')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Optimality gap')
# fig.tight_layout()
# fig.savefig('convergence.pdf')



fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# SGD
# [opt_gap] = np.load(f'sgd/sgd.npy', allow_pickle=True)
[opt_gap] = np.load(f'sgd/sgd_1e6.npy', allow_pickle=True)
ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, '-', label='SGD', ms=3, c='C0')

# SGD-MICE
# [opt_gap] = np.load(f'sgd_mice/sgd_mice.npy', allow_pickle=True)
[opt_gap] = np.load(f'sgd_mice/sgd_mice_1e6.npy', allow_pickle=True)
ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, label='SGD-MICE', c='C1')

# SGD-MICE-Bay
# [opt_gap, update_iters, op_err_norm, eigs, err_bay, eigs_bfgs, err_bfgs] = np.load(f'sgd_mice_bay/sgd_mice_bay.npy', allow_pickle=True)
[opt_gap, update_iters, op_err_norm, eigs, err_bay, eigs_bfgs, err_bfgs] = np.load(f'sgd_mice_bay/sgd_mice_bay_1e6.npy', allow_pickle=True)
update_iters = np.array(update_iters)
ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, label='SGD-MICE-Bay', c='C2')
ax.loglog(update_iters+1, np.array(opt_gap)[update_iters], 's', ms=3., c='C2', markeredgecolor='k')

# SGD-Bay-decr
# [opt_gap, update_iters, op_err_norm] = np.load(f'sgd_bay_decr/sgd_bay_decr.npy', allow_pickle=True)
[opt_gap, update_iters, op_err_norm] = np.load(f'sgd_bay_decr/sgd_bay_decr_1e6.npy', allow_pickle=True)
update_iters = np.array(update_iters)
ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, label='SGD-Bay decr. step', c='C3')
ax.loglog(update_iters+1, np.array(opt_gap)[update_iters], 's', ms=3., c='C3', markeredgecolor='k')

# SGD-Bay-fixed
# [opt_gap, update_iters, op_err_norm] = np.load(f'sgd_bay_fixed/sgd_bay_fixed.npy', allow_pickle=True)
[opt_gap, update_iters, op_err_norm] = np.load(f'sgd_bay_fixed/sgd_bay_fixed_1e6.npy', allow_pickle=True)
update_iters = np.array(update_iters)
ax.semilogy(np.arange(1, len(opt_gap)+1), opt_gap, label='SGD-Bay fixed step', c='C4')
ax.loglog(update_iters+1, np.array(opt_gap)[update_iters], 's', ms=3., c='C4', markeredgecolor='k')

lims = ax.get_ylim()[0]
lims_ = 10**(np.floor(np.log10(lims)))
ax.set_ylim(bottom=lims_)

ax.plot([], [], 'ws', label='Hessian update', markeredgecolor='k', ms=3.)
ax.legend(ncol=2, prop={'size': 10})
ax.set_title(r'Quadratic function, dim.: 10, cond. number: $10^6$')
ax.set_xlabel('Iteration')
ax.set_ylabel('Optimality gap')
fig.tight_layout()
fig.savefig('convergence_1e6.pdf')
plt.close('all')

data = np.load(f'sgd_mice_bay/sgd_mice_bay_1e6.npy', allow_pickle=True)
[opt_gap, update_iters, op_err_norm, eigs, err_bay, eigs_bfgs, err_bfgs] = data

eigs = np.array(eigs)
eigs_bfgs = np.array(eigs_bfgs)

fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
# axs[0].semilogy(update_iters, err_bay, 'o--', ms=3, label='SGD-MICE-Bay', c='C0')
# axs[0].semilogy(update_iters, err_bfgs, 'o--', ms=3, label='BFGS', c='C1')
# axs[0].legend()
# axs[0].set_ylabel(r'Dennis-Mor\'{e} Error')
# axs[0].set_xlabel('Iterations')

update_iters.insert(0, 0)
n_updates = len(update_iters)

ax.plot(update_iters, [1000*1.0]*n_updates, 'k--', label='Largest eigenvalue')
ax.plot(update_iters, [1/1.0]*n_updates, 'k-.', label='Smallest eigenvalue')
# ax.plot(update_iters, [1000*1.05]*n_updates, 'k--', label='Largest eigenvalue')
# ax.plot(update_iters, [1/1.05]*n_updates, 'k-.', label='Smallest eigenvalue')
ax.plot(update_iters, eigs[:, 1], 'o--', ms=3, c='C0')
ax.plot(update_iters, eigs[:, 0], 'o-.', ms=3, c='C0')
ax.plot(update_iters, eigs_bfgs[:, 1], 'o--', ms=3, c='C1')
ax.plot(update_iters, eigs_bfgs[:, 0], 'o-.', ms=3, c='C1')
ax.plot([], [], c='C0', label='SGD-MICE-Bay')
ax.plot([], [], c='C1', label='BFGS')
ax.set_title('Quadratic function, dim.: 10, cond. number: 1000')
ax.set_ylabel(r'Eigenvalues')
ax.set_xlabel('Iterations')
ax.legend()

fig.tight_layout()
fig.savefig('err_and_eigs_1e6.pdf')


# methods = ['sgd_bay_fixed', 'sgd_bay_decr', 'sgd_mice_bay']
# names = ['SGD-Bay fixed step', 'SGD-Bay decr. step', 'SGD-MICE-Bay']
# colors = ['C2', 'C3', 'C4']
#
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# for method, name, color in zip(methods, names, colors):
#     data = np.load(f'{method}/{method}.npy', allow_pickle=True)
#     [opt_gap, update_iters, op_err_norm] = data
#     ax.loglog(update_iters, op_err_norm, 'o--', c=color, label=name, ms=3., markeredgecolor='k')
# ax.legend()
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Hessian error norm')
# fig.tight_layout()
# fig.savefig('hess_error.pdf')



