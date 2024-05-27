import numpy as np
from numpy.linalg import norm, eig, solve, lstsq
import matplotlib.pyplot as plt
from mice import MICE, plot_mice
from quad_d import QuadraticProblem
from bayhess import BayHess
from bfgs import BFGS
from time import perf_counter

from plot_newton_cg import plot_newton_cg

# cond_number = 1000
cond_number = 1e3
n_dim = 10
max_grads = 1000000

quad = QuadraticProblem(n_dim=n_dim, cond_number=cond_number, seed=0)
# df = MICE(quad.dobjf, sampler=quad.sampler, eps=0.7, max_cost=max_grads,
#           restart_factor=1000, min_batch=50, mice_type='resampling', re_min_n=10, adpt=True, restart_param=2.)
df = MICE(quad.dobjf, sampler=quad.sampler, eps=0.7, min_batch=5, max_cost=max_grads, aggr_cost=2., mice_type='resampling')
bay = BayHess(n_dim=n_dim, strong_conv=quad.mu, smooth=quad.L, verbose=True,
              penal=1e-2, reg_param=1e-2,
              log='sgd_mice_bay/sgd_mice_bay_1e6.txt')

bfgs = BFGS(dim=n_dim)


def get_bfgs_dir_and_eigs(grad, bay):
    bfgs.sk = bay.sk
    bfgs.yk = bay.yk
    bfgs_hess = bfgs.hess(bay.sk, bay.yk, np.eye(n_dim)*(quad.L + quad.mu)/2)
    d = solve(bfgs_hess, grad)
    eigs_bfgs = np.linalg.eig(bfgs_hess)[0]
    eigs = [eigs_bfgs.min(), eigs_bfgs.max()]
    return d, eigs, bfgs_hess


def compute_dennis_more(hess, tr_hess, sk):
    ek = (hess - tr_hess) @ sk
    return norm(ek) / norm(sk)


x = np.ones(n_dim)/1e2
opt_gap = [quad.Eobjf(x) - quad.f_opt]
hess = np.eye(n_dim)/(2/(quad.L + quad.mu)/(1+df.eps**2))
n_hess_update = 5
# update_when = max_grads/n_hess_update
# last_update = - update_when
update_when = n_dim
last_update = 0
update_iters = []
op_err_norm = []
eigs_hess = np.linalg.eig(bay.hess)[0]
eigs = [[eigs_hess.min(), eigs_hess.max()]]
eigs_bfgs = [[eigs_hess.min(), eigs_hess.max()]]
err_bay = []
err_bfgs = []
t0 = perf_counter()

while True:
    grad = df(x)
    bay.update_curv_pairs_mice(df)
    if df.terminate:
        break
    # if df.counter > last_update + update_when and len(bay.sk_all) >= n_dim:
    if df.k > last_update + update_when and len(bay.sk_all) >= n_dim:
        bay.print(f'SGD-MICE-Bay iteration: {df.k}')
        last_update += update_when
        t0_ = perf_counter()
        hess = bay.find_hess()
        bay.print(f'Time spent in Bay-Hessian: {perf_counter() - t0_:.2f}')
        update_iters.append(df.k-1)
        op_err_norm.append(np.max(np.abs(eig(hess - quad.true_hess)[0])))
        d = -solve(hess, grad)
        err_bay.append(compute_dennis_more(bay.hess, quad.true_hess, d))
        eigs_hess = np.linalg.eig(bay.hess)[0]
        eigs.append([eigs_hess.min(), eigs_hess.max()])
        d_bfgs, eigs_bfgs_, bfgs_hess = get_bfgs_dir_and_eigs(grad, bay)
        err_bfgs.append(compute_dennis_more(bfgs_hess, quad.true_hess, d_bfgs))
        eigs_bfgs.append(eigs_bfgs_)

    x = x - 1/(1+df.eps**2)*solve(bay.hess, grad)
    opt_gap.append(quad.Eobjf(x) - quad.f_opt)

print(f'Runtime: {perf_counter() - t0}')
print(opt_gap[-1])
print(df.k)

np.save('sgd_mice_bay/sgd_mice_bay_1e6.npy', [opt_gap, update_iters, op_err_norm, eigs, err_bay, eigs_bfgs, err_bfgs])
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.loglog(opt_gap)
ax.loglog(update_iters, np.array(opt_gap)[update_iters], 'rx')
ax.set_xlabel('Iteration')
ax.set_ylabel('Optimality gap')
fig.tight_layout()
fig.savefig('sgd_mice_bay/sgd_mice_bay.pdf')

eigs = np.array(eigs)
eigs_bfgs = np.array(eigs_bfgs)

fig, axs = plt.subplots(2, 1, figsize=(6, 6))
axs[0].semilogy(update_iters, err_bay, label='SGD-MICE-Bay', c='C0')
axs[0].semilogy(update_iters, err_bfgs, label='BFGS', c='C1')
axs[0].legend()
axs[0].set_ylabel(r'Dennis-Mor\'{e} Error')
axs[0].set_xlabel('Iterations')

axs[1].plot([0, len(eigs)], [quad.L, quad.L], 'k--', label='Largest eigenvalue')
axs[1].plot([0, len(eigs)], [quad.mu, quad.mu], 'k-.', label='Smallest eigenvalue')
axs[1].plot(eigs[:, 1], '--', c='C0')
axs[1].plot(eigs[:, 0], '-.', c='C0')
axs[1].plot(eigs_bfgs[:, 1], '--', c='C1')
axs[1].plot(eigs_bfgs[:, 0], '-.', c='C1')
axs[1].set_ylabel(r'Eigenvalues')
axs[1].set_xlabel('Iterations')
axs[1].legend()

fig.tight_layout()
fig.savefig('sgd_mice_bay/err_and_eigs_1e6.pdf')

plot_newton_cg(folder='sgd_mice_bay', method='sgd_mice_bay_1e6')
