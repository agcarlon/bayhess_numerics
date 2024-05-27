import numpy as np
from numpy.linalg import norm, eig, solve, lstsq
import matplotlib.pyplot as plt
from mice import MICE, plot_mice
from quad_d import QuadraticProblem
from bayhess import BayHess

# cond_number = 1000
cond_number = 1e3
n_dim = 10
max_cost = 1000000
batch_size = 1000
iters = int(max_cost / (2*batch_size))

quad = QuadraticProblem(n_dim=n_dim, cond_number=cond_number)
bay = BayHess(n_dim=n_dim, strong_conv=quad.mu, smooth=quad.L, verbose=False, penal=1e-2, reg_param=1e-2,)

x = np.ones(n_dim)/1e2
opt_gap = [quad.Eobjf(x) - quad.f_opt]
hess = np.eye(n_dim)/(1/quad.L)
n_hess_update = 5
# update_when = max_cost/n_hess_update
# last_update = - update_when
update_when = np.ceil(iters/n_hess_update).astype('int')
last_update = 0
update_iters = []
op_err_norm = []

smp = quad.sampler(batch_size)
grads_ = quad.dobjf(x, smp)
sk = - solve(hess, grads_.mean(axis=0))
x = x + sk
grads = quad.dobjf(x, smp)
yk = grads - grads_
bay.update_curv_pairs(sk, yk)

for k in range(iters):
    smp = quad.sampler(batch_size)
    grads_ = quad.dobjf(x, smp)
    # if df.counter > last_update + update_when and len(bay.sk_all) >= n_dim:
    if k > last_update + update_when and len(bay.sk) >= n_dim:
        print(k)
        last_update += update_when
        hess = bay.find_hess()
        update_iters.append(k)
        op_err_norm.append(np.max(np.abs(eig(hess - quad.true_hess)[0])))
    grad = 0.5*grads_.mean(axis=0) + 0.5*grads.mean(axis=0)
    sk = - solve(hess, grad)
    x = x + sk
    grads = quad.dobjf(x, smp)
    yk = grads - grads_
    bay.update_curv_pairs(sk, yk)
    opt_gap.append(quad.Eobjf(x) - quad.f_opt)

np.save('sgd_bay_fixed/sgd_bay_fixed_1e6.npy', [opt_gap, update_iters, op_err_norm])
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.loglog(opt_gap)
ax.loglog(update_iters, np.array(opt_gap)[update_iters], 'rx')
ax.set_xlabel('Iteration')
ax.set_ylabel('Optimality gap')
fig.tight_layout()
fig.savefig('sgd_bay_fixed/sgd_bay_fixed_1e6.pdf')