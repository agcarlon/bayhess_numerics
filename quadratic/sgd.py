import numpy as np
from numpy.linalg import norm, eig, solve
import matplotlib.pyplot as plt
from mice import MICE, plot_mice
from quad_d import QuadraticProblem

# cond_number = 1000
cond_number = 1e3
n_dim = 10
max_cost = 1000000

quad = QuadraticProblem(n_dim=n_dim, cond_number=cond_number)

x = np.ones(n_dim)/1e2
step_size = 1/quad.L
opt_gap = [quad.Eobjf(x) - quad.f_opt]

for k in range(max_cost):
    smp = quad.sampler(1)
    grad = quad.dobjf(x, smp).mean(axis=0)
    x = x - step_size/np.sqrt(k+1)*grad
    opt_gap.append(quad.Eobjf(x) - quad.f_opt)

np.save('sgd/sgd_1e6.npy', [opt_gap])

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.loglog(opt_gap)
ax.set_xlabel('Iteration')
ax.set_ylabel('Optimality gap')
fig.tight_layout()
fig.savefig('sgd/sgd_1e6.pdf')