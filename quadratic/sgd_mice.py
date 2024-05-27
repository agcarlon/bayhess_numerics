import numpy as np
from numpy.linalg import norm, eig, solve
import matplotlib.pyplot as plt
from mice import MICE, plot_mice
from time import perf_counter
from quad_d import QuadraticProblem

# cond_number = 1000
cond_number = 1e3
n_dim = 10
max_grads = 1000000

quad = QuadraticProblem(n_dim=n_dim, cond_number=cond_number)
df = MICE(quad.dobjf, sampler=quad.sampler, eps=1., max_cost=max_grads,
          min_batch=5)

x = np.ones(n_dim)/1e2
step_size = 2/(quad.L + quad.mu)/(1+df.eps**2)
opt_gap = [quad.Eobjf(x) - quad.f_opt]
t0 = perf_counter()

while True:
    grad = df(x)
    if df.terminate:
        break
    x = x - step_size*grad
    opt_gap.append(quad.Eobjf(x) - quad.f_opt)
    if opt_gap[-1] < 0:
        print(1)
    print(f'{df.k}, opt. gap:{opt_gap[-1]}')

print(f'Runtime: {perf_counter() - t0}')
print(opt_gap[-1])
print(df.k)

np.save('sgd_mice/sgd_mice_1e6.npy', [opt_gap])

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.loglog(opt_gap)
ax.set_xlabel('Iteration')
ax.set_ylabel('Optimality gap')
fig.tight_layout()
fig.savefig('sgd_mice/sgd_mice_1e6.pdf')