from bayhess import BayHess
from bayhess.distributions import Secant, FrobeniusReg, LogBarrier
import numpy as np
from numpy.linalg import eig, norm, inv
from scipy.stats import ortho_group
from time import perf_counter

strong_conv = 1e-3
smooth = 1000
dims = np.arange(2, 1000, 5)
data = {"time": [],
        "dims": []}

for dim in dims:
    np.random.seed(0)
    q = ortho_group.rvs(dim=dim)
    eigs = np.linspace(strong_conv, smooth, dim)
    A = q @ np.diag(eigs) @ q.T
    A_inv = q @ np.diag(1 / eigs) @ q.T

    q = ortho_group.rvs(dim=dim)
    eigs = np.linspace(1, 100, dim)
    cov_matrix = q @ np.diag(eigs) @ q.T

    bay = BayHess(n_dim=dim, strong_conv=strong_conv, pairs_to_use=1000,
                  smooth=smooth, log=f"logs_hess_log_post/{dim}")

    sk = np.random.randn(100, dim)
    yk = (sk @ A)[:, np.newaxis, :] + np.random.randn(100, 10,
                                                      dim) @ cov_matrix
    for s, y in zip(sk, yk):
        bay.update_curv_pairs(s, y)

    factor = 0
    bay.pk = np.asarray(bay.pk_raw)
    for s, y, p in zip(bay.sk, bay.yk, bay.pk):
        factor += norm(p * (bay.hess @ s - y)) * norm(s)
    lkl = Secant(bay.sk, bay.yk, bay.pk / factor)
    prior_1 = FrobeniusReg(bay.hess, np.eye(bay.n_dim) * bay.reg_param)
    prior_2 = LogBarrier(bay.strong_conv, bay.smooth,
                         bay.penal * bay.homotopy_factor ** bay.homotopy_steps)
    post = prior_1 * prior_2 * lkl
    t0 = perf_counter()
    post.hess_log_pdf_action(A, A)
    data["time"].append(perf_counter() - t0)

    data["dims"].append(dim)
    np.save("data_hess_log_post", data)
    print(f"dimension: {dim}")
