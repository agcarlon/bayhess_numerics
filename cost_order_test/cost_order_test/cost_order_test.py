from bayhess import BayHess
import numpy as np
from numpy.linalg import eig, norm, inv
from scipy.stats import ortho_group
from time import perf_counter

strong_conv = 1e-3
smooth = 1000
dims = np.arange(2, 1000, 5)
data = {"obtaining curv. pairs": [],
        "finding Hessian": [],
        "inverting Hessian": [],
        "error": [],
        "dims": []}

for dim in dims:
    np.random.seed(0)
    q = ortho_group.rvs(dim=dim)
    eigs = np.linspace(strong_conv, smooth, dim)
    A = q @ np.diag(eigs) @ q.T
    A_inv = q @ np.diag(1/eigs) @ q.T

    q = ortho_group.rvs(dim=dim)
    eigs = np.linspace(1, 100, dim)
    cov_matrix = q @ np.diag(eigs) @ q.T

    bay = BayHess(n_dim=dim, strong_conv=strong_conv, pairs_to_use=1000,
                  smooth=smooth, log=f"logs/{dim}")
    data["obtaining curv. pairs"].append(0)
    data["finding Hessian"].append([])
    data["inverting Hessian"].append([])
    for step in range(10):
        sk = np.random.randn(100, dim)
        yk = (sk @ A)[:, np.newaxis, :] + np.random.randn(100, 10, dim) @ cov_matrix
        t0 = perf_counter()
        for s, y in zip(sk, yk):
            bay.update_curv_pairs(s, y)
        data["obtaining curv. pairs"][-1] += perf_counter() - t0

        bay.hess = np.eye(dim)/smooth
        t0 = perf_counter()
        bay.find_hess()
        data["finding Hessian"][-1].append(perf_counter() - t0)

        t0 = perf_counter()
        bay.eval_inverse_hess()
        data["inverting Hessian"][-1].append(perf_counter() - t0)

    error = np.abs(eig(A @ bay.inv_hess - np.eye(dim))[0]).max()
    data["error"].append(error)
    data["dims"].append(dim)
    np.save("data", data)
    print(f"dimension: {dim}")
    print(f"time spent assembling curv. pairs { data['obtaining curv. pairs'][-1] }")
    print(f"time spent finding Hessian { data['finding Hessian'][-1] }")




