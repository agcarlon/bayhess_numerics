import numpy as np
from numpy.linalg import norm, eig
from time import perf_counter
from bayhess.cg import CGLinear
from scipy.stats import ortho_group

from functools import partial
from bayhess import BayHess
from bayhess.distributions import Secant, FrobeniusReg, LogBarrier


def simple_cg_test():
    dims = np.arange(2, 100, 1)
    # dims= [27]
    data = {
        "time": [],
        "residue": [],
        "dimension": []
    }

    for dim in dims:
        np.random.seed(0)
        print(dim)
        # A = np.random.randn(dim, dim, dim, dim)
        # A = (A + np.transpose(A, axes=[1, 0, 2, 3]))/2
        # A = (A + np.transpose(A, axes=[0, 1, 3, 2]))/2
        # A = np.einsum("ijkl, klmn", A, A)
        # A_ = np.block([[*a] for a in A])
        # q = ortho_group.rvs(dim=dim ** 2)
        C = np.random.randn(dim**2, dim**2)
        C = C @ C.T
        eigs, q = eig(C)
        eigs = np.linspace(1e-3, 1000, dim ** 2)
        B = q @ np.diag(eigs) @ q.T

        A = np.zeros((dim, dim, dim, dim))
        # for i in range(dim):
        #     for j in range(dim):
        #         for k in range(dim):
        #             for l in range(dim):
        #                 A[i, j, k, l] = B[i * dim + k, j * dim + l]

        t0 = perf_counter()
        for i in range(dim):
            for j in range(dim):
                A[i, j] = B[i * dim: (i + 1) * dim, j * dim: (j + 1) * dim]
        print(perf_counter() - t0)
        b = np.random.randn(dim, dim)
        b = (b + b.T) / 2
        b = b @ b.T

        x0 = np.random.randn(dim, dim)
        x0 = (x0 + x0.T) / 2
        x0 = x0.T @ x0


        def product(A, x):
            return np.einsum('ijkl, jl', A, x)


        def action(x):
            return product(A, x)


        cg = CGLinear()
        t0 = perf_counter()
        x = cg(action, b, x0)
        data["time"].append(perf_counter() - t0)
        data["residue"].append(norm(action(x) - b))
        data["dimension"].append(dim)
        np.save("cg_data", data)


if __name__ == "__main__":
    simple_cg_test()
