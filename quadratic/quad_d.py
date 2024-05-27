import numpy as np
from numpy.linalg import eig, norm
import matplotlib.pyplot as plt


class QuadraticProblem:
    def __init__(self, n_dim=2, cond_number=100, rand_param=0, seed=0):
        self.seed = seed
        self.n_dim = n_dim
        self.kappa = cond_number
        self.rand_param = rand_param
        np.random.seed(self.seed)
        hess_1 = np.random.randn(n_dim, n_dim)
        hess_1 = hess_1 @ hess_1.T
        self.D = np.random.uniform(1, cond_number, n_dim)
        self.D[0] = cond_number
        self.D[-1] = 1
        eigs = eig(hess_1)
        self.Q = eigs[1]
        self.true_hess = self.Q @ np.diag(self.D) @ self.Q.T
        self.true_inv = np.linalg.inv(self.true_hess)
        self.b = np.ones(n_dim)
        self.L = cond_number
        self.mu = 1
        self.optimum = np.linalg.solve(self.true_hess, self.b)
        self.f_opt = self.Eobjf(self.optimum)

    def objf(self, x, thetas):
        fun = []
        for theta in thetas:
            fun.append(.5 * x @ self.Q @ np.diag(
                self.D * theta) @ self.Q.T @ x - x @ self.b)
        return np.array(fun)

    def dobjf(self, x, thetas):
        grad = []
        for theta in thetas:
            grad.append(
                x @ self.Q @ np.diag(self.D * theta) @ self.Q.T - self.b)
        return np.vstack(grad)

    def Eobjf(self, x):
        return .5 * (x @ self.true_hess @ x) - self.b @ x

    def Edobjf(self, x):
        return x @ self.true_hess - self.b

    def sampler(self, n):
        return np.random.uniform(self.rand_param, 2 - self.rand_param,
                                 (int(n), self.n_dim))


if __name__ == '__main__':
    quad = QuadraticProblem(n_dim=10)
    sample_size = 100000
    sample = quad.sampler(sample_size)
    x = np.random.randn(10)
    objf_gap = quad.objf(x, sample).mean() - quad.Eobjf(x)
    print(f'Objective function gap: {objf_gap:.3}')
    cum_mean = np.cumsum(quad.objf(x, sample), axis=0) \
        / np.arange(1, sample_size + 1)
    plt.loglog(np.abs(cum_mean - quad.Eobjf(x)))
    plt.show()
    dobjf_gap = norm(quad.dobjf(x, sample).mean(axis=0) - quad.Edobjf(x))
    print(f'Gradient function gap: {dobjf_gap:.3}')
    cum_mean = np.cumsum(quad.dobjf(x, sample), axis=0) \
        / np.arange(1, sample_size + 1)[:,np.newaxis]
    plt.loglog(norm(cum_mean - quad.Edobjf(x), axis=1))
    plt.show()
