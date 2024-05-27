import numpy as np
from IPython.core.debugger import set_trace


class BFGS():
    def __init__(self, verb=False, search_dir='lbfgs', ls='wolfe_armijo', dim=0):
        self.search_dir = getattr(self, search_dir)
        self.linesearch = getattr(self, ls)
        self.exit_flag = False
        self.dim = dim
        if verb:
            self.print = lambda str: print(str)
        else:
            self.print = lambda std: None
        self.cost = {
            'objf': 0,
            'grads': 0
        }
        self.sk = []
        self.yk = []
        self.H = []
        self.B = []

    def run(self, fun, grad_fun, x0, step, iters=1000, m=100, tol=1e-10):
        self.dim = len(x0)
        self.m = m
        X = [x0]
        grad = [grad_fun(X[0])]
        self.cost['grads'] += 1
        X.append(self.linesearch(X[0], -grad[-1], fun, grad_fun, step))
        for k in range(iters):
            grad.append(grad_fun(X[-1]))
            self.cost['grads'] += 1
            sk = X[-1] - X[-2]
            yk = grad[-1] - grad[-2]
            if sk @ yk / (sk @ sk) > 1e-2:
                self.sk.append(sk)
                self.yk.append(yk)
            d = self.search_dir(grad[-1])
            self.print(f'cos d and grad:{cos_vec(d, grad[-1])}')
            X_cand = self.linesearch(X[-1], d, fun, grad_fun, 1.)
            if not self.exit_flag:
                X.append(X_cand)
            if np.linalg.norm(grad[-1]) < tol or self.exit_flag:
                break
        tr_hess_post = jac(grad_fun, X[-1], dx=1e-5)
        inv_hess = np.linalg.inv(tr_hess_post)
        inv_appr = self.H[-1]
        # set_trace()
        return X[-1]

    def bfgs(self, grad):
        # self.H.append(self.hess_inv([self.sk[-1]], [self.yk[-1]], self.H[-1]))
        self.H.append(self.hess_inv(self.sk[-self.m:], self.yk[-self.m:], np.eye(self.dim)))
        return -self.H[-1] @ grad

    def hess_inv(self, sk, yk, H):
        for y, s in zip(yk, sk):
            if y @ s > 1e-10 * s @ s:
                rho = 1/(y @ s)
                H = ((np.eye(self.dim) - np.outer(s, y)*rho) @ H @
                     (np.eye(self.dim) - np.outer(y, s)*rho)) + np.outer(s, s)*rho
            else:
                self.print('bad pair')
        return H

    def hess(self, sk, yk, B):
        B_ = B.copy()
        for y, s in zip(yk, sk):
            if y @ s > 1e-10 * s @ s:
                alpha = 1/ (s @ y)
                beta = - 1 / (s @ B @ s)
                B_ = B_ + alpha*np.outer(y, y) + beta*np.outer(B_ @ s, B_ @ s)
            else:
                self.print('bad pair')
        return B_


    def lbfgs(self, grad):
        m = np.minimum(self.m, len(self.sk))
        sk = self.sk[-m:]
        yk = self.yk[-m:]
        q = grad.copy()
        alpha = np.zeros(m)
        beta = np.zeros(m)
        rho = np.zeros(m)
        for i in range(m-1, -1, -1):
            rho[i] = (sk[i] @ yk[i])**-1
            alpha[i] = rho[i]*(sk[i] @ q)
            q = q - alpha[i]*yk[i]
        gamma_k = (sk[-1] @ yk[-1])/(yk[-1] @ yk[-1])
        z = gamma_k * q
        for i in range(0, m):
            beta[i] = rho[i] * (yk[i] @ z)
            z = z + sk[i]*(alpha[i] - beta[i])
        z = -z
        return z

    def wolfe_armijo(self, x, d, fun, grad_fun, step):
        grad0 = grad_fun(x)
        f0 = fun(x)
        self.cost['grads'] += 1
        self.cost['objf'] += 1
        step_min = 1e-20
        step_max = np.nan

        c1 = 0.001
        c2 = 0.9
        for t in range(100):
            x1 = x + step*d
            grad1 = grad_fun(x1)
            f1 = fun(x1)
            self.cost['grads'] += 1
            self.cost['objf'] += 1
            wolfe = -grad1 @ d < -c2*grad0 @ d
            armijo = f1 < f0 + c1*step*(grad0 @ d)
            self.print(f'{wolfe}, {armijo}')
            if wolfe and armijo:
                self.print(f'Accepted step: {step}. Decreased objf from {f0} to {f1}')
                return x1
            elif (not wolfe) and (not armijo):
                self.exit_flag = True
                self.print('Error')
                return False
            elif (not wolfe) and armijo:
                step_min = np.copy(step)
                if np.isnan(step_max):
                    step = 2*step
                else:
                    step = (step_min + step_max)/2
                self.print(f'Increase to {step}')
            elif wolfe and (not armijo):
                step_max = np.copy(step)
                self.print(f'Decrease to {step}')
                step = (step_min + step_max)/2
        self.exit_flag = True
        self.print('100 steps')
        return False

    def backtrack_armijo(self, x, d, fun, grad_fun, step):
        grad0 = grad_fun(x)
        f0 = fun(x)
        self.cost['grads'] += 1
        self.cost['objf'] += 1
        # step_min = 1e-20
        c1 = 0.001
        inner = np.sum(grad0 * d)
        for t in range(100):
            x1 = x + step*d
            f1 = fun(x1)
            self.cost['objf'] += 1
            armijo = f1 < f0 + c1*step*inner
            self.print(f'{armijo}')
            if armijo:
                self.print(f'Accepted step: {step}. Decreased objf from {f0} to {f1}')
                return x1
            else:
                step /= 2
                self.print(f'Decrease to {step}')
        self.exit_flag = True
        self.print('100 steps')
        # set_trace()
        return False


def cos_vec(x, y):
    return (x @ y) / (x @ x)**.5 / (y @ y)**.5


def jac(fun, x, dx=1e-4):
    f0 = fun(x)
    dim_f = len(f0)
    dim_x = len(x)
    jac_f = np.zeros((dim_f, dim_x))
    for i in range(dim_x):
        x_ = np.copy(x)
        x_[i] += dx
        f_ = fun(x_)
        jac_f[i] = (f_ - f0)/dx
    return (jac_f + jac_f.T)/2
