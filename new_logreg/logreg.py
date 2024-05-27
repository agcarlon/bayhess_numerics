from functools import partial

import numpy as np
from numpy.linalg import norm, eig, solve
# from multiprocessing import Pool
from joblib import Parallel, delayed


def load_data(filename, data_size, n_features):
    X = np.zeros((data_size, n_features))
    Y = np.zeros((data_size,))

    with open(filename) as f:
        for i, line in enumerate(f):
            if i == data_size:
                break
            data = line.split()
            Y[i] = int(data[0])  # target value
            for item in data[1:]:
                j, k = item.split(':')
                # set_trace()
                X[i, int(j) - 1] = float(k)
    return X, Y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogReg:
    def __init__(self, dataset='cod-rna', data_size=59535, n_features=8,
                 reg_param=1e-5, seed=0, clean_data=False, label_format='-1 1'):
        self.seed = seed
        np.random.seed(seed)
        self.dataset = dataset
        self.data_size = data_size
        self.n_features = n_features
        if dataset == 'HIGGS':
            data = np.genfromtxt('HIGGS/HIGGS', delimiter=',')
            self.x = np.vstack(data[:, 1:])
            self.y = np.stack(data[:, 0])
        else:
            data = load_data(f'{dataset}/{dataset}', data_size, n_features)
            self.x = data[0]
            self.y = data[1]
        # self.pool = Pool(78)
        if label_format == '1 2':
            self.y = self.y * 2 - 3
        elif label_format == '0 1':
            self.y = self.y * 2 - 1
        elif label_format == '-1 1':
            pass
        try:
            self.optimum = np.load(f'{dataset}/Optimum.npy')
        except:
            print('Creating Optimum.npy with zeros')
            self.optimum = np.zeros(n_features)
            np.save(f'{dataset}/Optimum.npy', self.optimum)
        self.clean_data = clean_data
        if clean_data:
            self.do_clean_data()
            self.optimum = np.load(f'{dataset}/Optimum_clean.npy')
        # self.data = [*zip(self.x, self.y)]
        self.data = np.hstack([self.y.reshape((-1, 1)), self.x])
        # np.random.shuffle(self.data)
        np.random.shuffle(self.data)
        self.smooth = 0.25 * np.mean((self.x ** 2).sum(axis=1)) + reg_param
        self.almost_sure_smooth = 0.25 * np.max((self.x ** 2).sum(axis=1)) + reg_param
        self.hess_smooth = 3 ** .5 / 18 * np.mean(np.linalg.norm(self.x, axis=1) ** 3)
        self.reg_param = reg_param

    def loss_full(self, w):
        ls = (np.log(
            1 + np.exp(self.y * (self.x @ w)))) + .5 * self.reg_param * (
                         w @ w)
        return np.mean(ls)

    def loss_grad_full(self, w):
        gr3 = (sigmoid(self.y * (
                    self.x @ w)) * self.y) @ self.x / self.data_size \
              + self.reg_param * w
        return gr3

    def loss(self, w, thetas):
        ls = np.log(1 + np.exp(thetas[:, 0] * (thetas[:, 1:] @ w))) + .5 * self.reg_param * (
                    w @ w)
        return np.array(ls)

    def loss_grad(self, w, theta):
        gr2 = np.tile(sigmoid(theta[:, 0] * (theta[:, 1:] @ w)) * theta[:, 0],
                      [self.n_features, 1]).T * theta[:, 1:] + self.reg_param \
                                            * np.tile(w, [len(theta[:, 1:]), 1])
        return gr2
    # def loss_grad(self, w, theta):
    #     X_, Y_ = zip(*theta)
    #     gr2 = np.tile(sigmoid(Y_ * (X_ @ w)) * Y_,
    #                   [self.n_features, 1]).T * X_ + self.reg_param \
    #                                         * np.tile(w, [len(X_), 1])
    #     return gr2
    #
    # def loss_grad2(self, w, theta):
    #     gr3 = np.zeros((len(theta), self.n_features))
    #     for i, the in enumerate(theta):
    #         gr3[i] = sigmoid(the[1] * (the[0] @ w)) * \
    #             the[0] * the[1] + self.reg_param * w
    #     return gr3



    # def loss_grad_(self, w, theta):
    #     gr = (sigmoid(theta[1] * (
    #                 theta[0] @ w)) * theta[1]) @ theta[0] \
    #           + self.reg_param * w
    #     return gr
    #
    # def loss_grad_par(self, w, thetas):
    #     jobs = 78
    #     splits = np.array_split(thetas, jobs)
    #     # grads = Parallel(n_jobs=78)(delayed(loss_grad2)(w, theta, self.reg_param) for theta in splits)
    #     grads = self.pool.map(partial(loss_grad, w=w, reg_param=self.reg_param), splits)
    #     return np.vstack(grads)
    #
    # def loss_grad_par2(self, w, thetas):
    #     jobs = 78
    #     splits = np.array_split(thetas, jobs)
    #     with Parallel(n_jobs=jobs) as par:
    #         grads = par(delayed(partial(loss_grad, w=w, reg_param=self.reg_param))(split) for split in splits)
    #     return np.vstack(grads)

    def hessian(self, w):
        hess = np.zeros((self.n_features, self.n_features))
        for x, y in zip(self.x, self.y):
            z = y * (x @ w)
            hess += sigmoid(z) * (1 - sigmoid(z)) * np.outer(x, x)
        hess /= self.data_size
        hess += self.reg_param * np.eye(self.n_features)
        return hess

    def accuracy(self, w):
        p_true = sigmoid(-self.x @ w)
        p_false = sigmoid(self.x @ w)
        P = p_true > p_false
        acc = np.mean((P * 2 - 1) == self.y)
        return acc

    def do_clean_data(self):
        q, r = np.linalg.qr(self.x)
        mask = np.abs(np.diag(r)) > 1e-10
        self.x = self.x[:, mask]
        self.optimum = self.optimum[mask]
        self.n_features = int(np.sum(mask))


def loss_grad2(theta, w, reg_param):
    gr3 = np.zeros((len(theta), len(w)))
    for i, the in enumerate(theta):
        gr3[i] = sigmoid(the[1] * (the[0] @ w)) * \
            the[0] * the[1] + reg_param * w
    return gr3


def loss_grad(theta, w, reg_param):
    X_, Y_ = zip(*theta)
    gr2 = np.tile(sigmoid(Y_ * (X_ @ w)) * Y_,
                  [len(w), 1]).T * X_ + reg_param \
                                        * np.tile(w, [len(X_), 1])
    return gr2

if __name__ == "__main__":
    prob = LogReg()
    from time import perf_counter

    t0 = perf_counter()
    for i in range(100):
        prob.loss_grad(prob.optimum, prob.data)
    print(f"Time to run loss_grad: {(perf_counter() - t0) / 100}")

    # t0 = perf_counter()
    # for i in range(100):
    #     prob.loss_grad2(prob.optimum, prob.data)
    # print(f"Time to run loss_grad2: {(perf_counter() - t0) / 100}")

    t0 = perf_counter()
    for i in range(100):
        prob.loss_grad3(prob.optimum, prob.data_)
    print(f"Time to run loss_grad3: {(perf_counter() - t0) / 100}")

    # t0 = perf_counter()
    # for i in range(100):
    #     prob.loss_grad_(prob.optimum, prob.data)
    # print(f"Time to run loss_grad_: {(perf_counter() - t0) / 100}")

    # t0 = perf_counter()
    # for i in range(100):
    #     prob.loss_grad_par2(prob.optimum, prob.data)
    # print(f"Time to run loss_grad_par2: {(perf_counter() - t0) / 100}")

