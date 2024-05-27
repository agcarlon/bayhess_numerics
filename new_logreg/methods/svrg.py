# SAGA algorithm
import numpy as np
from functools import partial
from itertools import cycle
from IPython.core.debugger import set_trace


class SVRG():
    def __init__(self,
                 func,
                 sample,
                 batchsize=100,
                 max_cost=1000,
                 m=10000,
                 verbose=False):
        self.func = func
        self.sample = sample
        self.batchsize = batchsize
        self.max_cost = max_cost
        self.verbose = verbose
        self.counter = 0
        self.m = m
        self.average = []
        self.norm = np.linalg.norm
        self.datasize = len(sample)
        self.sampler = Sampler(np.arange(self.datasize), 0)
        self.log = []
        self.x_ = None
        self.grad_mem = None
        self.sk = []
        self.yk = []
        self.force_exit = False
        self.update_hess = False

    def evaluate(self, x, k):
        if self.verbose:
            print(f'Evaluating SVRG')
        if self.average == [] or not (k % self.m):
            print('SVRG restart')
            # self.grad_mem = np.zeros((len(self.sample), *x.shape))
            # for idx, samp in enumerate(self.sample):
            #     self.grad_mem[idx] = self.func(x, samp)
            self.grad_mem = self.func(x, self.sample)
            self.average = np.nanmean(self.grad_mem, axis=0)
            self.x_ = x
            self.counter += len(self.sample)
            self.log.append(self.counter)
            self.update_hess = False
            return self.average
        sample_idxs = self.sampler(self.batchsize)
        # deltas = []
        # for sample_idx in sample_idxs:
        #     deltas.append(self.func(x, self.sample[sample_idx])
        #                   - self.grad_mem[sample_idx])
        # sample = [self.sample[sample_idx] for sample_idx in sample_idxs]
        sample = self.sample[sample_idxs]
        deltas = self.func(x, sample) - self.grad_mem[sample_idxs]
        self.sk.append(x - self.x_)
        self.yk.append(deltas)
        self.update_hess = True
        estim = self.average + np.mean(deltas, axis=0)
        self.counter += self.batchsize
        self.log.append(self.counter)
        self.check_max_cost()
        return estim

    def check_max_cost(self):
        if self.counter > self.max_cost:
            self.force_exit = True
            return True
        else:
            return False

    def create_sampler(self):
        sample_idx = np.arange(len(self.sample))
        # np.random.shuffle(sample_idx)
        sample_iterator = cycle(sample_idx)
        return partial(sampler_finite, sample_iterator=sample_iterator)


def sampler_finite(n, sample_iterator):
    return [sample_idx for i, sample_idx in zip(range(n), sample_iterator)]


class Sampler:
    def __init__(self, data, start):
        self.data = data
        self.data_size = len(data)
        self.start = start
        self.counter = 0

    def __call__(self, n):
        idxs = np.mod(np.arange(self.start+self.counter,
                                self.start+self.counter+n), self.data_size)
        self.counter += n
        return self.data[idxs]