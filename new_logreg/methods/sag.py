# SAGA algorithm
import numpy as np
from functools import partial
from itertools import cycle
from IPython.core.debugger import set_trace


class SAG():
    def __init__(self,
                 func,
                 sample,
                 batchsize=100,
                 max_cost=1000,
                 verbose=False):
        self.func = func
        self.sample = sample
        self.batchsize = batchsize
        self.max_cost = max_cost
        self.verbose = verbose
        self.grad_mem = []
        self.sampler = self.create_sampler()
        self.norm = np.linalg.norm
        self.datasize = len(sample)
        self.counter = 0
        self.log = []
        self.force_exit = False

    # def evaluate(self, x):
    #     if self.verbose:
    #         print(f'Evaluating SAG')
    #     if self.grad_mem == []:
    #         self.grad_mem = np.zeros((len(self.sample), *x.shape))
    #         self.grad_mem *= np.nan
    #     sample_idxs = self.sampler(self.batchsize)
    #     # if np.isnan(self.grad_mem[sample_idx]).all():
    #     for sample_idx in sample_idxs:
    #         self.grad_mem[sample_idx] = self.func(x, self.sample[sample_idx])
    #     estim = np.nanmean(self.grad_mem, axis=0)
    #     self.counter += self.batchsize
    #     self.log.append(self.counter)
    #     self.check_max_cost()
    #     return estim

    def evaluate(self, x):
        if self.verbose:
            print(f'Evaluating SAG')
        if self.grad_mem == []:
            # self.grad_mem = np.zeros((len(self.sample), *x.shape))
            # for idx, samp in enumerate(self.sample):
            #     self.grad_mem[idx] = self.func(x, samp)
            self.grad_mem = self.func(x, self.sample)
            self.average = np.nanmean(self.grad_mem, axis=0)
            self.counter = len(self.sample)
            self.log.append(self.counter)
            return self.average
        sample_idxs = self.sampler(self.batchsize)
        # deltas = []
        # for sample_idx in sample_idxs:
        #     deltas.append(self.func(x, self.sample[sample_idx])
        #                   - self.grad_mem[sample_idx])
        #     self.grad_mem[sample_idx] += deltas[-1]
        sample = [self.sample[sample_idx] for sample_idx in sample_idxs]
        deltas = self.func(x, sample) - self.grad_mem[sample_idxs]
        self.average += np.sum(deltas, axis=0)/self.datasize
        self.grad_mem[sample_idxs] += deltas
        # print((np.mean(self.grad_mem, axis=0) - self.average).max())
        self.counter += self.batchsize
        self.log.append(self.counter)
        self.check_max_cost()
        return self.average

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
