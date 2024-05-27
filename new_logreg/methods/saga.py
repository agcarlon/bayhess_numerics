# SAGA algorithm
import numpy as np
from functools import partial
from itertools import cycle
from IPython.core.debugger import set_trace


class SAGA():
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
        self.counter = 0
        self.norm = np.linalg.norm
        self.datasize = len(sample)
        self.log = []
        self.force_exit = False

    def evaluate(self, x):
        if self.verbose:
            print(f'Evaluating SAGA')
        if self.grad_mem == []:
            # self.grad_mem = np.zeros((len(self.sample), *x.shape))
            # for idx, samp in enumerate(self.sample):
            #     self.grad_mem[idx] = self.func(x, samp)
            self.grad_mem = self.func(x, self.sample)
            self.average = np.nanmean(self.grad_mem, axis=0)
            self.counter = len(self.sample)
            self.log.append(self.counter)
            return self.average
            # self.grad_mem *= np.nan
        # estim_ = np.nanmean(self.grad_mem, axis=0)
        sample_idxs = self.sampler(self.batchsize)
        deltas = []
        # update_idxs = []
        # for sample_idx in sample_idxs:
        #     # if np.isnan(self.grad_mem[sample_idx]).all():
        #     #     self.grad_mem[sample_idx] = self.func(x, self.sample[sample_idx])
        #     # else:
        #     deltas.append(self.func(x, self.sample[sample_idx])
        #                   - self.grad_mem[sample_idx])
        #     self.grad_mem[sample_idx] += deltas[-1]
            # update_idxs.append(sample_idx)
        sample = [self.sample[sample_idx] for sample_idx in sample_idxs]
        deltas = self.func(x, sample) - self.grad_mem[sample_idxs]
        # estim = np.nanmean(self.grad_mem, axis=0)
        estim = self.average + np.mean(deltas, axis=0)
        # D = np.zeros(len(x))
        # D[np.where(deltas)[1]] = self.D[np.where(deltas)[1]]
        # estim = self.average*D + np.mean(deltas, axis=0)
        self.grad_mem[sample_idxs] += deltas
        self.average += np.sum(deltas, axis=0)/self.datasize
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
