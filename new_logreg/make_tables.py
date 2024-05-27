import numpy as np
from mice import plot_mice
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


def make_tables(dataset):
    opt_gaps, runtimes, names = [], [], ['SGD', 'SVRG', 'SAGA', 'Adam', 'SGD-MICE', 'SGD-MICE-Bay', 'SVRG-Bay']
    [opt_gap, runtime] = np.load(f'{dataset}/sgd.npy', allow_pickle=True)
    opt_gaps.append(opt_gap)
    runtimes.append(np.asarray(runtime))
    [opt_gap, runtime] = np.load(f'{dataset}/svrg.npy', allow_pickle=True)
    opt_gaps.append(opt_gap)
    runtimes.append(np.asarray(runtime))
    [opt_gap, runtime] = np.load(f'{dataset}/saga.npy', allow_pickle=True)
    opt_gaps.append(opt_gap)
    runtimes.append(np.asarray(runtime))
    [opt_gap, runtime] = np.load(f'{dataset}/adam.npy', allow_pickle=True)
    opt_gaps.append(opt_gap)
    runtimes.append(np.asarray(runtime))
    [opt_gap, curr_epoch, runtime] = np.load(f'{dataset}/sgd_mice.npy', allow_pickle=True)
    opt_gaps.append(opt_gap)
    runtimes.append(np.asarray(runtime))
    [opt_gap, update_iters, curr_epoch, err_mice, runtime] = np.load(f'{dataset}/sgd_mice_bay.npy', allow_pickle=True)
    opt_gaps.append(opt_gap)
    runtimes.append(np.asarray(runtime))
    [opt_gap, update_iters, runtime] = np.load(f'{dataset}/svrg_bay.npy', allow_pickle=True)[:3]
    opt_gaps.append(opt_gap)
    runtimes.append(np.asarray(runtime))
    max = np.max([np.max(opt_gap) for opt_gap in opt_gaps])
    min = np.min([np.min(opt_gap) for opt_gap in opt_gaps])
    max_10 = 10**np.floor(np.log10(max))
    min_10 = 10**np.ceil(np.log10(min))
    base = 10
    tols = np.arange(np.floor(np.log10(max)), np.floor(np.log10(min)), -1)
    if len(tols) < 2:
        base = 2
        tols = np.arange(np.floor(np.log2(max)), np.floor(np.log2(min)), -1)
    cols = [fr'{base}^{tol}' for tol in tols]
    data = pd.DataFrame(columns=cols, index=names)
    for opt_gap, runtime, name in zip(opt_gaps, runtimes, names):
        print(name)
        for col, tol in zip(cols, tols):
            tol_ = base**tol
            if len(runtime[opt_gap < tol_]):
                data[col][name] = runtime[opt_gap < tol_][0]
            else:
                data[col][name] = '-'
    print(dataset)
    print(data)
    print(data.to_latex(float_format="{:0.2f}".format))
    1


if __name__ == '__main__':
    make_tables('mushrooms')
    make_tables('cod-rna')
    make_tables('ijcnn1')
    make_tables('w8a')
    make_tables('HIGGS')