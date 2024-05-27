from time import time

from logreg import LogReg
import numpy as np
import matplotlib.pyplot as plt
from mice import MICE, plot_mice


def sgd_mice(dataset, data_size, n_features, reg_param, label_format,
             epochs, clean_data=False, seed=0, mice_params={}):
    print(f'Using SGD-MICE to train the {dataset} dataset')
    prob = LogReg(seed=seed, clean_data=clean_data, dataset=dataset,
                  reg_param=reg_param,
                  data_size=data_size,
                  n_features=n_features,
                  label_format=label_format)

    df = MICE(prob.loss_grad, sampler=prob.data,
              max_cost=epochs * prob.data_size, **mice_params)

    step_size = 2 / (prob.smooth + prob.reg_param) / (1 + df.eps ** 2)
    # step_size = 1 / prob.smooth

    w = np.zeros(prob.n_features)
    losses = [prob.loss_full(w)]
    runtimes = [0.]
    opt_loss = prob.loss_full(prob.optimum)
    curr_epoch = [0]

    t0 = time()
    print(f'Cond. number: {prob.smooth / prob.reg_param}')
    print(f'Epoch {0}, iteration: {df.k}, loss: {losses[-1]}, '
          f'opt. gap: {losses[-1] - opt_loss}, time: {time() - t0:.2f}s')

    while True:
        grad = df(w)
        if df.terminate:
            break
        w = w - step_size * grad
        epoch = np.floor(df.counter / prob.data_size)
        if epoch > curr_epoch[-1]:
            curr_epoch.append(epoch)
            losses.append(prob.loss_full(w))
            runtimes.append(time() - t0)
            print(f'Epoch {epoch}, iteration: {df.k}, loss: {losses[-1]}, '
                  f'opt. gap: {losses[-1] - opt_loss}, '
                  f'time: {time() - t0:.2f}')
            if losses[-1] < opt_loss:
                np.save(f'{dataset}/Optimum.npy', w)
                opt_loss = losses[-1]
                print('OPTIMUM UPDATED, RECOMPUTE EVERYTHING!')
            # if epoch > 10:
            #     print(11)

    if epoch < epochs:
        epoch = np.round(df.counter / prob.data_size)
        curr_epoch.append(epoch)
        losses.append(prob.loss_full(w))
        runtimes.append(time() - t0)
        print(f'Epoch {epoch}, iteration: {df.k}, loss: {losses[-1]}, '
              f'opt. gap: {losses[-1] - opt_loss}, '
              f'time: {runtimes[-1]:.2f}')

    opt_gap = np.array(losses) - opt_loss
    print(f'Finished using SGD-MICE to train the {dataset} dataset')
    print(f'Time spent: {runtimes[-1]:.2f}')
    np.save(f'{dataset}/sgd_mice.npy', [opt_gap, curr_epoch, runtimes])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(curr_epoch, opt_gap, 'o-', ms=3., label='SGD-MICE')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Opt. gap')
    fig.tight_layout()
    fig.savefig(f'{dataset}/sgd_mice_opt_gap.pdf')

    log = df.get_log()
    log['epochs'] = log['num_grads'] / data_size

    fig, ax = plt.subplots(1, 1, sharex=True)
    ax = plot_mice(log, ax, x='epochs', y='hier_length', style='semilogy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Opt. gap')
    fig.savefig(f'{dataset}/sgd_mice.pdf')