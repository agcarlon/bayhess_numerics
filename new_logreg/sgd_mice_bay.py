from time import time

from logreg import LogReg
import numpy as np
from numpy.linalg import solve, eigvalsh, norm
import matplotlib.pyplot as plt
from mice import MICE, plot_mice
from bayhess import BayHess


def sgd_mice_bay(dataset, data_size, n_features, reg_param, label_format,
                 update_when,
                 epochs, clean_data=False, seed=0,
                 mice_params={},
                 bay_params={}):
    print(f'Using SGD-MICE-Bay to train the {dataset} dataset')
    prob = LogReg(seed=seed, clean_data=clean_data, dataset=dataset,
                  reg_param=reg_param,
                  data_size=data_size,
                  n_features=n_features,
                  label_format=label_format)

    df = MICE(prob.loss_grad, sampler=prob.data,
              max_cost=epochs * prob.data_size, **mice_params)

    bay = BayHess(n_dim=prob.n_features, strong_conv=prob.reg_param,
                  smooth=prob.smooth, log=f'{dataset}/sgd_mice_bay.txt',
                  **bay_params)

    # hess = np.eye(prob.n_features) * (1 / step_size)

    bay.hess = (prob.smooth+prob.reg_param)/2 * np.eye(prob.n_features)
    bay.inv_hess = 2/(prob.smooth+prob.reg_param) * np.eye(prob.n_features)

    w = np.zeros(prob.n_features)
    # w = prob.optimum
    losses = [prob.loss_full(w)]
    runtimes = [0.]
    opt_loss = prob.loss_full(prob.optimum)

    last_update = 0
    update_iters = []
    op_err_norm = []
    curr_epoch = [0]
    len_sks = [0]
    hessians = []

    t0 = time()
    print(f'Cond. number: {prob.smooth / prob.reg_param}')
    print(f'Epoch {0}, iteration: {df.k}, loss: {losses[-1]}, '
          f'opt. gap: {losses[-1] - opt_loss}, time: {time() - t0:.2f}s')

    while True:
        grad = df(w)
        if df.terminate:
            break
        bay.update_curv_pairs_mice(df)
        len_sks.append(len(bay.sk_all))
        epoch = np.floor(df.counter / prob.data_size)
        # if epoch > last_update + update_when and len(bay.sk_all) >= prob.n_features:
        if last_update + update_when <= epoch < epochs:
            bay.print(f'SGD-MICE-Bay iteration: {df.k}')
            last_update += update_when
            # Sk = np.vstack(bay.sk_all)
            # svd = np.linalg.svd(Sk)
            # cond = np.linalg.cond(Sk)
            # print(cond)
            t0_ = time()
            bay.find_hess()
            bay.inv_hess = np.linalg.inv(bay.hess)
            bay.print(f'Time spent in Bay-Hessian: {time()-t0_:.2f}')
            # tr_hess = prob.hessian(w)
            # op_err_norm.append(np.max(np.abs(eigvalsh(hess - tr_hess))))
            update_iters.append(epoch)
            hessians.append((bay.hess, w, 1 / (1 + df.eps ** 2) * bay.inv_hess @ grad))
            print(f'min eig of Hess: {np.min(eigvalsh(bay.hess))}')
        w = w - 1 / (1 + df.eps ** 2) * bay.inv_hess @ grad
        # w = w - .5 * bay.hess @ grad
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

    if epoch < epochs:
        epoch = np.ceil(df.counter / prob.data_size)
        curr_epoch.append(epoch)
        losses.append(prob.loss_full(w))
        runtimes.append(time() - t0)
        print(f'Epoch {epoch}, iteration: {df.k}, loss: {losses[-1]}, '
              f'opt. gap: {losses[-1] - opt_loss}, '
              f'time: {time() - t0:.2f}')

    opt_gap = np.array(losses) - opt_loss
    print(f'Finished using SGD-MICE-Bay to train the {dataset} dataset')
    print(f'Time spent: {runtimes[-1]:.2f}')

        # Computing Dennis-More errors
    err = []
    # for hess, w, dw in hessians:
    #     tr_hess = prob.hessian(w)
    #     err.append(norm((hess - tr_hess) @ dw) / norm(dw))

    np.save(f'{dataset}/sgd_mice_bay.npy', [opt_gap, update_iters, curr_epoch,
                                            err, runtimes])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(curr_epoch, opt_gap, 'o-', ms=3., label='SGD-MICE-Bay')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Opt. gap')
    fig.tight_layout()
    fig.savefig(f'{dataset}/sgd_mice_bay_opt_gap.pdf')

    log = df.get_log()
    log['len_sks'] = len_sks
    log['epochs'] = log['num_grads'] / data_size

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0] = plot_mice(log, axs[0], x='epochs', y='hier_length', style='semilogy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Opt. gap')
    axs[1] = plot_mice(log, axs[1], x='epochs', y='len_sks', style='plot')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Num. of curv. pairs')
    fig.savefig(f'{dataset}/sgd_mice_bay.pdf')
