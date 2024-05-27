from time import time

from logreg import LogReg
import numpy as np
from numpy.linalg import solve, eigvalsh, norm
import matplotlib.pyplot as plt
from methods.svrg import SVRG
from bayhess import BayHess


def svrg_bay(dataset, data_size, n_features, reg_param, label_format,
             step_param, batch_size, epochs, update_when, clean_data=False,
             seed=0, bay_params={}):
    print(f'Using SVRG-Bay to train the {dataset} dataset')
    prob = LogReg(seed=seed, clean_data=clean_data, dataset=dataset,
                  reg_param=reg_param,
                  data_size=data_size,
                  n_features=n_features,
                  label_format=label_format)

    df = SVRG(prob.loss_grad,
              sample=prob.data,
              max_cost=epochs * prob.data_size,
              batchsize=batch_size,
              m=int(prob.data_size / batch_size * 2),
              verbose=False)

    bay = BayHess(n_dim=prob.n_features, strong_conv=prob.reg_param,
                  smooth=prob.almost_sure_smooth, log=f'{dataset}/svrg_bay.txt',
                  **bay_params)

    bay.hess = prob.almost_sure_smooth * np.eye(prob.n_features)
    bay.inv_hess = 1/prob.almost_sure_smooth * np.eye(prob.n_features)

    w = np.zeros(prob.n_features)
    losses = [prob.loss_full(w)]
    runtimes = [0.]
    opt_loss = prob.loss_full(prob.optimum)

    k = 0
    t0 = time()
    print(f'Cond. number: {prob.smooth / prob.reg_param}')
    print(f'Epoch {0}, iteration: {k}, loss: {losses[-1]}, '
          f'opt. gap: {losses[-1] - opt_loss}, time: {time() - t0:.2f}s')
    epoch = 0

    last_update = 0
    update_iters = []
    hessians = []

    while not df.force_exit:
        k += 1
        grad = df.evaluate(w, k)
        if df.force_exit:
            break
        if df.update_hess:
            bay.update_curv_pairs(df.sk[-1], df.yk[-1])
        if last_update + update_when <= epoch < epochs:
            bay.print(f'SVRG-Bay iteration: {k}')
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
            hessians.append((bay.hess, w, step_param*bay.inv_hess @ grad))
            print(f'min eig of Hess: {np.min(eigvalsh(bay.hess))}')
        w = w - step_param*bay.inv_hess @ grad
        if df.counter >= (epoch+1)*prob.data_size:
            loss_k = prob.loss_full(w)
            losses.append(loss_k)
            runtimes.append(time() - t0)
            epoch = np.floor(df.counter / prob.data_size).astype('int')
            print(f'Epoch {epoch}, iteration: {k}, loss: {losses[-1]}, '
                  f'opt. gap: {losses[-1] - opt_loss}, '
                  f'time: {time() - t0:.2f}')
            if losses[-1] < opt_loss:
                np.save(f'{dataset}/Optimum.npy', w)
                opt_loss = losses[-1]
                print('OPTIMUM UPDATED, RECOMPUTE EVERYTHING!')

    if len(losses) < epochs + 1:
        epoch = np.ceil(df.counter / prob.data_size)
        losses.append(prob.loss_full(w))
        runtimes.append(time() - t0)
    print(f'Epoch {epoch}, iteration: {k}, loss: {losses[-1]}, '
          f'opt. gap: {losses[-1] - opt_loss}, '
          f'time: {time() - t0:.2f}')

    opt_gap = np.array(losses) - opt_loss
    print(f'Finished using SVRG-Bay to train the {dataset} dataset')
    print(f'Time spent: {runtimes[-1]:.2f}')

    # Computing Dennis-More errors
    err = []
    for hess, w, dw in hessians:
        tr_hess = prob.hessian(w)
        err.append(norm((hess - tr_hess) @ dw) / norm(dw))

    np.save(f'{dataset}/svrg_bay.npy', [opt_gap, update_iters, runtimes, err])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(opt_gap, 'o-', ms=3., label='SVRG-Bay')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Opt. gap')
    fig.tight_layout()
    fig.savefig(f'{dataset}/svrg_bay_opt_gap.pdf')


