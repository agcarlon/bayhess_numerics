from time import time
from logreg import LogReg
import numpy as np
import matplotlib.pyplot as plt


def sgd(dataset, data_size, n_features, reg_param, label_format,
        epochs, clean_data=False, seed=0):
    print(f'Using SGD to train the {dataset} dataset')
    prob = LogReg(seed=seed, clean_data=clean_data, dataset=dataset,
                  reg_param=reg_param,
                  data_size=data_size,
                  n_features=n_features,
                  label_format=label_format)

    step_size = 1/prob.smooth

    w = np.zeros(prob.n_features)
    losses = [prob.loss_full(w)]
    opt_loss = prob.loss_full(prob.optimum)
    runtimes = [0.]
    k = 0
    t0 = time()

    # Code to compute Hessians computing time
    # data = []
    # for i in range(10000):
    #     ww = np.random.randn(prob.n_features)
    #     t0_ = time()
    #     prob.hessian(ww)
    #     data.append(time() - t0_)
    #     print((i, np.mean(data)))



    print(f'Cond. number: {prob.smooth / prob.reg_param}')
    print(f'Epoch {0}, iteration: {k}, loss: {losses[-1]}, '
          f'opt. gap: {losses[-1] - opt_loss}, time: {time() - t0:.2f}s')

    for epoch in range(epochs):
        for datum in prob.data:
            k += 1
            grad = prob.loss_grad(w, datum.reshape(1, -1))
            w = w - step_size/np.sqrt(k)*grad.mean(axis=0)
            # w = w - step_size*grad.mean(axis=0)
        losses.append(prob.loss_full(w))
        runtimes.append(time() - t0)
        print(f'Epoch {epoch}, iteration: {k}, loss: {losses[-1]}, '
              f'opt. gap: {losses[-1] - opt_loss}, time: {time() - t0:.2f}s')
        # np.save('Optimum.npy', w)
    opt_gap = np.array(losses) - opt_loss
    print(f'Finished using SGD to train the {dataset} dataset')
    print(f'Time spent: {runtimes[-1]:.2f}')

    np.save(f'{dataset}/sgd.npy', [opt_gap, runtimes])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(opt_gap, 'o-', ms=3., label='SGD')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Opt. gap')
    fig.tight_layout()
    fig.savefig(f'{dataset}/sgd_opt_gap.pdf')