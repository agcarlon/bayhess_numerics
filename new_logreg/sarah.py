from time import time

from logreg import LogReg
import numpy as np
import matplotlib.pyplot as plt
from methods.sarah import SARAH


def sarah(dataset, data_size, n_features, reg_param, label_format,
         step_param, batch_size,
         epochs, clean_data=False, seed=0):
    print(f'Using SARAH to train the {dataset} dataset')
    prob = LogReg(seed=seed, clean_data=clean_data, dataset=dataset,
                  reg_param=reg_param,
                  data_size=data_size,
                  n_features=n_features,
                  label_format=label_format)

    step_size = step_param / prob.almost_sure_smooth

    df = SARAH(prob.loss_grad,
              sample=prob.data,
              max_cost=epochs * prob.data_size,
              batchsize=batch_size,
              m=int(prob.data_size / batch_size * 2),
              verbose=False)

    w = np.zeros(prob.n_features)
    losses = [prob.loss_full(w)]
    runtimes = [0.]
    opt_loss = prob.loss_full(prob.optimum)

    kepoch = [1]

    k = 0
    t0 = time()
    print(f'Cond. number: {prob.smooth / prob.reg_param}')
    print(f'Epoch {0}, iteration: {k}, loss: {losses[-1]}, '
          f'opt. gap: {losses[-1] - opt_loss}, time: {time() - t0:.2f}s')
    epoch = 0

    while not df.force_exit:
        k += 1
        grad = df.evaluate(w, k)
        if df.force_exit:
            break
        w = w - step_size * grad
        if df.counter >= (epoch + 1) * prob.data_size:
            loss_k = prob.loss_full(w)
            losses.append(loss_k)
            runtimes.append(time() - t0)
            epoch = np.floor(df.counter / prob.data_size).astype('int')
            kepoch.append(k)
            print(f'Epoch {epoch}, iteration: {k}, loss: {losses[-1]}, '
                  f'opt. gap: {losses[-1] - opt_loss}, '
                  f'time: {time() - t0:.2f}')

    if len(losses) < epochs + 1:
        epoch = np.ceil(df.counter / prob.data_size)
        losses.append(prob.loss_full(w))
        runtimes.append(time() - t0)
    print(f'Epoch {epoch}, iteration: {k}, loss: {losses[-1]}, '
          f'opt. gap: {losses[-1] - opt_loss}, '
          f'time: {time() - t0:.2f}')

    opt_gap = np.array(losses) - opt_loss
    print(f'Finished using SARAH to train the {dataset} dataset')
    print(f'Time spent: {runtimes[-1]:.2f}')

    np.save(f'{dataset}/sarah.npy', [opt_gap, runtimes])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.semilogy(opt_gap, 'o-', ms=3., label='SARAH')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Opt. gap')
    fig.tight_layout()
    fig.savefig(f'{dataset}/sarah_opt_gap.pdf')
