import numpy as np
from mice import plot_mice
import matplotlib.pyplot as plt
import matplotlib


def make_plots(dataset, data_size):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))

    # SGD
    [opt_gap, runtimes] = np.load(f'{dataset}/sgd.npy', allow_pickle=True)
    print(f'SGD runtime: {runtimes[-1]:.2f}s')
    ax.semilogy(np.arange(len(opt_gap)) * data_size, opt_gap, '-', label='SGD', ms=3, c='C0')
    ax2.loglog(runtimes, opt_gap, '-', label='SGD', ms=3, c='C0')


    # SGD-MICE
    [opt_gap, curr_epoch, runtimes] = np.load(f'{dataset}/sgd_mice.npy', allow_pickle=True)
    print(f'SGD-MICE runtime: {runtimes[-1]:.2f}s')
    curr_epoch = np.asarray(curr_epoch)
    ax.semilogy(curr_epoch * data_size, opt_gap, '-', label='SGD-MICE', ms=3, c='C1')
    ax2.loglog(runtimes, opt_gap, '-', label='SGD-MICE', ms=3, c='C1')

    # SGD-MICE-Bay
    [opt_gap, update_iters, curr_epoch, err_mice, runtimes] = np.load(f'{dataset}/sgd_mice_bay.npy', allow_pickle=True)
    print(f'SGD-MICE-Bay runtime: {runtimes[-1]:.2f}s')
    curr_epoch = np.asarray(curr_epoch)
    update_iters = np.array(update_iters).astype('int')
    ax.semilogy(curr_epoch * data_size, opt_gap, '--', ms=3, label='SGD-MICE-Bay', c='C1')
    ax2.loglog(runtimes, opt_gap, '--', ms=3, label='SGD-MICE-Bay', c='C1')
    idxs = [np.where(curr_epoch == epch)[0][0] for epch in update_iters]
    ax.semilogy(update_iters * data_size, np.array(opt_gap)[idxs], 's', ms=3., c='C1', markeredgecolor='k')
    ax2.loglog(np.asarray(runtimes)[idxs], np.array(opt_gap)[idxs], 's', ms=3., c='C1', markeredgecolor='k')


    # # SGD-MICE-Bay-LR
    # [opt_gap, update_iters, curr_epoch, err_mice, runtimes] = np.load(f'{dataset}/sgd_mice_bay_lr.npy', allow_pickle=True)
    # print(f'SGD-MICE-Bay-LR runtime: {runtimes[-1]:.2f}s')
    # curr_epoch = np.asarray(curr_epoch)
    # update_iters = np.array(update_iters).astype('int')
    # ax.semilogy(curr_epoch * data_size, opt_gap, '-.', ms=3, label='SGD-MICE-Bay-LR', c='C1')
    # ax2.loglog(runtimes, opt_gap, '-.', ms=3, label='SGD-MICE-Bay-LR', c='C1')
    # idxs = [np.where(curr_epoch == epch)[0][0] for epch in update_iters]
    # ax.semilogy(update_iters * data_size, np.array(opt_gap)[idxs], 's', ms=3., c='C1', markeredgecolor='k')
    # ax2.loglog(np.asarray(runtimes)[idxs], np.array(opt_gap)[idxs], 's', ms=3., c='C1', markeredgecolor='k')

    # SVRG
    [opt_gap, runtimes] = np.load(f'{dataset}/svrg.npy', allow_pickle=True)
    print(f'SVRG runtime: {runtimes[-1]:.2f}s')
    ax.semilogy(np.arange(len(opt_gap)) * data_size, opt_gap, '-', label='SVRG', ms=3, c='C2')
    ax2.loglog(runtimes, opt_gap, '-', label='SVRG', ms=3, c='C2')

    # SVRG-Bay
    [opt_gap, update_iters, runtimes] = np.load(f'{dataset}/svrg_bay.npy', allow_pickle=True)[:3]
    print(f'SVRG-Bay runtime: {runtimes[-1]:.2f}s')
    update_iters = np.array(update_iters).astype('int')
    ax.semilogy(np.arange(len(opt_gap)) * data_size, opt_gap, '--', ms=3, label='SVRG-Bay', c='C2')
    ax.semilogy(update_iters * data_size, np.array(opt_gap)[update_iters], 's', ms=3., c='C2', markeredgecolor='k')
    ax2.loglog(runtimes, opt_gap, '--', ms=3, label='SVRG-Bay', c='C2')
    ax2.loglog(np.asarray(runtimes)[update_iters], np.array(opt_gap)[update_iters], 's', ms=3., c='C2', markeredgecolor='k')

    # # SVRG-New
    # [opt_gap, update_iters, runtimes] = np.load(f'{dataset}/svrg_new.npy', allow_pickle=True)[:3]
    # print(f'SVRG-New runtime: {runtimes[-1]:.2f}s')
    # update_iters = np.array(update_iters).astype('int')
    # ax.semilogy(np.arange(len(opt_gap)) * data_size, opt_gap, '--', ms=3, label='SVRG-New', c='C5')
    # ax.semilogy(update_iters * data_size, np.array(opt_gap)[update_iters], 's', ms=3., c='C5', markeredgecolor='k')
    # ax2.loglog(runtimes, opt_gap, '--', ms=3, label='SVRG-New', c='C5')
    # ax2.loglog(np.asarray(runtimes)[update_iters], np.array(opt_gap)[update_iters], 's', ms=3., c='C5', markeredgecolor='k')

    # # SAGA
    # [opt_gap, runtimes] = np.load(f'{dataset}/saga.npy', allow_pickle=True)
    # print(f'SAGA runtime: {runtimes[-1]:.2f}s')
    # ax.semilogy(np.arange(len(opt_gap)) * data_size, opt_gap, '-', label='SAGA', ms=3, c='C3')
    # ax2.loglog(runtimes, opt_gap, '-', label='SAGA', ms=3, c='C3')

    # SARAH
    [opt_gap, runtimes] = np.load(f'{dataset}/sarah.npy', allow_pickle=True)
    print(f'SARAH runtime: {runtimes[-1]:.2f}s')
    ax.semilogy(np.arange(len(opt_gap)) * data_size, opt_gap, '-', label='SARAH', ms=3, c='C3')
    ax2.loglog(runtimes, opt_gap, '-', label='SARAH', ms=3, c='C3')

    # SARAH-Bay
    [opt_gap, update_iters, runtimes] = np.load(f'{dataset}/sarah_bay.npy', allow_pickle=True)[:3]
    print(f'SARAH-Bay runtime: {runtimes[-1]:.2f}s')
    update_iters = np.array(update_iters).astype('int')
    ax.semilogy(np.arange(len(opt_gap)) * data_size, opt_gap, '--', ms=3, label='SARAH-Bay', c='C3')
    ax.semilogy(update_iters * data_size, np.array(opt_gap)[update_iters], 's', ms=3., c='C3', markeredgecolor='k')
    ax2.loglog(runtimes, opt_gap, '--', ms=3, label='SARAH-Bay', c='C3')
    ax2.loglog(np.asarray(runtimes)[update_iters], np.array(opt_gap)[update_iters], 's', ms=3., c='C3', markeredgecolor='k')

    # # Adam
    # [opt_gap, runtimes] = np.load(f'{dataset}/adam.npy', allow_pickle=True)
    # print(f'Adam runtime: {runtimes[-1]:.2f}s')
    # ax.semilogy(np.arange(len(opt_gap)) * data_size, opt_gap, '-', label='Adam', ms=3, c='C5')
    # ax2.loglog(runtimes, opt_gap, '-', label='Adam', ms=3, c='C5')

    lims = ax.get_ylim()[0]
    lims_ = 10**(np.floor(np.log10(lims)))
    ax.set_ylim(bottom=lims_)
    ax2.set_ylim(bottom=lims_)

    # ax.set_ylim(top=1.)
    # ax2.set_ylim(top=1.)


    ax.plot([], [], 'ws', label='Hessian update', markeredgecolor='k', ms=3.)
    ax.legend(ncol=2, prop={'size': 10})
    ax.set_title(fr'Logistic regression, \emph{{{dataset}}}  dataset')
    ax.set_xlabel(r'Number of gradient evaluations')
    ax.set_ylabel('Optimality gap')
    fig.tight_layout()
    fig.savefig(f'{dataset}/convergence.pdf')
    # plt.close('all')

    ax2.plot([], [], 'ws', label='Hessian update', markeredgecolor='k', ms=3.)
    ax2.legend(ncol=2, prop={'size': 10})
    ax2.set_title(fr'Logistic regression, \emph{{{dataset}}}  dataset')
    ax2.set_xlabel(r'Runtime (s)')
    ax2.set_ylabel('Optimality gap')
    fig2.tight_layout()
    fig2.savefig(f'{dataset}/convergence_per_time.pdf')
    plt.close('all')

    # err_svrg = np.load(f'{dataset}/svrg_bay.npy', allow_pickle=True)[3]
    # err_sarah = np.load(f'{dataset}/sarah_bay.npy', allow_pickle=True)[3]
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.plot(err_svrg, label='SVRG-Bay')
    # ax.plot(err_sarah, label='SARAH-Bay')
    # ax.plot(err_mice, label='SGD-MICE-Bay')
    # ax.legend()
    # fig.savefig(f'{dataset}/dennis_more.pdf')

    try:
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(5, 5))
        # SGD-MICE
        [opt_gap, curr_epoch, runtimes] = np.load(f'{dataset}/sgd_mice.npy', allow_pickle=True)
        print(f'SGD-MICE runtime: {runtimes[-1]:.2f}s')
        curr_epoch = np.asarray(curr_epoch)
        axs[0].semilogy(curr_epoch * data_size, opt_gap, '-', label='SGD-MICE', ms=3, c='C1')
        axs[1].loglog(runtimes, opt_gap, '-', label='SGD-MICE', ms=3, c='C1')

        # SGD-MICE-Bay
        [opt_gap, update_iters, curr_epoch, err_mice, runtimes] = np.load(f'{dataset}/sgd_mice_bay_wh.npy', allow_pickle=True)
        print(f'SGD-MICE-Bay with Wishart prior runtime: {runtimes[-1]:.2f}s')
        curr_epoch = np.asarray(curr_epoch)
        update_iters = np.array(update_iters).astype('int')
        axs[0].semilogy(curr_epoch * data_size, opt_gap, '--', ms=3, label='Wishart prior', c='C1')
        axs[1].loglog(runtimes, opt_gap, '--', ms=3, label='SGD-MICE-Bay', c='C1')
        idxs = [np.where(curr_epoch == epch)[0][0] for epch in update_iters]
        axs[0].semilogy(update_iters * data_size, np.array(opt_gap)[idxs], 's', ms=3., c='C1', markeredgecolor='k')
        axs[1].loglog(np.asarray(runtimes)[idxs], np.array(opt_gap)[idxs], 's', ms=3., c='C1', markeredgecolor='k')

        # SGD-MICE-Bay
        [opt_gap, update_iters, curr_epoch, err_mice, runtimes] = np.load(f'{dataset}/sgd_mice_bay.npy', allow_pickle=True)
        print(f'SGD-MICE-Bay with Frobenius norm prior runtime: {runtimes[-1]:.2f}s')
        curr_epoch = np.asarray(curr_epoch)
        update_iters = np.array(update_iters).astype('int')
        axs[0].semilogy(curr_epoch * data_size, opt_gap, '--', ms=3, label='Frobenius norm prior', c='C2')
        axs[1].loglog(runtimes, opt_gap, '--', ms=3, label='SGD-MICE-Bay', c='C2')
        idxs = [np.where(curr_epoch == epch)[0][0] for epch in update_iters]
        axs[0].semilogy(update_iters * data_size, np.array(opt_gap)[idxs], 's', ms=3., c='C2', markeredgecolor='k')
        axs[1].loglog(np.asarray(runtimes)[idxs], np.array(opt_gap)[idxs], 's', ms=3., c='C2', markeredgecolor='k')
        axs[0].legend()
        fig.savefig(f'{dataset}/wishart_frob.pdf')
    except:
        pass