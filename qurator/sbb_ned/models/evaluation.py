import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_lr(target_lr, n_epochs, train_set_size, batch_size, warmup):
    total = (n_epochs - 1) * int(np.ceil(train_set_size / batch_size))
    progress = [float(t) / total for t in range(0, total)]

    factor = [p / warmup if p < warmup else max((p - 1.) / (warmup - 1.), 0.) for p in progress]

    lr = [f * target_lr for f in factor]

    return lr


def load_train_log(directories, num_epochs, target_lr, **kwargs):
    parts = []
    for d, ep, t_lr in zip(directories, num_epochs, target_lr):
        files = ['{}/loss_ep{}.pkl'.format(d, i) for i in range(1, ep)]

        files = [f for f in files if os.path.exists(f)]

        part = pd.concat([pd.read_pickle(f) for f in files])

        part['lr'] = compute_lr(target_lr=t_lr, n_epochs=ep, **kwargs)[0:len(part)]

        parts.append(part)

    return pd.concat(parts).reset_index(drop=True)


def plot_loss_against_lr(loss, wnd_size=6000):
    fig = plt.figure(figsize=(11.69, 8.27))

    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('time')
    ax1.set_ylabel('loss', color='b')

    ax1.plot(loss.loss.rolling(wnd_size).mean(), color='b')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('learning rate', color='r')

    ax2.plot(loss.lr.rolling(wnd_size).mean(), 'r')
