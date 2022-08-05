import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def display_images(data, n_rows=3, n_cols=3):
    figure, axs = plt.subplots(n_rows, n_cols, figsize=(24, 12))
    axs = axs.flatten()
    for img, ax in zip(data, axs):
        ax.imshow(img)

    return figure


def unnormalize_if_necessary(x):
    if isinstance(x, np.ndarray):
        if x.min() < 0:

            return (x * 0.5) + 0.5
    elif isinstance(x, tf.Tensor):
        if x.numpy().min() < 0:
            return (x * 0.5) + 0.5
    return x


def display_true_pred(y_true, y_pred, n_cols=3):

    fig = plt.figure(constrained_layout=True, figsize=(24, 12))

    y_true = unnormalize_if_necessary(y_true)
    y_pred = unnormalize_if_necessary(y_pred)

    images = [y_pred, y_true]

    # create 2x1 subfigs
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle("Prediction" if row == 0 else "Ground truth", fontsize=24)

        # create 1xn_cols subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=n_cols)
        for col, ax in enumerate(axs):
            if row == 0:
                ax.imshow(images[row][col])
            else:
                ax.imshow(images[row][col])

    return fig
