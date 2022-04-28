# %%
from ganime.data.base import load_dataset
from ganime.model.p2p.p2p import P2P
import tensorflow as tf
from tqdm import tqdm

tf.get_logger().setLevel("ERROR")

# %%
train_ds, test_ds, input_shape = load_dataset(
    "moving_mnist", "../../data", batch_size=20
)

# %%
model = P2P()
# model.encoder(tf.zeros((input_shape)))
# model.encoder.summary()
model.decoder(model.encoder(tf.ones((input_shape))))
model.decoder.summary()

# %%
for epoch in tqdm(range(100)):
    epoch_mse = 0
    epoch_kld = 0
    epoch_align = 0
    epoch_cpc = 0

    for i in tqdm(range(300)):
        x = next(train_ds.as_numpy_iterator())
        x = x[0]

        start_ix = 0
        cp_ix = -1
        cp_ix = x.shape[1] - 1

        mse, kld, cpc, align = model(x, start_ix, cp_ix)
        epoch_mse += mse
        epoch_kld += kld
        epoch_cpc += cpc
        epoch_align += align
    print("EPOCH", epoch)
    print("epoch mse", epoch_mse)
    print("epoch kld", epoch_kld)
    print("epoch cpc", epoch_cpc)
    print("epoch align", epoch_align)

# %%
x = next(test_ds.as_numpy_iterator())
n_samples = 10
samples = []
for i in range(n_samples):
    generated = model.p2p_generate(x[1][i : 20 + i], 20, 19)
    samples.append(generated[0])

samples = tf.stack(samples)

# %%
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from IPython.display import HTML


def display_videos(data, n_rows=3, n_cols=3):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False)
    ims = []

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_rows + j
            video = data[idx]
            im = axs[i][j].imshow(video[0, :, :, :], animated=True)
            ims.append(im)

            plt.close()  # this is required to not display the generated image

    def init():
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_rows + j
                video = data[idx]
                im = ims[idx]
                im.set_data(video[0, :, :, :])
        return ims

    def animate(frame_id):
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_rows + j
                video = data[idx]
                ims[idx].set_data(video[frame_id, :, :, :])
        return ims

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=data.shape[1], blit=True, interval=100
    )
    # return HTML(anim.to_html5_video())

    writergif = animation.PillowWriter(fps=5)
    anim.save("animation_longer.gif", writer=writergif)


# %%
display_videos(samples)

# %%
