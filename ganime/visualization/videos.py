import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation


def display_videos(data, ground_truth=None, n_rows=3, n_cols=3):

    if ground_truth is not None:
        data = np.concatenate((data, ground_truth), axis=2)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(16, 9))

    # remove grid and ticks
    plt.setp(axs, xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0, hspace=0)

    ims = []

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            video = data[idx]
            im = axs[i][j].imshow(video[0, :, :, :], animated=True)
            ims.append(im)

            plt.close()  # this is required to not display the generated image

    def init():
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                video = data[idx]
                im = ims[idx]
                im.set_data(video[0, :, :, :])
        return ims

    def animate(frame_id):
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                video = data[idx]
                d = video[frame_id, :, :, :]
                # if frame_id % 2 == 0:
                #     d[0:2, :, 0] = 255
                #     d[0:2, :, 1] = 0
                #     d[0:2, :, 2] = 0
                #     d[-2:, :, 0] = 255
                #     d[-2:, :, 1] = 0
                #     d[-2:, :, 2] = 0
                ims[idx].set_data(d)
        return ims

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=data.shape[1], blit=True, interval=200
    )
    # FFwriter = animation.FFMpegWriter(fps=10, codec="libx264")
    # anim.save("basic_animation1.mp4", writer=FFwriter)

    return HTML(anim.to_html5_video())
