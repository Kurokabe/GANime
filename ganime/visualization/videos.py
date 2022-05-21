import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


def display_images(data, n_rows=3, n_cols=3):
    figure, axs = plt.subplots(n_rows, n_cols, figsize=(24, 12))
    axs = axs.flatten()
    for img, ax in zip(data, axs):
        ax.imshow(img)

    return figure


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
    # FFwriter = animation.FFMpegWriter(fps=10, codec="libx264")
    # anim.save("basic_animation1.mp4", writer=FFwriter)

    return HTML(anim.to_html5_video())
