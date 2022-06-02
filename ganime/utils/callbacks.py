import io
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ganime.data.base import SequenceDataset
from ganime.visualization.images import display_true_pred


def get_logdir(parent_folder: str, experiment_name: Optional[str] = None) -> str:
    """Get the logdir used for logging in tensorboard. The logdir will be the parent folder with the experiment name and the current date and time.

    Args:
        parent_folder (str): The parent folder of the logdir
        experiment_name (str, optional): Optinal name of the experiment. Defaults to "".

    Returns:
        str: The path of the logdir that can be used by Tensorboard
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sub_folder = (
        f"{experiment_name}_{current_time}" if experiment_name else current_time
    )
    logdir = os.path.join(parent_folder, sub_folder)
    return logdir


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class TensorboardImage(tf.keras.callbacks.Callback):
    def __init__(
        self,
        logdir: str,
        train: SequenceDataset,
        validation: SequenceDataset = None,
        n_images: int = 3,
    ):
        super().__init__()
        self.logdir = logdir
        self.train = train
        self.validation = validation
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.n_images = n_images

    def on_epoch_end(self, epoch, logs):
        train_X, train_y = self.train.get_fixed_batch(0)
        train_X, train_y = self.truncate_X_y(train_X, train_y, self.n_images)
        train_pred = self.model.predict(train_X)
        self.write_to_tensorboard(train_y, train_pred, "Training data", epoch)

        if self.validation is not None:
            validation_X, validation_y = self.validation.get_fixed_batch(0)
            validation_X, validation_y = self.truncate_X_y(
                validation_X, validation_y, self.n_images
            )
            validation_pred = self.model.predict(validation_X)
            self.write_to_tensorboard(
                validation_y, validation_pred, "Validation data", epoch
            )

    def truncate_X_y(self, X, y, n_images):
        """Truncate the X and y arrays to the first n_images."""
        X = X[:n_images]
        y = y[:n_images]
        return X, y

    def write_to_tensorboard(self, y_true, y_pred, tag, step):
        with self.file_writer.as_default():
            tf.summary.image(
                tag,
                plot_to_image(display_true_pred(y_true, y_pred, n_cols=len(y_true))),
                step=step,
            )


class TensorboardVideo(tf.keras.callbacks.Callback):
    def __init__(
        self,
        logdir: str,
        train: SequenceDataset,
        validation: SequenceDataset = None,
        n_videos: int = 3,
    ):
        super().__init__()
        self.logdir = logdir
        self.train = train
        self.validation = validation
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.n_videos = n_videos

    def on_epoch_end(self, epoch, logs):
        train_X, train_y = self.train.get_fixed_batch(0)
        train_X, train_y = self.truncate_X_y(train_X, train_y, self.n_videos)
        train_pred = self.model.predict(train_X)
        self.write_to_tensorboard(train_y, train_pred, "Training data", epoch)

        if self.validation is not None:
            validation_X, validation_y = self.validation.get_fixed_batch(0)
            validation_X, validation_y = self.truncate_X_y(
                validation_X, validation_y, self.n_videos
            )
            validation_pred = self.model.predict(validation_X)
            self.write_to_tensorboard(
                validation_y, validation_pred, "Validation data", epoch
            )

    def truncate_X_y(self, X, y, n_videos):
        """Truncate the X and y arrays to the first n_videos."""
        X = X[:n_videos]
        y = y[:n_videos]
        return X, y

    def write_to_tensorboard(self, y_true, y_pred, tag, step):
        y_true = tf.concat(
            [y_pred[:, 0:1, ...], y_true], axis=1
        )  # Add first frame of pred to true to have same shape
        stacked = tf.concat([y_pred, y_true], axis=2)
        self.video_summary(tag, stacked, step)

    def video_summary(self, name, video, step=None, fps=20):
        name = tf.constant(name).numpy().decode("utf-8")
        video = np.array(video)
        if video.dtype in (np.float32, np.float64):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        B, T, H, W, C = video.shape

        with self.file_writer.as_default():
            try:
                frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
                summary = tf.compat.v1.Summary()
                image = tf.compat.v1.Summary.Image(
                    height=B * H, width=T * W, colorspace=C
                )
                image.encoded_image_string = self.encode_gif(frames, fps)
                summary.value.add(tag=name + "/gif", image=image)
                tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
            except (IOError, OSError) as e:
                print("GIF summaries require ffmpeg in $PATH.", e)
                frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
                tf.summary.image(name + "/grid", frames, step)

    def encode_gif(self, frames, fps):
        from subprocess import PIPE, Popen

        h, w, c = frames[0].shape
        pxfmt = {1: "gray", 3: "rgb24"}[c]
        cmd = " ".join(
            [
                f"ffmpeg -y -f rawvideo -vcodec rawvideo",
                f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
                f"[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
                f"-r {fps:.02f} -f gif -",
            ]
        )
        proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        for image in frames:
            proc.stdin.write(image.tostring())
        out, err = proc.communicate()
        if proc.returncode:
            raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
        del proc
        return out
