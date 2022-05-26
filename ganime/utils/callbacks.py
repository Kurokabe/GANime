import io
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
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
    ):
        super().__init__()
        self.logdir = logdir
        self.train = train
        self.validation = validation
        self.file_writer = tf.summary.create_file_writer(logdir)

    def on_epoch_end(self, epoch, logs):
        train_X, train_y = self.train.get_fixed_batch(0)
        val_X, val_y = self.validation.get_fixed_batch(0)

        train_pred = self.model.predict(train_X)
        val_pred = self.model.predict(val_X)

        with self.file_writer.as_default():
            tf.summary.image(
                "Training data",
                plot_to_image(display_true_pred(train_y, train_pred)),
                step=epoch,
            )
            tf.summary.image(
                "Validation data",
                plot_to_image(display_true_pred(val_y, val_pred)),
                step=epoch,
            )
