from typing import Tuple
import numpy as np
import tensorflow as tf
import os


def load_moving_mnist(
    dataset_path: str, batch_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tuple]:
    data = np.load(os.path.join(dataset_path, "mnist_test_seq.npy"))
    data.shape

    # We can see that data is of shape (window, n_samples, width, height)
    # But we want for keras something of shape (n_samples, window, width, height)
    data = np.moveaxis(data, 0, 1)
    # Also expand dimensions to have channels at the end (n_samples, window, width, height, channels)
    data = np.expand_dims(data, axis=-1)

    def _preprocess(sample):
        image = tf.cast(sample, tf.float32) / 255.0  # Scale to unit interval.
        image = image < tf.random.uniform(tf.shape(image))  # Randomly binarize.
        return image, image

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(data[:9000])
        .map(_preprocess)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .shuffle(int(10e3))
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(data[9000:])
        .map(_preprocess)
        .batch(256)
        .prefetch(tf.data.AUTOTUNE)
        .shuffle(int(10e3))
    )

    return train_dataset, test_dataset, data.shape[1:]


def load_dataset(
    dataset_name: str, dataset_path: str, batch_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tuple]:
    if dataset_name == "moving_mnist":
        return load_moving_mnist(dataset_path, batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
