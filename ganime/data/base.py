from typing import Tuple
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
from abc import ABC, abstractmethod
from typing import Literal
import math
from ganime.data.experimental import ImageDataset


# class SequenceDataset(Sequence):
#     def __init__(
#         self,
#         dataset_path: str,
#         batch_size: int,
#         split: Literal["train", "validation", "test"] = "train",
#     ):
#         self.batch_size = batch_size
#         self.split = split
#         self.data = self.load_data(dataset_path, split)
#         self.data = self.preprocess_data(self.data)

#         self.indices = np.arange(self.data.shape[0])
#         self.on_epoch_end()

#     @abstractmethod
#     def load_data(self, dataset_path: str, split: str) -> np.ndarray:
#         pass

#     def preprocess_data(self, data: np.ndarray) -> np.ndarray:
#         return data

#     def __len__(self):
#         return math.ceil(len(self.data) / self.batch_size)

#     def __getitem__(self, idx):
#         inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
#         batch_x = self.data[inds]
#         batch_y = batch_x

#         return batch_x, batch_y

#     def get_fixed_batch(self, idx):
#         self.fixed_indices = (
#             self.fixed_indices
#             if hasattr(self, "fixed_indices")
#             else self.indices[
#                 idx * self.batch_size : (idx + 1) * self.batch_size
#             ].copy()
#         )
#         batch_x = self.data[self.fixed_indices]
#         batch_y = batch_x

#         return batch_x, batch_y

#     def on_epoch_end(self):
#         np.random.shuffle(self.indices)


# def load_kny_images(
#     dataset_path: str, batch_size: int
# ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tuple]:
#     import skvideo.io

#     if os.path.exists(os.path.join(dataset_path, "kny", "kny_images.npy")):
#         data = np.load(os.path.join(dataset_path, "kny", "kny_images.npy"))
#     else:
#         data = skvideo.io.vread(os.path.join(dataset_path, "kny", "01.mp4"))
#     np.random.shuffle(data)

#     def _preprocess(sample):
#         image = tf.cast(sample, tf.float32) / 255.0  # Scale to unit interval.
#         # video = video < tf.random.uniform(tf.shape(video))  # Randomly binarize.
#         image = tf.image.resize(image, [64, 64])

#         return image, image

#     train_dataset = (
#         tf.data.Dataset.from_tensor_slices(data[:5000])
#         .map(_preprocess)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#         .shuffle(int(10e3))
#     )
#     test_dataset = (
#         tf.data.Dataset.from_tensor_slices(data[5000:6000])
#         .map(_preprocess)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#         .shuffle(int(10e3))
#     )

#     return train_dataset, test_dataset, data.shape[1:]


# def load_moving_mnist_vae(
#     dataset_path: str, batch_size: int
# ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tuple]:
#     data = np.load(os.path.join(dataset_path, "moving_mnist", "mnist_test_seq.npy"))
#     data.shape

#     # We can see that data is of shape (window, n_samples, width, height)
#     # But we want for keras something of shape (n_samples, window, width, height)
#     data = np.moveaxis(data, 0, 1)
#     # Also expand dimensions to have channels at the end (n_samples, window, width, height, channels)
#     data = np.expand_dims(data, axis=-1)

#     def _preprocess(sample):
#         video = tf.cast(sample, tf.float32) / 255.0  # Scale to unit interval.
#         # video = video < tf.random.uniform(tf.shape(video))  # Randomly binarize.
#         return video, video

#     train_dataset = (
#         tf.data.Dataset.from_tensor_slices(data[:9000])
#         .map(_preprocess)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#         .shuffle(int(10e3))
#     )
#     test_dataset = (
#         tf.data.Dataset.from_tensor_slices(data[9000:])
#         .map(_preprocess)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#         .shuffle(int(10e3))
#     )

#     return train_dataset, test_dataset, data.shape[1:]


# def load_moving_mnist(
#     dataset_path: str, batch_size: int
# ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tuple]:
#     data = np.load(os.path.join(dataset_path, "moving_mnist", "mnist_test_seq.npy"))
#     data.shape

#     # We can see that data is of shape (window, n_samples, width, height)
#     # But we want for keras something of shape (n_samples, window, width, height)
#     data = np.moveaxis(data, 0, 1)
#     # Also expand dimensions to have channels at the end (n_samples, window, width, height, channels)
#     data = np.expand_dims(data, axis=-1)

#     def _preprocess(sample):
#         video = tf.cast(sample, tf.float32) / 255.0  # Scale to unit interval.
#         # video = video < tf.random.uniform(tf.shape(video))  # Randomly binarize.
#         first_frame = video[0:1, ...]
#         last_frame = video[-1:, ...]
#         first_last = tf.concat([first_frame, last_frame], axis=0)

#         return first_last, video

#     train_dataset = (
#         tf.data.Dataset.from_tensor_slices(data[:9000])
#         .map(_preprocess)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#         .shuffle(int(10e3))
#     )
#     test_dataset = (
#         tf.data.Dataset.from_tensor_slices(data[9000:])
#         .map(_preprocess)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#         .shuffle(int(10e3))
#     )

#     return train_dataset, test_dataset, data.shape[1:]


# def load_mnist(
#     dataset_path: str, batch_size: int
# ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tuple]:
#     data = np.load(os.path.join(dataset_path, "moving_mnist", "mnist_test_seq.npy"))
#     data.shape

#     # We can see that data is of shape (window, n_samples, width, height)
#     # But we want for keras something of shape (n_samples, window, width, height)
#     data = np.moveaxis(data, 0, 1)
#     # Also expand dimensions to have channels at the end (n_samples, window, width, height, channels)
#     data = np.expand_dims(data, axis=-1)

#     def _preprocess(sample):
#         video = tf.cast(sample, tf.float32) / 255.0  # Scale to unit interval.
#         # video = video < tf.random.uniform(tf.shape(video))  # Randomly binarize.
#         first_frame = video[0, ...]

#         first_frame = tf.image.grayscale_to_rgb(first_frame)

#         return first_frame, first_frame

#     train_dataset = (
#         tf.data.Dataset.from_tensor_slices(data[:9000])
#         .map(_preprocess)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#         .shuffle(int(10e3))
#     )
#     test_dataset = (
#         tf.data.Dataset.from_tensor_slices(data[9000:])
#         .map(_preprocess)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#         .shuffle(int(10e3))
#     )

#     return train_dataset, test_dataset, data.shape[1:]
def preprocess_image(element):
    element = tf.reshape(element, (tf.shape(element)[0], tf.shape(element)[1], 3))
    element = tf.cast(element, tf.float32) / 255.0
    return element, element


def load_kny_images_light(dataset_path, batch_size):
    dataset_length = 34045
    path = os.path.join(dataset_path, "kny", "images_tfrecords_light")
    dataset = ImageDataset(path).load()
    dataset = dataset.shuffle(
        dataset_length, reshuffle_each_iteration=True, seed=10
    ).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_size = int(dataset_length * 0.8)
    validation_size = int(dataset_length * 0.1)

    train_ds = dataset.take(train_size)
    validation_ds = dataset.skip(train_size).take(validation_size)
    test_ds = dataset.skip(train_size + validation_size).take(validation_size)

    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )
    validation_ds = validation_ds.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_ds, validation_ds, test_ds


def load_kny_images(dataset_path, batch_size):
    dataset_length = 52014
    path = os.path.join(dataset_path, "kny", "images_tfrecords")
    dataset = ImageDataset(path).load()
    dataset = dataset.shuffle(dataset_length, reshuffle_each_iteration=True).map(
        preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    train_size = int(dataset_length * 0.8)
    validation_size = int(dataset_length * 0.1)

    train_ds = dataset.take(train_size)
    validation_ds = dataset.skip(train_size).take(validation_size)
    test_ds = dataset.skip(train_size + validation_size).take(validation_size)

    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )
    validation_ds = validation_ds.batch(batch_size, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return train_ds, validation_ds, test_ds


def load_dataset(
    dataset_name: str, dataset_path: str, batch_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    # if dataset_name == "moving_mnist_vae":
    #     return load_moving_mnist_vae(dataset_path, batch_size)
    # elif dataset_name == "moving_mnist":
    #     return load_moving_mnist(dataset_path, batch_size)
    # elif dataset_name == "mnist":
    #     return load_mnist(dataset_path, batch_size)
    # elif dataset_name == "kny_images":
    #     return load_kny_images(dataset_path, batch_size)
    if dataset_name == "kny_images":
        return load_kny_images(dataset_path, batch_size)
    if dataset_name == "kny_images_light":
        return load_kny_images_light(dataset_path, batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
