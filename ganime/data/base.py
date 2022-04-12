from pyprojroot.pyprojroot import here

import tensorflow_datasets as tfds
import tensorflow as tf


def load_dataset(dataset: str):
    ds, ds_info = tfds.load(
        dataset, shuffle_files=True, as_supervised=False, with_info=True
    )
    ds_train = (
        ds["train"]
        .map(_preprocess)
        .batch(256)
        .prefetch(tf.data.AUTOTUNE)
        .shuffle(int(10e3))
    )
    return ds_train, ds_info


def _preprocess(sample):
    video = sample["video"]
    video = tf.cast(video, tf.float32) / 255.0  # Scale to unit interval.
    video = video < tf.random.uniform(tf.shape(video))  # Randomly binarize.
    return video, video
