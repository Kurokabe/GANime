import glob
import os
from typing import Literal

import numpy as np

from .base import SequenceDataset
import math


class MovingMNISTImage(SequenceDataset):
    def load_data(self, dataset_path: str, split: str) -> np.ndarray:
        data = np.load(os.path.join(dataset_path, "moving_mnist", "mnist_test_seq.npy"))
        # Data is of shape (window, n_samples, width, height)
        # But we want for keras something of shape (n_samples, window, width, height)
        data = np.moveaxis(data, 0, 1)
        # Also expand dimensions to have channels at the end (n_samples, window, width, height, channels)
        data = np.expand_dims(data, axis=-1)
        if split == "train":
            data = data[:-1000]
        else:
            data = data[-1000:]

        data = np.concatenate([data, data, data], axis=-1)

        return data

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.data[inds, 0, ...]
        batch_y = self.data[inds, 4, ...]

        return batch_x, batch_y

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data / 255


class MovingMNIST(SequenceDataset):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        split: Literal["train", "validation", "test"] = "train",
    ):
        self.batch_size = batch_size
        self.split = split
        root_path = os.path.join(dataset_path, "moving_mnist", split)
        self.paths = glob.glob(os.path.join(root_path, "*.npy"))
        # self.data = self.preprocess_data(self.data)

        self.indices = np.arange(len(self.paths))
        self.on_epoch_end()

    # def load_data(self, dataset_path: str, split: str) -> np.ndarray:
    # data = np.load(os.path.join(dataset_path, "moving_mnist", "mnist_test_seq.npy"))
    # # Data is of shape (window, n_samples, width, height)
    # # But we want for keras something of shape (n_samples, window, width, height)
    # data = np.moveaxis(data, 0, 1)
    # # Also expand dimensions to have channels at the end (n_samples, window, width, height, channels)
    # data = np.expand_dims(data, axis=-1)
    # if split == "train":
    #     data = data[:100]
    # else:
    #     data = data[100:110]

    # data = np.concatenate([data, data, data], axis=-1)

    # return data

    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        data = self.load_indices(inds)
        batch_x = np.concatenate([data[:, 0:1, ...], data[:, -1:, ...]], axis=1)
        batch_y = data[:, 1:, ...]

        return batch_x, batch_y

    def get_fixed_batch(self, idx):
        self.fixed_indices = (
            self.fixed_indices
            if hasattr(self, "fixed_indices")
            else self.indices[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ].copy()
        )
        data = self.load_indices(self.fixed_indices)
        batch_x = np.concatenate([data[:, 0:1, ...], data[:, -1:, ...]], axis=1)
        batch_y = data[:, 1:, ...]

        return batch_x, batch_y

    def load_indices(self, indices):
        paths_to_load = [self.paths[index] for index in indices]
        data = [np.load(path) for path in paths_to_load]
        data = np.array(data)
        return self.preprocess_data(data)

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data / 255
