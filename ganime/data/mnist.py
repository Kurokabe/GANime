import os

import numpy as np

from .base import SequenceDataset


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

        return data

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data / 255
