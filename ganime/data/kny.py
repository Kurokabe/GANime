import os

import numpy as np

from .base import SequenceDataset


class KNYImage(SequenceDataset):
    def load_data(self, dataset_path: str, split: str) -> np.ndarray:
        data = np.load(os.path.join(dataset_path, "kny", "kny_images_64x128.npy"))
        if split == "train":
            data = data[:-5000]
        else:
            data = data[-5000:]

        return data

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        return data / 255
