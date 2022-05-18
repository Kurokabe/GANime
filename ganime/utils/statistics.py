import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds


def dataset_statistics(ds):
    if isinstance(ds, tf.data.Dataset):
        ds_numpy = tfds.as_numpy(ds)
    elif isinstance(ds, tf.keras.utils.Sequence):
        ds_numpy = ds
    data = []

    for da in tqdm(ds_numpy):
        X, y = da
        data.append(X)
    all_data = np.concatenate(data)
    return np.mean(all_data), np.var(all_data), np.std(all_data)
