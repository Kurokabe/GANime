from abc import ABC, abstractclassmethod, abstractmethod
import glob
import math
import os
from typing import Dict
from typing_extensions import dataclass_transform

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


class Dataset(ABC):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @classmethod
    def _parse_single_element(cls, element) -> tf.train.Example:

        features = tf.train.Features(feature=cls._get_features(element))

        return tf.train.Example(features=features)

    @abstractclassmethod
    def _get_features(cls, element) -> Dict[str, tf.train.Feature]:
        pass

    @abstractclassmethod
    def _parse_tfr_element(cls, element):
        pass

    @classmethod
    def write_to_tfr(cls, data: np.ndarray, out_dir: str, filename: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Write all elements to a single tfrecord file
        single_file_name = cls.__write_to_single_tfr(data, out_dir, filename)

        # The optimal size for a single tfrecord file is around 100 MB. Get the number of files that need to be created
        number_splits = cls.__get_number_splits(single_file_name)

        if number_splits > 1:
            os.remove(single_file_name)
            cls.__write_to_multiple_tfr(data, out_dir, filename, number_splits)

    @classmethod
    def __write_to_multiple_tfr(
        cls, data: np.array, out_dir: str, filename: str, n_splits: int
    ):

        file_count = 0

        max_files = math.ceil(data.shape[0] / n_splits)

        print(f"Creating {n_splits} files with {max_files} elements each.")

        for i in tqdm(range(n_splits)):
            current_shard_name = os.path.join(
                out_dir,
                f"{filename}.tfrecords-{str(i).zfill(len(str(n_splits)))}-of-{n_splits}",
            )
            writer = tf.io.TFRecordWriter(current_shard_name)

            current_shard_count = 0
            while current_shard_count < max_files:  # as long as our shard is not full
                # get the index of the file that we want to parse now
                index = i * max_files + current_shard_count
                if index >= len(
                    data
                ):  # when we have consumed the whole data, preempt generation
                    break

                current_element = data[index]

                # create the required Example representation
                out = cls._parse_single_element(element=current_element)

                writer.write(out.SerializeToString())
                current_shard_count += 1
                file_count += 1

        writer.close()
        print(f"\nWrote {file_count} elements to TFRecord")
        return file_count

    @classmethod
    def __get_number_splits(cls, filename: str):
        target_size = 100 * 1024 * 1024  # 100mb

        single_file_size = os.path.getsize(filename)
        number_splits = math.ceil(single_file_size / target_size)
        return number_splits

    @classmethod
    def __write_to_single_tfr(cls, data: np.array, out_dir: str, filename: str):

        current_path_name = os.path.join(
            out_dir,
            f"{filename}.tfrecords-0-of-1",
        )

        writer = tf.io.TFRecordWriter(current_path_name)
        for element in tqdm(data):
            writer.write(cls._parse_single_element(element).SerializeToString())
        writer.close()

        return current_path_name

    def load(self) -> tf.data.TFRecordDataset:
        path = self.dataset_path
        dataset = None

        if os.path.isdir(path):
            dataset = self._load_folder(path)
        elif os.path.isfile(path):
            dataset = self._load_file(path)
        else:
            raise ValueError(f"Path {path} is not a valid file or folder.")

        dataset = dataset.map(self._parse_tfr_element)
        return dataset

    def _load_file(self, path) -> tf.data.TFRecordDataset:
        return tf.data.TFRecordDataset(path)

    def _load_folder(self, path) -> tf.data.TFRecordDataset:

        return tf.data.TFRecordDataset(
            glob.glob(os.path.join(path, "**/*.tfrecords*"), recursive=True)
        )


class VideoDataset(Dataset):
    @classmethod
    def _get_features(cls, element) -> Dict[str, tf.train.Feature]:
        return {
            "frames": _int64_feature(element.shape[0]),
            "height": _int64_feature(element.shape[1]),
            "width": _int64_feature(element.shape[2]),
            "depth": _int64_feature(element.shape[3]),
            "raw_video": _bytes_feature(serialize_array(element)),
        }

    @classmethod
    def _parse_tfr_element(cls, element):
        # use the same structure as above; it's kinda an outline of the structure we now want to create
        data = {
            "frames": tf.io.FixedLenFeature([], tf.int64),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "raw_video": tf.io.FixedLenFeature([], tf.string),
            "depth": tf.io.FixedLenFeature([], tf.int64),
        }

        content = tf.io.parse_single_example(element, data)

        frames = content["frames"]
        height = content["height"]
        width = content["width"]
        depth = content["depth"]
        raw_video = content["raw_video"]

        # get our 'feature'-- our image -- and reshape it appropriately
        feature = tf.io.parse_tensor(raw_video, out_type=tf.uint8)
        feature = tf.reshape(feature, shape=[frames, height, width, depth])
        return feature


class ImageDataset(Dataset):
    @classmethod
    def _get_features(cls, element) -> Dict[str, tf.train.Feature]:
        return {
            "height": _int64_feature(element.shape[0]),
            "width": _int64_feature(element.shape[1]),
            "depth": _int64_feature(element.shape[2]),
            "raw_image": _bytes_feature(serialize_array(element)),
        }

    @classmethod
    def _parse_tfr_element(cls, element):
        # use the same structure as above; it's kinda an outline of the structure we now want to create
        data = {
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "raw_image": tf.io.FixedLenFeature([], tf.string),
            "depth": tf.io.FixedLenFeature([], tf.int64),
        }

        content = tf.io.parse_single_example(element, data)

        height = content["height"]
        width = content["width"]
        depth = content["depth"]
        raw_image = content["raw_image"]

        # get our 'feature'-- our image -- and reshape it appropriately
        feature = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
        feature = tf.reshape(feature, shape=[height, width, depth])
        return feature
