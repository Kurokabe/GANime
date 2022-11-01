import glob
from typing import Optional, Tuple

import click
import numpy as np
import tensorflow as tf
from ganime.data.experimental import VideoDataset
from joblib import Parallel, delayed
from tqdm.auto import tqdm

video_extensions = ["mp4", "mkv", "avi"]


def load_npy(path: str) -> np.ndarray:
    images = np.load(path)
    return images


def get_filepaths(path: str, extension: str) -> list:
    filepaths = sorted(glob.glob(f"{path}/*.{extension}"))
    if len(filepaths) == 0:
        raise ValueError(f"No files found in {path} with the extension {extension}")
    return filepaths


def get_value_to_split(
    video: np.ndarray, min_ideal_length: int, max_ideal_length: int
) -> int:
    n_elements = video.shape[0]

    lowest_remainder = max_ideal_length
    lowest_id = max_ideal_length
    for i in range(min_ideal_length, max_ideal_length):
        remainder = n_elements % i
        if remainder <= lowest_remainder:
            lowest_id = i
            lowest_remainder = remainder
    return lowest_id


def cut_long_video(video, length):
    n_splits = video.shape[0] // length
    if n_splits == 0:
        yield video
    else:
        for i in range(n_splits):
            start = i * length
            end = (i + 1) * length
            if video.shape[0] - end < length:
                end = video.shape[0]
            yield video[start:end]


def load_videos(
    path: str,
    extension: str,
    resize: Optional[Tuple[int, int]],
    min_ideal_length: int,
    max_ideal_length: int,
    n_jobs: int,
) -> np.ndarray:
    assert (
        extension in video_extensions
    ), f"Extension {extension} must be one of {video_extensions}"

    video_paths = get_filepaths(path, extension)
    print("loading videos...")
    # videos = Parallel(n_jobs=n_jobs)(
    #     delayed(load_and_resize_video)(path, resize, min_ideal_length, max_ideal_length)
    #     for path in tqdm(video_paths)
    # )
    videos = [load_and_resize_video(path, resize, min_ideal_length, max_ideal_length) for path in tqdm(video_paths)]
    print("before concat")
    videos = tf.concat(videos, axis=0)
    print("after concat")
    videos = videos.numpy()
    return videos


def load_and_resize_video(
    image_path: str,
    resize: Optional[Tuple[int, int]],
    min_ideal_length: int,
    max_ideal_length: int,
    is_pyscene_detect_generated: bool = True,
    skip_frames: int = 1,
) -> np.array:
    import skvideo.io

    videos = []
    video = skvideo.io.vread(image_path)
    if resize is not None:
        height, width = resize
        video = resize_images(video, width, height)

    if is_pyscene_detect_generated:
        video = video[2:]

    video = video[:: 1 + skip_frames]

    for video_split in cut_long_video(
        video, get_value_to_split(video, min_ideal_length, max_ideal_length)
    ):
        videos.append(video_split)

    videos = tf.ragged.stack([tf.convert_to_tensor(video) for video in videos], axis=0)

    return videos


def resize_images(images: np.ndarray, width: int, height: int) -> np.ndarray:
    import skimage.transform

    n_images = images.shape[0]
    resized_images = np.zeros((n_images, height, width, 3), dtype=np.uint8)
    # print("resizing images...")
    for i in range(n_images):
        resized = skimage.transform.resize(
            images[i], (height, width), anti_aliasing=True
        )
        resized = (resized * 255).astype(np.uint8)
        resized_images[i] = resized
    return resized_images


@click.command()
@click.option(
    "--path",
    type=click.Path(),
    help="The path to the source",
)
@click.option(
    "--resize",
    type=(int, int),
    help="If specified, resize to this (height, width)",
)
@click.option(
    "--extension",
    type=click.Choice(video_extensions, case_sensitive=False),
    default="mp4",
    help="The extension of the source",
)
@click.option(
    "--output_dir",
    type=str,
    required=True,
    help="The directory to save the output",
)
@click.option(
    "--filename",
    type=str,
    required=True,
    help="The name of the files inside the output directory",
)
@click.option(
    "--n_jobs",
    type=int,
    default=1,
    help="The number of parallel jobs",
)
@click.option(
    "--min_ideal_length",
    type=int,
    default=15,
    help="The minimal ideal length to split long scenes",
)
@click.option(
    "--max_ideal_length",
    type=int,
    default=25,
    help="The maximal ideal length to split long scenes",
)
def run(
    path: str,
    resize: Tuple[int, int],
    extension: str,
    output_dir: str,
    filename: str,
    min_ideal_length: int,
    max_ideal_length: int,
    n_jobs: int,
):
    videos = load_videos(
        path,
        extension,
        resize,
        min_ideal_length=min_ideal_length,
        max_ideal_length=max_ideal_length,
        n_jobs=n_jobs,
    )

    VideoDataset.write_to_tfr(videos, out_dir=output_dir, filename=filename)


if __name__ == "__main__":
    run()
