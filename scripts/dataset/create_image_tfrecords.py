import glob
from typing import Optional, Tuple

import click
import numpy as np
from ganime.data.experimental import ImageDataset
from joblib import Parallel, delayed
from tqdm.auto import tqdm

image_extensions = ["jpg", "jpeg", "png"]
video_extensions = ["mp4", "mkv", "avi"]


def load_and_resize_image(
    image_path: str, resize: Optional[Tuple[int, int]]
) -> np.array:
    import skimage.io

    image = skimage.io.imread(image_path)
    image = np.expand_dims(image, axis=0)
    if resize is not None:
        height, width = resize
        image = resize_images(image, width, height)
    return image


def load_npy(path: str) -> np.ndarray:
    images = np.load(path)
    return images


def get_filepaths(path: str, extension: str) -> list:
    filepaths = sorted(glob.glob(f"{path}/*.{extension}"))
    if len(filepaths) == 0:
        raise ValueError(f"No files found in {path} with the extension {extension}")
    return filepaths


def load_images(
    path: str, extension: str, resize: Optional[Tuple[int, int]], n_jobs: int
) -> np.ndarray:
    assert (
        extension in image_extensions
    ), f"Extension {extension} must be one of {image_extensions}"

    image_paths = get_filepaths(path, extension)
    print("loading images...")
    images = Parallel(n_jobs=n_jobs)(
        delayed(load_and_resize_image)(path, resize) for path in tqdm(image_paths)
    )
    return np.concatenate(images, axis=0)


def load_videos(
    path: str, extension: str, resize: Optional[Tuple[int, int]], skip: int, n_jobs: int
) -> np.ndarray:
    assert (
        extension in video_extensions
    ), f"Extension {extension} must be one of {video_extensions}"

    video_paths = get_filepaths(path, extension)
    print("loading videos...")
    videos = Parallel(n_jobs=n_jobs)(
        delayed(load_and_resize_video)(path, resize, skip) for path in tqdm(video_paths)
    )
    images = np.concatenate(videos, axis=0)
    return images


def load_and_resize_video(
    image_path: str, resize: Optional[Tuple[int, int]], skip: int
) -> np.array:
    import skvideo.io

    video = skvideo.io.vread(image_path)
    video = video[::skip]
    if resize is not None:
        height, width = resize
        video = resize_images(video, width, height)
    return video


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


def get_resize_shape(new_width, new_height, images) -> Tuple[int, int]:
    if new_width is None:
        new_width = images.shape[2]
    if new_height is None:
        new_height = images.shape[1]
    return new_width, new_height


@click.command()
@click.option(
    "--source",
    type=click.Choice(
        ["npy", "images", "videos"],
        case_sensitive=False,
    ),
    default="npy",
    help="The type of source to use",
)
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
    type=click.Choice(image_extensions + video_extensions, case_sensitive=False),
    default="jpg",
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
    "--skip",
    type=int,
    default=1,
    help="If set, used when loading from videos, will take one every n frames",
)
def run(
    source: str,
    path: str,
    resize: Tuple[int, int],
    extension: str,
    output_dir: str,
    filename: str,
    n_jobs: int,
    skip: int,
):
    if source == "npy":
        images = load_npy(path)
    elif source == "images":
        images = load_images(path, extension, resize, n_jobs=n_jobs)
    elif source == "videos":
        images = load_videos(path, extension, resize, skip=skip, n_jobs=n_jobs)

    ImageDataset.write_to_tfr(images, out_dir=output_dir, filename=filename)


if __name__ == "__main__":
    run()
