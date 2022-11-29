import tempfile

import cv2
import ffmpegio
import gradio as gr
import numpy as np
import omegaconf
import tensorflow as tf
from pyprojroot.pyprojroot import here

from ganime.model.vqgan_clean.experimental.net2net_v3 import Net2Net

IMAGE_SHAPE = (64, 128, 3)

cfg = omegaconf.OmegaConf.load(here("configs/kny_video_gpt2_large_gradio.yaml"))
model = Net2Net(**cfg["model"], trainer_config=cfg["train"], num_replicas=1)
model.first_stage_model.build((20, *IMAGE_SHAPE))


# def save_video(video):
#     b, f, h, w, c = 1, 20, 500, 500, 3

#     # filename = output_file.name
#     filename = "./test_video.mp4"
#     images = []
#     for i in range(f):
#         # image = video[0][i].numpy()
#         # image = 255 * image  # Now scale by 255
#         # image = image.astype(np.uint8)
#         images.append(np.random.randint(0, 255, (h, w, c), dtype=np.uint8))

#     ffmpegio.video.write(filename, 20, np.array(images), overwrite=True)
#     return filename


def save_video(video):
    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    b, f, h, w, c = video.shape

    filename = output_file.name
    video = video.numpy()
    video = video * 255
    video = video.astype(np.uint8)
    ffmpegio.video.write(filename, 20, video, overwrite=True)
    return filename


def resize_if_necessary(image):
    if image.shape[0] != 64 and image.shape[1] != 128:
        image = tf.image.resize(image, (64, 128))
    return image


def normalize(image):
    image = (tf.cast(image, tf.float32) / 127.5) - 1

    return image


def generate(first, last, n_frames):
    # n_frames = 20
    n_frames = int(n_frames)
    first = resize_if_necessary(first)
    last = resize_if_necessary(last)
    first = normalize(first)
    last = normalize(last)
    data = {
        "first_frame": np.expand_dims(first, axis=0),
        "last_frame": np.expand_dims(last, axis=0),
        "y": None,
        "n_frames": [n_frames],
        "remaining_frames": [list(reversed(range(n_frames)))],
    }
    generated = model.predict(data)

    return save_video(generated)


gr.Interface(
    generate,
    inputs=[
        gr.Image(label="Upload the first image"),
        gr.Image(label="Upload the last image"),
        gr.Slider(
            label="Number of frame to generate",
            minimum=15,
            maximum=100,
            value=15,
            step=1,
        ),
    ],
    outputs="video",
    title="Generate a video from the first and last frame",
).launch(share=True)
