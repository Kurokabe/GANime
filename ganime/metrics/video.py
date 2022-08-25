import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub
from sklearn.metrics.pairwise import polynomial_kernel
from tqdm.auto import tqdm

i3d = hub.KerasLayer("https://tfhub.dev/deepmind/i3d-kinetics-400/1")


def resize_videos(videos, target_resolution):
    """Runs some preprocessing on the videos for I3D model.
    Args:
        videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
            preprocessed. We don't care about the specific dtype of the videos, it can
            be anything that tf.image.resize_bilinear accepts. Values are expected to
            be in [-1, 1].
        target_resolution: (width, height): target video resolution
    Returns:
        videos: <float32>[batch_size, num_frames, height, width, depth]
    """
    min_frames = 9
    B, T, H, W, C = videos.shape
    videos = tf.transpose(videos, (1, 0, 2, 3, 4))
    if T < min_frames:
        videos = tf.concat([tf.zeros((min_frames - T, B, H, W, C)), videos], axis=0)
    scaled_videos = tf.map_fn(lambda x: tf.image.resize(x, target_resolution), videos)
    scaled_videos = tf.transpose(scaled_videos, (1, 0, 2, 3, 4))
    return scaled_videos


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd


def calculate_ssim_videos(fake, real):
    fake = tf.cast(((fake * 0.5) + 1) * 255, tf.uint8)
    real = tf.cast(((real * 0.5) + 1) * 255, tf.uint8)
    ssims = []
    for i in range(fake.shape[0]):
        ssims.append(tf.image.ssim(fake[i], real[i], 255).numpy().mean())

    return np.array(ssims).mean()


def calculate_psnr_videos(fake, real):
    fake = tf.cast(((fake * 0.5) + 1) * 255, tf.uint8)
    real = tf.cast(((real * 0.5) + 1) * 255, tf.uint8)
    psnrs = []
    for i in range(fake.shape[0]):
        psnrs.append(tf.image.psnr(fake[i], real[i], 255).numpy().mean())

    return np.array(psnrs).mean()


def calculate_videos_metrics(dataset, model, total_length):
    fake_embeddings = []
    real_embeddings = []

    psnrs = []
    ssims = []

    for sample in tqdm(dataset, total=total_length):
        generated = model(sample, training=False)
        generated, real = generated[:, 1:], sample["y"][:, 1:]  # ignore first frame

        real_resized = resize_videos(real, (224, 224))
        generated_resized = resize_videos(generated, (224, 224))

        real_activations = i3d(real_resized)
        generated_activations = i3d(generated_resized)
        fake_embeddings.append(generated_activations)
        real_embeddings.append(real_activations)

        psnrs.append(calculate_psnr_videos(generated, real))
        ssims.append(calculate_ssim_videos(generated, real))

    # fake_concat, real_concat = tf.concat(fake_embeddings, axis=0), tf.concat(real_embeddings, axis=0)
    fvd = tfgan.eval.frechet_classifier_distance_from_activations(
        tf.concat(fake_embeddings, axis=0), tf.concat(real_embeddings, axis=0)
    )
    kvd = polynomial_mmd(
        tf.concat(fake_embeddings, axis=0), tf.concat(real_embeddings, axis=0)
    )
    psnr = np.array(psnrs).mean()
    ssim = np.array(ssims).mean()
    return {"fvd": fvd, "kvd": kvd, "ssim": ssim, "psnr": psnr}
