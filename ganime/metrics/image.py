import numpy as np
import tensorflow as tf
from scipy import linalg
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tqdm.auto import tqdm

inceptionv3 = InceptionV3(include_top=False, weights="imagenet", pooling="avg")


def resize_images(images, new_shape):
    images = tf.image.resize(images, new_shape)
    return images


def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(
        generated_embeddings, rowvar=False
    )
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_images_metrics(dataset, model, total_length):
    fake_embeddings = []
    real_embeddings = []

    psnrs = []
    ssims = []

    for sample in tqdm(dataset, total=total_length):
        generated = model(sample[0], training=False)[0]
        generated, real = generated, sample[0]

        real_resized = resize_images(real, (299, 299))
        generated_resized = resize_images(generated, (299, 299))

        real_activations = inceptionv3(real_resized, training=False)
        generated_activations = inceptionv3(generated_resized, training=False)
        fake_embeddings.append(generated_activations)
        real_embeddings.append(real_activations)

        fake_scaled = tf.cast(((generated * 0.5) + 1) * 255, tf.uint8)
        real_scaled = tf.cast(((real * 0.5) + 1) * 255, tf.uint8)

        psnrs.append(tf.image.psnr(fake_scaled, real_scaled, 255).numpy())
        ssims.append(tf.image.ssim(fake_scaled, real_scaled, 255).numpy())

    fid = calculate_fid(
        tf.concat(fake_embeddings, axis=0).numpy(),
        tf.concat(real_embeddings, axis=0).numpy(),
    )

    # kid = calculate_kid(
    #     tf.concat(fake_embeddings, axis=0).numpy(),
    #     tf.concat(real_embeddings, axis=0).numpy(),
    # )

    psnr = np.array(psnrs).mean()
    ssim = np.array(ssims).mean()
    return {"fid": fid, "ssim": ssim, "psnr": psnr}
