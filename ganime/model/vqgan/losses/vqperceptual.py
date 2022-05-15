from typing import List, Literal

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import Loss

from ganime.model.vqgan.losses.lpips import LPIPS

from ..discriminator.model import NLayerDiscriminator


def hinge_d_loss(logits_real, logits_fake):
    loss_real = tf.reduce_mean(layers.ReLU(1.0 - logits_real))
    loss_fake = tf.reduce_mean(layers.ReLU(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        tf.reduce_mean(keras.activations.softplus(-logits_real))
        + tf.reduce_mean(keras.activations.softplus(logits_fake))
    )
    return d_loss


class VQLPIPSWithDiscriminator(Loss):
    def __init__(
        self,
        *,
        disc_num_layers: int = 3,
        disc_factor: float = 1.0,
        disc_iter_start: int = 0,
        disc_conditional: bool = False,
        disc_in_channels: int = 3,
        disc_weight: float = 0.8,
        disc_filters: int = 64,
        disc_loss: Literal["hinge", "vanilla"] = "hinge",
        codebook_weight: float = 1.0,
        pixelloss_weight: float = 1.0,
        perceptual_weight: float = 1.0,
    ):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_loss = LPIPS()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(
            input_channels=disc_in_channels,
            filters=disc_filters,
            n_layers=disc_num_layers,
        )

        self.discriminator_iter_start = disc_iter_start

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def call(
        self,
        y_true,
        y_pred,
        codebook_loss,
        optimizer_idx,
        global_step,
        cond=None,
        split="train",
    ):
        reconstruction_loss = tf.abs(y_true - y_pred)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(y_true, y_pred)
            reconstruction_loss += self.perceptual_weight * perceptual_loss
        else:
            perceptual_loss = 0.0

        neg_log_likelihood = tf.reduce_mean(reconstruction_loss)

        # # GAN part
        # if optimizer_idx == 0:
        #     if cond is None:
        #         assert not self.disc_conditional
        #         logits_fake = self.discriminator(y_pred)
        #     else:
        #         assert self.disc_conditional
        #         logits_fake = self.discriminator(tf.concat([y_pred, cond], axis=-1))
        #     g_loss = -tf.reduce_mean(logits_fake)
