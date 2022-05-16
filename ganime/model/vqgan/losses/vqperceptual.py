from typing import List, Literal

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.losses import Loss

from ganime.model.vqgan.losses.lpips import LPIPS

from ..discriminator.model import NLayerDiscriminator


class VQLPIPSWithDiscriminator(Loss):
    def __init__(
        self, *, pixelloss_weight: float = 1.0, perceptual_weight: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_loss = LPIPS(reduction=tf.keras.losses.Reduction.NONE)
        self.perceptual_weight = perceptual_weight

    def call(
        self,
        y_true,
        y_pred,
    ):
        reconstruction_loss = tf.abs(y_true - y_pred)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(y_true, y_pred)
            reconstruction_loss += self.perceptual_weight * perceptual_loss
        else:
            perceptual_loss = 0.0

        neg_log_likelihood = tf.reduce_mean(reconstruction_loss)

        return neg_log_likelihood

        # # GAN part
        # if optimizer_idx == 0:
        #     if cond is None:
        #         assert not self.disc_conditional
        #         logits_fake = self.discriminator(y_pred)
        #     else:
        #         assert self.disc_conditional
        #         logits_fake = self.discriminator(tf.concat([y_pred, cond], axis=-1))
        #     g_loss = -tf.reduce_mean(logits_fake)
