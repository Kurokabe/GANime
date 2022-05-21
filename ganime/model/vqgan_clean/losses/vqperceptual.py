from typing import List, Literal

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss

from .lpips import LPIPS


class PerceptualLoss(Loss):
    def __init__(self, *, perceptual_weight: float = 1.0, **kwargs):
        """Perceptual loss based on the LPIPS metric.

        Args:
            perceptual_weight (float, optional): The weight of the perceptual loss. Defaults to 1.0.
        """
        super().__init__(**kwargs)
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
