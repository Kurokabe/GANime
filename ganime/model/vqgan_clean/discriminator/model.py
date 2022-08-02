from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal


class NLayerDiscriminator(Model):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, filters: int = 64, n_layers: int = 3, **kwargs):
        super().__init__(**kwargs)

        init = RandomNormal(stddev=0.02)
        self.filters = filters
        self.n_layers = n_layers

        kernel_size = 4
        self.sequence = [
            # layers.AveragePooling2D(pool_size=2),
            layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=2,
                # strides=1,
                padding="same",
                kernel_initializer=init,
            ),
            layers.LeakyReLU(alpha=0.2),
        ]

        filters_mult = 1
        for n in range(1, n_layers):
            filters_mult = min(2**n, 8)

            self.sequence += [
                # layers.AveragePooling2D(pool_size=2),
                layers.Conv2D(
                    filters * filters_mult,
                    kernel_size=kernel_size,
                    # strides=1,  # 2,
                    strides=2,
                    padding="same",
                    use_bias=False,
                    kernel_initializer=init,
                ),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),
            ]

        filters_mult = min(2**n_layers, 8)
        self.sequence += [
            # layers.AveragePooling2D(pool_size=2),
            layers.Conv2D(
                filters * filters_mult,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                use_bias=False,
                kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
        ]

        self.sequence += [
            layers.Conv2D(
                1,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                # activation="sigmoid",
                kernel_initializer=init,
            ),
        ]

    def call(self, inputs, training=True, mask=None):
        h = inputs
        for seq in self.sequence:
            h = seq(h)
        return h

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "n_layers": self.n_layers,
            }
        )
        return config
