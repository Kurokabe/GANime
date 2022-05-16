from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers


class NLayerDiscriminator(Model):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_channels: int = 3, filters: int = 64, n_layers: int = 3):
        super().__init__()

        kernel_size = 4
        sequence = [
            layers.Conv2D(filters, kernel_size=kernel_size, padding="same"),
            layers.LeakyReLU(alpha=0.2),
        ]

        filters_mult = 1
        for n in range(1, n_layers):
            filters_mult = min(2**n, 8)

            sequence += [
                layers.Conv2D(
                    filters * filters_mult,
                    kernel_size=kernel_size,
                    strides=2,
                    padding="same",
                    use_bias=False,
                ),
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=0.2),
            ]

        filters_mult = min(2**n_layers, 8)
        sequence += [
            layers.Conv2D(
                filters * filters_mult,
                kernel_size=kernel_size,
                strides=1,
                padding="same",
                use_bias=False,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
        ]

        sequence += [
            layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding="same")
        ]

        self.main = Sequential(sequence)

    def call(self, inputs, training=True, mask=None):
        return self.main(inputs)
