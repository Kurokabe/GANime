from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow_addons.layers import GroupNormalization

from .layers import AttentionBlock, ResnetBlock, Upsample


# @tf.keras.utils.register_keras_serializable()
class Decoder(layers.Layer):
    def __init__(
        self,
        *,
        channels: int,
        output_channels: int = 3,
        channels_multiplier: List[int],
        num_res_blocks: int,
        attention_resolution: List[int],
        resolution: int,
        z_channels: int,
        dropout: float,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.channels = channels
        self.output_channels = output_channels
        self.channels_multiplier = channels_multiplier
        self.num_resolutions = len(channels_multiplier)
        self.num_res_blocks = num_res_blocks
        self.attention_resolution = attention_resolution
        self.resolution = resolution
        self.z_channels = z_channels
        self.dropout = dropout

        block_in = channels * channels_multiplier[-1]
        current_resolution = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, current_resolution, current_resolution)

        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        self.conv_in = layers.Conv2D(block_in, kernel_size=3, strides=1, padding="same")

        # middle
        self.mid = {}
        self.mid["block_1"] = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid["attn_1"] = AttentionBlock(block_in)
        self.mid["block_2"] = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )

        # upsampling

        self.upsampling_list = []

        for i_level in reversed(range(self.num_resolutions)):
            block_out = channels * channels_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                self.upsampling_list.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                if current_resolution in attention_resolution:
                    # attentions.append(layers.Attention())
                    self.upsampling_list.append(AttentionBlock(block_in))

            if i_level != 0:
                self.upsampling_list.append(Upsample(block_in))
                current_resolution *= 2

        # end
        self.norm_out = GroupNormalization(groups=32, epsilon=1e-6)
        self.conv_out = layers.Conv2D(
            output_channels,
            kernel_size=3,
            strides=1,
            activation="sigmoid",
            padding="same",
        )

    def call(self, inputs, training=True, mask=None):

        h = self.conv_in(inputs)

        # middle
        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)

        for upsampling in self.upsampling_list:
            h = upsampling(h)

        # end
        h = self.norm_out(h)
        h = keras.activations.swish(h)
        h = self.conv_out(h)
        return h
