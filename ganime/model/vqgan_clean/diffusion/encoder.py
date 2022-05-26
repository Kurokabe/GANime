from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow_addons.layers import GroupNormalization
from .layers import ResnetBlock, AttentionBlock, Downsample


# @tf.keras.utils.register_keras_serializable()
class Encoder(layers.Layer):
    def __init__(
        self,
        *,
        channels: int,
        channels_multiplier: List[int],
        num_res_blocks: int,
        attention_resolution: List[int],
        resolution: int,
        z_channels: int,
        dropout: float,
        **kwargs
    ):
        """Encode an image into a latent vector. The encoder will be constitued of multiple levels (lenght of `channels_multiplier`) with for each level `num_res_blocks` ResnetBlock.

        Args:
            channels (int, optional): The number of channel for the first layer. Defaults to 128.
            channels_multiplier (List[int], optional): The channel multiplier for each level (previous level channels X multipler). Defaults to [1, 1, 2, 2].
            num_res_blocks (int, optional): Number of ResnetBlock at each level. Defaults to 1.
            attention_resolution (List[int], optional): Add an attention block if the current resolution is in this array. Defaults to [16].
            resolution (int, optional): The starting resolution. Defaults to 64.
            z_channels (int, optional): The number of channel at the end of the encoder. Defaults to 128.
            dropout (float, optional): The dropout ratio for each ResnetBlock. Defaults to 0.0.
        """
        super().__init__(**kwargs)

        self.channels = channels
        self.channels_multiplier = channels_multiplier
        self.num_resolutions = len(channels_multiplier)
        self.num_res_blocks = num_res_blocks
        self.attention_resolution = attention_resolution
        self.resolution = resolution
        self.z_channels = z_channels
        self.dropout = dropout

        self.conv_in = layers.Conv2D(
            self.channels, kernel_size=3, strides=1, padding="same"
        )

        current_resolution = resolution

        in_channels_multiplier = (1,) + tuple(channels_multiplier)

        self.downsampling_list = []

        for i_level in range(self.num_resolutions):
            block_in = channels * in_channels_multiplier[i_level]
            block_out = channels * channels_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                self.downsampling_list.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                if current_resolution in attention_resolution:
                    # attentions.append(layers.Attention())
                    self.downsampling_list.append(AttentionBlock(block_in))
                    current_resolution = current_resolution // 2

            if i_level != self.num_resolutions - 1:
                self.downsampling_list.append(Downsample(block_in))

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

        # end
        self.norm_out = GroupNormalization(groups=32, epsilon=1e-6)
        self.conv_out = layers.Conv2D(
            z_channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )

    # def get_config(self):
    #     config = super().get_config()
    #     config.update(
    #         {
    #             "channels": self.channels,
    #             "channels_multiplier": self.channels_multiplier,
    #             "num_res_blocks": self.num_res_blocks,
    #             "attention_resolution": self.attention_resolution,
    #             "resolution": self.resolution,
    #             "z_channels": self.z_channels,
    #             "dropout": self.dropout,
    #         }
    #     )
    #     return config

    def call(self, inputs, training=True, mask=None):
        h = self.conv_in(inputs)
        for downsampling in self.downsampling_list:
            h = downsampling(h)

        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)

        # end
        h = self.norm_out(h)
        h = keras.activations.swish(h)
        h = self.conv_out(h)
        return h
