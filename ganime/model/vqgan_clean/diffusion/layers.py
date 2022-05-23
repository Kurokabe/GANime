import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow_addons.layers import GroupNormalization


@tf.keras.utils.register_keras_serializable()
class ResnetBlock(layers.Layer):
    def __init__(
        self,
        *,
        in_channels,
        dropout=0.0,
        out_channels=None,
        conv_shortcut=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.dropout_rate = dropout
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = GroupNormalization(groups=32, epsilon=1e-6)

        self.conv1 = layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same"
        )

        self.norm2 = GroupNormalization(groups=32, epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

        self.conv2 = layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same"
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = layers.Conv2D(
                    out_channels, kernel_size=3, strides=1, padding="same"
                )
            else:
                self.nin_shortcut = layers.Conv2D(
                    out_channels, kernel_size=1, strides=1, padding="valid"
                )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "dropout": self.dropout_rate,
                "out_channels": self.out_channels,
                "conv_shortcut": self.use_conv_shortcut,
            }
        )
        return config

    def call(self, x):
        h = x
        h = self.norm1(h)
        h = keras.activations.swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = keras.activations.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


@tf.keras.utils.register_keras_serializable()
class AttentionBlock(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.norm = GroupNormalization(groups=32, epsilon=1e-6)
        self.q = layers.Conv2D(channels, kernel_size=1, strides=1, padding="valid")
        self.k = layers.Conv2D(channels, kernel_size=1, strides=1, padding="valid")
        self.v = layers.Conv2D(channels, kernel_size=1, strides=1, padding="valid")
        self.proj_out = layers.Conv2D(
            channels, kernel_size=1, strides=1, padding="valid"
        )

        self.attention = layers.Attention()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
            }
        )
        return config

    def call(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        (
            b,
            h,
            w,
            c,
        ) = q.shape
        if b is None:
            b = -1
        q = tf.reshape(q, [b, h * w, c])
        k = tf.reshape(k, [b, h * w, c])
        v = tf.reshape(v, [b, h * w, c])

        h_ = self.attention([q, v, k])

        h_ = tf.reshape(h_, [b, h, w, c])

        h_ = self.proj_out(h_)

        return x + h_


@tf.keras.utils.register_keras_serializable()
class Downsample(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.down_sample = self.down_sample = layers.AveragePooling2D(
            pool_size=2, strides=2
        )
        self.conv = layers.Conv2D(channels, kernel_size=3, strides=1, padding="same")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
            }
        )
        return config

    def call(self, x):
        x = self.down_sample(x)
        x = self.conv(x)
        return x


@tf.keras.utils.register_keras_serializable()
class Upsample(layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.up_sample = layers.UpSampling2D(size=2, interpolation="nearest")
        self.conv = layers.Conv2D(channels, kernel_size=3, strides=1, padding="same")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
            }
        )
        return config

    def call(self, x):
        x = self.up_sample(x)
        x = self.conv(x)
        return x
