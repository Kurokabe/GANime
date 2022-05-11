from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow_addons.layers import GroupNormalization


INPUT_SHAPE = (64, 64, 3)
ENCODER_OUTPUT_SHAPE = (8, 8, 128)


class VQGAN(keras.Model):
    def __init__(
        self,
        train_variance: float,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        z_channels: int = 128,  # 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_variance = train_variance

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantize = VectorQuantizer(num_embeddings, embedding_dim, beta=beta)

        self.quant_conv = layers.Conv2D(embedding_dim, kernel_size=1)
        self.post_quant_conv = layers.Conv2D(z_channels, kernel_size=1)

        self.vqvae = self.get_vqvae()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

    def get_vqvae(self):
        inputs = keras.Input(shape=INPUT_SHAPE)
        quant = self.encode(inputs)
        reconstructed = self.decode(quant)
        return keras.Model(inputs, reconstructed, name="vq_vae")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantize(h)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def call(self, inputs, training=True, mask=None):
        return self.vqvae(inputs)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            reconstructions = self(x, training=True)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((y - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class Encoder(Model):
    def __init__(
        self,
        *,
        channels: int = 128,
        output_channels: int = 3,
        channels_multiplier: List[int] = [1, 1, 2, 2],  # [1, 1, 2, 2, 4],
        num_res_blocks: int = 1,  # 2,
        attention_resolution: List[int] = [16],
        resolution: int = 64,  # 256,
        z_channels=128,  # 256,
        dropout=0.0,
        double_z=False,
        resamp_with_conv=True,
    ):
        super().__init__()

        self.channels = channels
        self.timestep_embeddings_channel = 0
        self.num_resolutions = len(channels_multiplier)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.conv_in = layers.Conv2D(
            self.channels, kernel_size=3, strides=1, padding="same"
        )

        current_resolution = resolution

        in_channels_multiplier = (1,) + tuple(channels_multiplier)
        self.downsampling = []

        for i_level in range(self.num_resolutions):
            block = []
            attentions = []
            block_in = channels * in_channels_multiplier[i_level]
            block_out = channels * channels_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        timestep_embedding_channels=self.timestep_embeddings_channel,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                if current_resolution in attention_resolution:
                    # attentions.append(layers.Attention())
                    attentions.append(AttentionBlock(block_in))

            down = {}
            down["block"] = block
            down["attention"] = attentions
            if i_level != self.num_resolutions - 1:
                down["downsample"] = Downsample(block_in, resamp_with_conv)
            self.downsampling.append(down)

        # middle
        self.mid = {}
        self.mid["block_1"] = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            timestep_embedding_channels=self.timestep_embeddings_channel,
            dropout=dropout,
        )
        self.mid["attn_1"] = AttentionBlock(block_in)
        self.mid["block_2"] = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            timestep_embedding_channels=self.timestep_embeddings_channel,
            dropout=dropout,
        )

        # end
        self.norm_out = GroupNormalization(groups=32, epsilon=1e-6)
        self.conv_out = layers.Conv2D(
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            strides=1,
            padding="same",
        )

    def summary(self):
        x = layers.Input(shape=INPUT_SHAPE)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def call(self, inputs, training=True, mask=None):
        hs = [self.conv_in(inputs)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.downsampling[i_level]["block"][i_block](hs[-1])
                if len(self.downsampling[i_level]["attention"]) > 0:
                    h = self.downsampling[i_level]["attention"][i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.downsampling[i_level]["downsample"](hs[-1]))

        h = hs[-1]
        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)

        # end
        h = self.norm_out(h)
        h = keras.activations.swish(h)
        h = self.conv_out(h)
        return h


class Decoder(Model):
    def __init__(
        self,
        *,
        channels: int = 128,
        output_channels: int = 3,
        channels_multiplier: List[int] = [1, 1, 2, 2],  # [1, 1, 2, 2, 4],
        num_res_blocks: int = 1,  # 2,
        attention_resolution: List[int] = [16],
        resolution: int = 64,  # 256,
        z_channels=128,  # 256,
        dropout=0.0,
        give_pre_end=False,
        resamp_with_conv=True,
    ):
        super().__init__()

        self.channels = channels
        self.timestep_embeddings_channel = 0
        self.num_resolutions = len(channels_multiplier)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        in_channels_multiplier = (1,) + tuple(channels_multiplier)
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
            timestep_embedding_channels=self.timestep_embeddings_channel,
            dropout=dropout,
        )
        self.mid["attn_1"] = AttentionBlock(block_in)
        self.mid["block_2"] = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            timestep_embedding_channels=self.timestep_embeddings_channel,
            dropout=dropout,
        )

        # upsampling
        self.upsampling = []

        for i_level in reversed(range(self.num_resolutions)):
            block = []
            attentions = []
            block_out = channels * channels_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        timestep_embedding_channels=self.timestep_embeddings_channel,
                        dropout=dropout,
                    )
                )
                block_in = block_out

                if current_resolution in attention_resolution:
                    # attentions.append(layers.Attention())
                    attentions.append(AttentionBlock(block_in))

            upsampling = {}
            upsampling["block"] = block
            upsampling["attention"] = attentions
            if i_level != 0:
                upsampling["upsample"] = Upsample(block_in, resamp_with_conv)
                current_resolution *= 2
            self.upsampling.insert(0, upsampling)

        # end
        self.norm_out = GroupNormalization(groups=32, epsilon=1e-6)
        self.conv_out = layers.Conv2D(
            output_channels,
            kernel_size=3,
            strides=1,
            activation="sigmoid",
            padding="same",
        )

    def summary(self):
        x = layers.Input(shape=ENCODER_OUTPUT_SHAPE)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def call(self, inputs, training=True, mask=None):

        self.laz_z_shape = inputs.shape

        h = self.conv_in(inputs)

        # middle
        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.upsampling[i_level]["block"][i_block](h)
                if len(self.upsampling[i_level]["attention"]) > 0:
                    h = self.upsampling[i_level]["attention"][i_block](h)
            if i_level != 0:
                h = self.upsampling[i_level]["upsample"](h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = keras.activations.swish(h)
        h = self.conv_out(h)
        return h


class ResnetBlock(layers.Layer):
    def __init__(
        self,
        *,
        in_channels,
        dropout=0.0,
        out_channels=None,
        conv_shortcut=False,
        timestep_embedding_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = GroupNormalization(groups=32, epsilon=1e-6)

        self.conv1 = layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same"
        )

        if timestep_embedding_channels > 0:
            self.timestep_embedding_projection = layers.Dense(out_channels)

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

    def call(self, x):
        h = x
        h = self.norm1(h)
        h = keras.activations.swish(h)
        h = self.conv1(h)

        # if timestamp_embedding is not None:
        #     h = h + self.timestep_embedding_projection(keras.activations.swish(timestamp_embedding))

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


class AttentionBlock(layers.Layer):
    def __init__(self, channels):
        super().__init__()

        self.norm = GroupNormalization(groups=32, epsilon=1e-6)
        self.q = layers.Conv2D(channels, kernel_size=1, strides=1, padding="valid")
        self.k = layers.Conv2D(channels, kernel_size=1, strides=1, padding="valid")
        self.v = layers.Conv2D(channels, kernel_size=1, strides=1, padding="valid")
        self.proj_out = layers.Conv2D(
            channels, kernel_size=1, strides=1, padding="valid"
        )

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
        w_ = tf.matmul(
            q, k, transpose_b=True
        )  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = keras.activations.softmax(w_)

        # attend to values
        v = tf.reshape(v, [b, h * w, c])
        # w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = tf.matmul(
            v, w_, transpose_a=True
        )  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # h_ = h_.reshape(b, c, h, w)
        h_ = tf.reshape(h_, [b, h, w, c])

        h_ = self.proj_out(h_)

        return x + h_


class Downsample(layers.Layer):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.down_sample = layers.Conv2D(
                channels, kernel_size=3, strides=2, padding="same"
            )
        else:
            self.down_sample = layers.AveragePooling2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.down_sample(x)
        return x


class Upsample(layers.Layer):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.up_sample = layers.Conv2DTranspose(
                channels, kernel_size=3, strides=2, padding="same"
            )
        else:
            self.up_sample = layers.UpSampling2D(size=2, interpolation="nearest")

    def call(self, x):
        x = self.up_sample(x)
        return x