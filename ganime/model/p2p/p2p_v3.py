import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    Dense,
    Flatten,
    Input,
    Layer,
    LeakyReLU,
    MaxPooling2D,
    Reshape,
    TimeDistributed,
    UpSampling2D,
)


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class P2P(Model):
    def __init__(
        self,
        channels: int = 1,
        g_dim: int = 128,
        z_dim: int = 10,
        rnn_size: int = 256,
        prior_rnn_layers: int = 1,
        posterior_rnn_layers: int = 1,
        predictor_rnn_layers: float = 1,
        skip_prob: float = 0.1,
        n_past: int = 1,
        last_frame_skip: bool = False,
        beta: float = 0.9,
        weight_align: float = 0.0,
        weight_cpc: float = 1000.0,
        kl_beta=5.0,
    ):
        super().__init__()
        # Models parameters
        self.channels = channels
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.rnn_size = rnn_size
        self.prior_rnn_layers = prior_rnn_layers
        self.posterior_rnn_layers = posterior_rnn_layers
        self.predictor_rnn_layers = predictor_rnn_layers

        # Training parameters
        self.skip_prob = skip_prob
        self.n_past = n_past
        self.last_frame_skip = last_frame_skip
        self.beta = beta
        self.weight_align = weight_align
        self.weight_cpc = weight_cpc
        self.kl_beta = kl_beta

        self.frame_predictor = self.build_lstm()
        self.prior = self.build_gaussian_lstm()
        self.posterior = self.build_gaussian_lstm()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    # region Model building
    def build_lstm(self):
        input = Input(shape=(20, self.g_dim + self.z_dim + 1))
        embed = TimeDistributed(Dense(self.rnn_size))(input)
        lstm = LSTM(self.rnn_size, return_sequences=True)(embed)
        output = TimeDistributed(Dense(self.g_dim))(lstm)

        return Model(inputs=input, outputs=output, name="frame_predictor")

    def build_gaussian_lstm(self):

        input = Input(shape=(20, self.g_dim + self.g_dim + 1))
        embed = TimeDistributed(Dense(self.rnn_size))(input)
        lstm = LSTM(self.rnn_size, return_sequences=True)(embed)
        mu = TimeDistributed(Dense(self.z_dim))(lstm)
        logvar = TimeDistributed(Dense(self.z_dim))(lstm)
        z = TimeDistributed(Sampling())([mu, logvar])

        return Model(inputs=input, outputs=[mu, logvar, z])

    def build_encoder(self):

        input = Input(shape=(2, 64, 64, 1))

        h = TimeDistributed(Conv2D(64, kernel_size=4, strides=2, padding="same"))(input)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, padding="same"))(h)

        h = TimeDistributed(Conv2D(128, kernel_size=4, strides=2, padding="same"))(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, padding="same"))(h)

        h = TimeDistributed(Conv2D(256, kernel_size=4, strides=2, padding="same"))(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, padding="same"))(h)

        h = TimeDistributed(Conv2D(512, kernel_size=4, strides=2, padding="same"))(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = TimeDistributed(MaxPooling2D(pool_size=2, strides=2, padding="same"))(h)

        h = Flatten()(h)
        mu = Dense(self.g_dim)(h)
        logvar = Dense(self.g_dim)(h)

        z = Sampling()([mu, logvar])

        return Model(inputs=input, outputs=[mu, logvar, z], name="encoder")

    def build_decoder(self):
        latent_inputs = Input(shape=(self.g_dim,))
        x = Dense(20 * 1 * 1 * 512, activation="relu")(latent_inputs)
        x = Reshape((20, 1, 1, 512))(x)
        x = TimeDistributed(
            Conv2DTranspose(512, kernel_size=4, strides=1, padding="valid")
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = TimeDistributed(
            Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = TimeDistributed(
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = TimeDistributed(
            Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = TimeDistributed(
            Conv2DTranspose(1, kernel_size=4, strides=2, padding="same")
        )(x)
        x = Activation("sigmoid")(x)

        return Model(inputs=latent_inputs, outputs=x, name="decoder")

    # endregion

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(y, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.kl_beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(data, reconstruction)
        )
        reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
