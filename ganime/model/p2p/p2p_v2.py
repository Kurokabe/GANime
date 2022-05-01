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
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import KLDivergence, MeanSquaredError
from tqdm.auto import tqdm


class KLCriterion(Loss):
    def call(self, y_true, y_pred):
        (mu1, logvar1), (mu2, logvar2) = y_true, y_pred

        """KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2))"""
        sigma1 = tf.exp(tf.math.multiply(logvar1, 0.5))
        sigma2 = tf.exp(tf.math.multiply(logvar2, 0.5))

        kld = (
            tf.math.log(sigma2 / sigma1)
            + (tf.exp(logvar1) + tf.square(mu1 - mu2)) / (2 * tf.exp(logvar2))
            - 0.5
        )
        return kld


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
        beta: float = 0.0001,
        weight_align: float = 0.1,
        weight_cpc: float = 100,
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
        self.align_loss_tracker = tf.keras.metrics.Mean(name="align_loss")
        self.cpc_loss_tracker = tf.keras.metrics.Mean(name="align_loss")

        self.kl_loss = KLCriterion(
            reduction=tf.keras.losses.Reduction.NONE
        )  # KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        self.mse = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.align_loss = MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # self.optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        # )
        # self.prior_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        # )

    # region Model building
    def build_lstm(self):
        input = Input(shape=(None, self.g_dim + self.z_dim))
        embed = TimeDistributed(Dense(self.rnn_size))(input)
        lstm = LSTM(self.rnn_size)(embed)
        output = Dense(self.g_dim)(lstm)
        output = (tf.expand_dims(output, axis=1),)

        return Model(inputs=input, outputs=output, name="frame_predictor")

    def build_gaussian_lstm(self):

        input = Input(shape=(None, self.g_dim))
        embed = TimeDistributed(Dense(self.rnn_size))(input)
        lstm = LSTM(self.rnn_size)(embed)
        mu = Dense(self.z_dim)(lstm)
        logvar = Dense(self.z_dim)(lstm)
        z = Sampling()([mu, logvar])

        return Model(inputs=input, outputs=[mu, logvar, z])

    def build_encoder(self):

        input = Input(shape=(1, 64, 64, 1))

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
        output = Dense(self.g_dim)(h)
        output = tf.expand_dims(output, axis=1)
        return Model(inputs=input, outputs=output, name="encoder")
        # mu = Dense(self.g_dim)(h)
        # logvar = Dense(self.g_dim)(h)

        # z = Sampling()([mu, logvar])

        # mu, logvar, z = (
        #     tf.expand_dims(mu, axis=1),
        #     tf.expand_dims(logvar, axis=1),
        #     tf.expand_dims(z, axis=1),
        # )

        # return Model(inputs=input, outputs=[mu, logvar, z], name="encoder")

    def build_decoder(self):
        latent_inputs = Input(
            shape=(
                1,
                self.g_dim,
            )
        )
        x = Dense(1 * 1 * 1 * 512, activation="relu")(latent_inputs)
        x = Reshape((1, 1, 1, 512))(x)
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
            self.align_loss_tracker,
            self.cpc_loss_tracker,
        ]

    def call(self, inputs, training=None, mask=None):
        first_frame = inputs[:, 0:1, ...]
        last_frame = inputs[:, -1:, ...]

        desired_length = 20
        previous_frame = first_frame
        generated = [first_frame]

        z_last = self.encoder(last_frame)
        for i in range(1, desired_length):

            z_prev = self.encoder(previous_frame)
            prior_input = tf.concat([z_prev, z_last], axis=1)

            z_mean_prior, z_log_var_prior, z_prior = self.prior(prior_input)

            predictor_input = tf.concat(
                (z_prev, tf.expand_dims(z_prior, axis=1)), axis=-1
            )
            z_pred = self.frame_predictor(predictor_input)

            current_frame = self.decoder(z_pred)
            generated.append(current_frame)
            previous_frame = current_frame
        return tf.concat(generated, axis=1)

    def train_step(self, data):
        global_batch_size = 100  # * 8
        x, y = data

        first_frame = x[:, 0:1, ...]
        last_frame = x[:, -1:, ...]
        desired_length = y.shape[1]
        previous_frame = first_frame

        reconstruction_loss = 0
        kl_loss = 0
        align_loss = 0
        cpc_loss = 0

        with tf.GradientTape(persistent=True) as tape:
            z_last = self.encoder(last_frame)
            for i in tqdm(range(1, desired_length)):
                current_frame = y[:, i : i + 1, ...]
                z_prev = self.encoder(previous_frame)
                z_curr = self.encoder(current_frame)

                prior_input = tf.concat([z_prev, z_last], axis=1)
                posterior_input = tf.concat([z_curr, z_last], axis=1)

                z_mean_prior, z_log_var_prior, z_prior = self.prior(prior_input)
                z_mean_posterior, z_log_var_posterior, z_posterior = self.posterior(
                    posterior_input
                )

                # predictor_input = z_prev
                predictor_input = tf.concat(
                    (z_prev, tf.expand_dims(z_posterior, axis=1)), axis=-1
                )

                z_pred = self.frame_predictor(predictor_input)

                kl_loss += tf.reduce_sum(
                    self.kl_loss(
                        (z_mean_prior, z_log_var_prior),
                        (z_mean_posterior, z_log_var_posterior),
                    )
                ) * (1.0 / global_batch_size)

                align_loss += tf.reduce_sum(self.align_loss(z_pred, z_curr)) * (
                    1.0 / global_batch_size
                )

                if i == desired_length - 1:
                    h_pred_p = self.frame_predictor(
                        tf.concat([z_prev, tf.expand_dims(z_prior, axis=1)], axis=-1)
                    )
                    x_pred_p = self.decoder(h_pred_p)
                    cpc_loss += tf.reduce_sum(self.mse(x_pred_p, current_frame)) * (
                        1.0 / global_batch_size
                    )

                prediction = self.decoder(z_pred)
                reconstruction_loss += tf.reduce_sum(
                    self.mse(prediction, current_frame)
                ) * (1.0 / global_batch_size)

                previous_frame = current_frame

            loss = (
                reconstruction_loss
                + kl_loss * self.beta
                + align_loss * self.weight_align
                + cpc_loss * self.weight_cpc
            )

            prior_loss = kl_loss + cpc_loss * self.weight_cpc

        grads_without_prior = tape.gradient(
            loss,
            (
                self.encoder.trainable_weights
                + self.decoder.trainable_weights
                + self.posterior.trainable_weights
                + self.frame_predictor.trainable_weights
            ),
        )
        self.optimizer.apply_gradients(
            zip(
                grads_without_prior,
                (
                    self.encoder.trainable_weights
                    + self.decoder.trainable_weights
                    + self.posterior.trainable_weights
                    + self.frame_predictor.trainable_weights
                ),
            )
        )

        grads_prior = tape.gradient(
            prior_loss,
            self.prior.trainable_weights,
        )

        self.optimizer.apply_gradients(
            zip(
                grads_prior,
                self.prior.trainable_weights,
            )
        )
        del tape

        self.total_loss_tracker.update_state(loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.align_loss_tracker.update_state(align_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.cpc_loss_tracker.update_state(cpc_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "align_loss": self.align_loss_tracker.result(),
            "cpc_loss": self.cpc_loss_tracker.result(),
        }

        # print("KL_LOSS")
        # print(kl_loss)
        # print("ALIGN_LOSS")
        # print(align_loss)
        # print("RECONSTRUCTION_LOSS")
        # print(reconstruction_loss)

        # with tf.GradientTape() as tape:
        #     z_mean, z_log_var, z = self.encoder(x)
        #     reconstruction = self.decoder(z)
        #     reconstruction_loss = tf.reduce_mean(
        #         tf.reduce_sum(
        #             tf.keras.losses.binary_crossentropy(y, reconstruction),
        #             axis=(1, 2),
        #         )
        #     )
        #     kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        #     total_loss = reconstruction_loss + self.kl_beta * kl_loss
        # grads = tape.gradient(total_loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # self.total_loss_tracker.update_state(total_loss)
        # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.kl_loss_tracker.update_state(kl_loss)
        # return {
        #     "loss": self.total_loss_tracker.result(),
        #     "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        #     "kl_loss": self.kl_loss_tracker.result(),
        # }

    # def test_step(self, data):
    #     if isinstance(data, tuple):
    #         data = data[0]

    #     z_mean, z_log_var, z = self.encoder(data)
    #     reconstruction = self.decoder(z)
    #     reconstruction_loss = tf.reduce_mean(
    #         tf.keras.losses.binary_crossentropy(data, reconstruction)
    #     )
    #     reconstruction_loss *= 28 * 28
    #     kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    #     kl_loss = tf.reduce_mean(kl_loss)
    #     kl_loss *= -0.5
    #     total_loss = reconstruction_loss + kl_loss
    #     return {
    #         "loss": total_loss,
    #         "reconstruction_loss": reconstruction_loss,
    #         "kl_loss": kl_loss,
    #     }
