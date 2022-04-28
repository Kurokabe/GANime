import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    LeakyReLU,
    TimeDistributed,
)
from tensorflow.keras.losses import Loss


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
        return tf.reduce_sum(kld) / 20


class Encoder(Model):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        self.c1 = Sequential(
            [
                TimeDistributed(Conv2D(64, kernel_size=4, strides=2, padding="same")),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.c2 = Sequential(
            [
                TimeDistributed(Conv2D(128, kernel_size=4, strides=2, padding="same")),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.c3 = Sequential(
            [
                TimeDistributed(Conv2D(256, kernel_size=4, strides=2, padding="same")),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.c4 = Sequential(
            [
                TimeDistributed(Conv2D(512, kernel_size=4, strides=2, padding="same")),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.c5 = Sequential(
            [
                TimeDistributed(
                    Conv2D(self.dim, kernel_size=4, strides=1, padding="valid")
                ),
                BatchNormalization(),
                Activation("tanh"),
            ]
        )

    def call(self, input):
        sequence_length = input.shape[1]
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return tf.reshape(h5, (-1, sequence_length, self.dim)), [h1, h2, h3, h4, h5]


class Decoder(Model):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        self.upc1 = Sequential(
            [
                TimeDistributed(
                    Conv2DTranspose(512, kernel_size=4, strides=1, padding="valid")
                ),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.upc2 = Sequential(
            [
                TimeDistributed(
                    Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")
                ),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.upc3 = Sequential(
            [
                TimeDistributed(
                    Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")
                ),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.upc4 = Sequential(
            [
                TimeDistributed(
                    Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")
                ),
                BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.upc5 = Sequential(
            [
                TimeDistributed(
                    Conv2DTranspose(1, kernel_size=4, strides=2, padding="same")
                ),
                Activation("sigmoid"),
            ]
        )

    def call(self, input):
        vec, skip = input
        # TODO change the sequence_length
        sequence_length = 20 - 1
        d1 = self.upc1(tf.reshape(vec, (-1, sequence_length, 1, 1, self.dim)))
        d2 = self.upc2(tf.concat([d1, skip[3]], axis=-1))
        d3 = self.upc3(tf.concat([d2, skip[2]], axis=-1))
        d4 = self.upc4(tf.concat([d3, skip[1]], axis=-1))
        output = self.upc5(tf.concat([d4, skip[0]], axis=-1))
        return output


class MyLSTM(Model):
    def __init__(self, input_shape, hidden_size, output_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = TimeDistributed(Dense(hidden_size, input_dim=input_shape))
        self.lstm = LSTM(hidden_size, return_sequences=True)
        self.out = TimeDistributed(Dense(output_size))

    def call(self, inputs):
        h_in = self.embed(inputs)
        h_out = self.lstm(h_in)
        return self.out(h_out)


class MyGaussianLSTM(Model):
    def __init__(self, input_shape, hidden_size, output_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = TimeDistributed(Dense(hidden_size, input_dim=input_shape))
        self.lstm = LSTM(hidden_size, return_sequences=True)
        self.mu_net = TimeDistributed(Dense(output_size))
        self.logvar_net = TimeDistributed(Dense(output_size))

    def reparameterize(self, mu, logvar: tf.Tensor):
        logvar = tf.math.exp(logvar * 0.5)
        eps = tf.random.normal(logvar.shape)
        return tf.add(tf.math.multiply(eps, logvar), mu)

    def call(self, inputs):
        h_in = self.embed(inputs)
        h_in = self.lstm(h_in)
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


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

        self.frame_predictor = MyLSTM(
            self.g_dim + self.z_dim + 1 + 1,
            self.rnn_size,
            self.g_dim,
            self.predictor_rnn_layers,
        )

        self.prior = MyGaussianLSTM(
            self.g_dim + self.g_dim + 1 + 1,
            self.rnn_size,
            self.z_dim,
            self.prior_rnn_layers,
        )

        self.posterior = MyGaussianLSTM(
            self.g_dim + self.g_dim + 1 + 1,
            self.rnn_size,
            self.z_dim,
            self.posterior_rnn_layers,
        )

        self.encoder = Encoder(self.g_dim, self.channels)
        self.decoder = Decoder(self.g_dim, self.channels)

        # criterions
        self.mse_criterion = tf.keras.losses.MeanSquaredError()
        self.kl_criterion = KLCriterion()
        self.align_criterion = tf.keras.losses.MeanSquaredError()

        # optimizers
        self.frame_predictor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )
        self.posterior_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )
        self.prior_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )
        self.encoder_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )
        self.decoder_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )

    def get_time_until_control_point(self, batch_size, sequence_length):
        nx, ny = (sequence_length - 1, batch_size)

        # Create array of [batch_size, sequence_length] with value ranging from 0 to 1
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xv, yv = np.meshgrid(x, y)

        xv = np.expand_dims(xv, axis=-1)
        return tf.convert_to_tensor(
            1 - xv, dtype=tf.float32
        )  # We want value going from 1 to 0, hence the 1 - xv

    def call(self, x):
        batch_size, sequence_length, height, width, channels = x.shape

        with tf.GradientTape(persistent=True) as tape:
            encoded_last_frame = self.encoder(x[:, -1:, ...])[0]
            encoded_last_frame = np.concatenate(
                [encoded_last_frame for _ in range(19)], axis=1
            )  # concat to have array of shape [batch_size, sequence_length - 1, n_features]

            time_until_control_point = self.get_time_until_control_point(
                batch_size=batch_size, sequence_length=sequence_length
            )

            # TODO change the delta time
            delta_time = tf.fill(
                [batch_size, sequence_length - 1], 1.0 / sequence_length
            )
            delta_time = np.expand_dims(delta_time, axis=-1)

            h, skip = self.encoder(x[:, 0:-1, ...])
            h_target = self.encoder(x[:, 1:, ...])[0]

            h_control_point_aware = tf.concat(
                [h, encoded_last_frame, time_until_control_point, delta_time], axis=2
            )
            h_target_control_point_aware = tf.concat(
                [h_target, encoded_last_frame, time_until_control_point, delta_time],
                axis=2,
            )

            zt, mu, logvar = self.posterior(h_control_point_aware)
            zt_p, mu_p, logvar_p = self.posterior(h_target_control_point_aware)

            concat = tf.concat([h, zt, time_until_control_point, delta_time], axis=2)
            h_pred = self.frame_predictor(concat)
            x_pred = self.decoder([h_pred, skip])

            # TODO handle prediction on last frame (cpc_loss)

            align_loss = self.align_criterion(h, h_pred)
            mse_loss = self.mse_criterion(x_pred, x[:, 1:, ...])
            kld_loss = self.kl_criterion((mu, logvar), (mu_p, logvar_p))
            loss = mse_loss + kld_loss * self.beta + align_loss * self.weight_align
            loss_prior = kld_loss  # + cpc_loss * self.weight_cpc

        var_list_without_prior = (
            self.frame_predictor.trainable_variables
            + self.posterior.trainable_variables
            + self.encoder.trainable_variables
            + self.decoder.trainable_variables
        )

        var_list_prior = self.prior.trainable_variables

        gradients_without_prior = tape.gradient(loss, var_list_without_prior)

        gradients_prior = tape.gradient(
            loss_prior,
            var_list_prior,
        )

        self.update_model_without_prior(
            gradients_without_prior,
            var_list_without_prior,
        )
        self.update_prior(gradients_prior, var_list_prior)
        del tape

        return mse_loss, kld_loss, loss_prior, align_loss

    def update_model_without_prior(self, gradients, var_list):
        self.frame_predictor_optimizer.apply_gradients(zip(gradients, var_list))
        self.posterior_optimizer.apply_gradients(zip(gradients, var_list))
        self.encoder_optimizer.apply_gradients(zip(gradients, var_list))
        self.decoder_optimizer.apply_gradients(zip(gradients, var_list))

    def update_prior(self, gradients, var_list):
        self.prior_optimizer.apply_gradients(zip(gradients, var_list))
