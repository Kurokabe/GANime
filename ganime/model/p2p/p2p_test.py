from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    LSTMCell,
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

# from tensorflow_probability.python.layers.dense_variational import (
#     DenseReparameterization,
# )
# import tensorflow_probability as tfp
from tensorflow.keras.losses import Loss

initializer_conv_dense = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
initializer_batch_norm = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)


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
        return tf.reduce_sum(kld) / 100


class Encoder(Model):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        self.c1 = Sequential(
            [
                Conv2D(
                    64,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.c2 = Sequential(
            [
                Conv2D(
                    128,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.c3 = Sequential(
            [
                Conv2D(
                    256,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.c4 = Sequential(
            [
                Conv2D(
                    512,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.c5 = Sequential(
            [
                Conv2D(
                    self.dim,
                    kernel_size=4,
                    strides=1,
                    padding="valid",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                Activation("tanh"),
            ]
        )

    def call(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return tf.reshape(h5, (-1, self.dim)), [h1, h2, h3, h4, h5]


class Decoder(Model):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        self.upc1 = Sequential(
            [
                Conv2DTranspose(
                    512,
                    kernel_size=4,
                    strides=1,
                    padding="valid",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.upc2 = Sequential(
            [
                Conv2DTranspose(
                    256,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.upc3 = Sequential(
            [
                Conv2DTranspose(
                    128,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.upc4 = Sequential(
            [
                Conv2DTranspose(
                    64,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer_conv_dense,
                ),
                # BatchNormalization(),
                LeakyReLU(alpha=0.2),
            ]
        )
        self.upc5 = Sequential(
            [
                Conv2DTranspose(
                    1,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    kernel_initializer=initializer_conv_dense,
                ),
                Activation("sigmoid"),
            ]
        )

    def call(self, input):
        vec, skip = input
        d1 = self.upc1(tf.reshape(vec, (-1, 1, 1, self.dim)))
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
        self.embed = Dense(
            hidden_size,
            input_dim=input_shape,
            kernel_initializer=initializer_conv_dense,
        )
        # self.lstm = Sequential(
        #     [LSTMCell(hidden_size) for _ in range(n_layers)], name="lstm"
        # )
        # self.lstm = self.create_lstm(hidden_size, n_layers)
        self.lstm = [
            LSTMCell(
                hidden_size  # , return_sequences=False if i == self.n_layers - 1 else True
            )
            for i in range(self.n_layers)
        ]  # LSTMCell(hidden_size)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm[0], return_state=True)
        self.out = Dense(output_size, kernel_initializer=initializer_conv_dense)

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append(
                (
                    tf.Variable(tf.zeros([batch_size, self.hidden_size])),
                    tf.Variable(tf.zeros([batch_size, self.hidden_size])),
                )
            )
        self.__dict__["hidden"] = hidden

    def build(self, input_shape):
        self.init_hidden(input_shape[0])

    def call(self, inputs):
        h_in = self.embed(inputs)
        h_in = tf.reshape(h_in, (-1, 1, self.hidden_size))
        h_in, *state = self.lstm_rnn(h_in)
        for i in range(self.n_layers):
            h_in, state = self.lstm[i](h_in, state)
        return self.out(h_in)


class MyGaussianLSTM(Model):
    def __init__(self, input_shape, hidden_size, output_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = Dense(
            hidden_size,
            input_dim=input_shape,
            kernel_initializer=initializer_conv_dense,
        )
        # self.lstm = Sequential(
        #     [LSTMCell(hidden_size) for _ in range(n_layers)], name="lstm"
        # )
        self.lstm = [
            LSTMCell(
                hidden_size  # , return_sequences=False if i == self.n_layers - 1 else True
            )
            for i in range(self.n_layers)
        ]  # LSTMCell(hidden_size)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm[0], return_state=True)
        self.mu_net = Dense(output_size, kernel_initializer=initializer_conv_dense)
        self.logvar_net = Dense(output_size, kernel_initializer=initializer_conv_dense)
        # self.out = Sequential(
        #     [
        #         tf.keras.layers.Dense(
        #             tfp.layers.MultivariateNormalTriL.params_size(output_size),
        #             activation=None,
        #         ),
        #         tfp.layers.MultivariateNormalTriL(output_size),
        #     ]
        # )

    def reparameterize(self, mu, logvar: tf.Tensor):
        logvar = tf.math.exp(logvar * 0.5)
        eps = tf.random.normal(logvar.shape)
        return tf.add(tf.math.multiply(eps, logvar), mu)

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append(
                (
                    tf.Variable(tf.zeros([batch_size, self.hidden_size])),
                    tf.Variable(tf.zeros([batch_size, self.hidden_size])),
                )
            )
        self.__dict__["hidden"] = hidden

    def build(self, input_shape):
        self.init_hidden(input_shape[0])

    def call(self, inputs):
        h_in = self.embed(inputs)
        # for i in range(self.n_layers):
        #     # print(h_in.shape, self.hidden[i][0].shape, self.hidden[i][0].shape)

        #     _, self.hidden[i] = self.lstm(h_in, self.hidden[i])
        #     h_in = self.hidden[i][0]
        h_in = tf.reshape(h_in, (-1, 1, self.hidden_size))
        h_in, *state = self.lstm_rnn(h_in)
        for i in range(self.n_layers):
            h_in, state = self.lstm[i](h_in, state)

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
        predictor_rnn_layers: float = 2,
        skip_prob: float = 0.5,
        n_past: int = 1,
        last_frame_skip: bool = False,
        beta: float = 0.0001,
        weight_align: float = 0.1,
        weight_cpc: float = 100,
    ):
        super().__init__()
        self.channels = channels
        self.g_dim = g_dim
        self.z_dim = z_dim
        self.rnn_size = rnn_size
        self.prior_rnn_layers = prior_rnn_layers
        self.posterior_rnn_layers = posterior_rnn_layers
        self.predictor_rnn_layers = predictor_rnn_layers

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

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.align_loss_tracker = tf.keras.metrics.Mean(name="align_loss")
        self.cpc_loss_tracker = tf.keras.metrics.Mean(name="align_loss")

        # optimizers
        # self.frame_predictor_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        # )
        # self.posterior_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        # )
        # self.prior_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        # )
        # self.encoder_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        # )
        # self.decoder_optimizer = tf.keras.optimizers.Adam(
        #     learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        # )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.align_loss_tracker,
            self.cpc_loss_tracker,
        ]

    def get_global_descriptor(self, x, start_ix=0, cp_ix=None):
        """Get the global descriptor based on x, start_ix, cp_ix."""
        if cp_ix is None:
            cp_ix = x.shape[1] - 1

        x_cp = x[:, cp_ix, ...]
        h_cp = self.encoder(x_cp)[0]  # 1 is input for skip-connection

        return x_cp, h_cp

    def compile(
        self,
        frame_predictor_optimizer,
        prior_optimizer,
        posterior_optimizer,
        encoder_optimizer,
        decoder_optimizer,
    ):
        super().compile()
        self.frame_predictor_optimizer = frame_predictor_optimizer
        self.prior_optimizer = prior_optimizer
        self.posterior_optimizer = posterior_optimizer
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

    def train_step(self, data):
        y, x = data
        batch_size = 100

        mse_loss = 0
        kld_loss = 0
        cpc_loss = 0
        align_loss = 0

        seq_len = x.shape[1]
        start_ix = 0
        cp_ix = seq_len - 1
        x_cp, global_z = self.get_global_descriptor(
            x, start_ix, cp_ix
        )  # here global_z is h_cp

        skip_prob = self.skip_prob

        prev_i = 0
        max_skip_count = seq_len * skip_prob
        skip_count = 0
        probs = np.random.uniform(low=0, high=1, size=seq_len - 1)

        with tf.GradientTape(persistent=True) as tape:
            for i in tqdm(range(1, seq_len)):
                if (
                    probs[i - 1] <= skip_prob
                    and i >= self.n_past
                    and skip_count < max_skip_count
                    and i != 1
                    and i != cp_ix
                ):
                    skip_count += 1
                    continue

                if i > 1:
                    align_loss += self.align_criterion(h, h_pred)

                time_until_cp = tf.fill(
                    [batch_size, 1],
                    (cp_ix - i + 1) / cp_ix,
                )
                delta_time = tf.fill([batch_size, 1], ((i - prev_i) / cp_ix))
                prev_i = i

                h = self.encoder(x[:, i - 1, ...])
                h_target = self.encoder(x[:, i, ...])[0]

                if self.last_frame_skip or i <= self.n_past:
                    h, skip = h
                else:
                    h = h[0]

                # Control Point Aware
                h_cpaw = tf.concat([h, global_z, time_until_cp, delta_time], axis=-1)
                h_target_cpaw = tf.concat(
                    [h_target, global_z, time_until_cp, delta_time], axis=-1
                )

                zt, mu, logvar = self.posterior(h_target_cpaw)
                zt_p, mu_p, logvar_p = self.prior(h_cpaw)

                frame_predictor_input = tf.concat(
                    [h, zt, time_until_cp, delta_time], axis=-1
                )
                h_pred = self.frame_predictor(frame_predictor_input)
                x_pred = self.decoder([h_pred, skip])

                if i == cp_ix:  # the gen-cp-frame should be exactly as x_cp
                    h_pred_p = self.frame_predictor(
                        tf.concat([h, zt_p, time_until_cp, delta_time], axis=-1)
                    )
                    x_pred_p = self.decoder([h_pred_p, skip])
                    cpc_loss = self.mse_criterion(x_pred_p, x_cp)

                mse_loss += self.mse_criterion(x_pred, x[:, i, ...])
                kld_loss += self.kl_criterion((mu, logvar), (mu_p, logvar_p))

            # backward
            loss = (
                mse_loss
                + kld_loss * self.beta
                + align_loss * self.weight_align
                # + cpc_loss * self.weight_cpc
            )

            prior_loss = kld_loss + cpc_loss * self.weight_cpc

        var_list_frame_predictor = self.frame_predictor.trainable_variables
        var_list_posterior = self.posterior.trainable_variables
        var_list_prior = self.prior.trainable_variables
        var_list_encoder = self.encoder.trainable_variables
        var_list_decoder = self.decoder.trainable_variables

        # mse: frame_predictor + decoder
        # align: frame_predictor + encoder
        # kld: posterior + prior + encoder

        var_list = (
            var_list_frame_predictor
            + var_list_posterior
            + var_list_encoder
            + var_list_decoder
            + var_list_prior
        )

        gradients = tape.gradient(
            loss,
            var_list,
        )
        gradients_prior = tape.gradient(
            prior_loss,
            var_list_prior,
        )

        self.update_model(
            gradients,
            var_list,
        )
        self.update_prior(gradients_prior, var_list_prior)
        del tape

        self.total_loss_tracker.update_state(loss)
        self.kl_loss_tracker.update_state(kld_loss)
        self.align_loss_tracker.update_state(align_loss)
        self.reconstruction_loss_tracker.update_state(mse_loss)
        self.cpc_loss_tracker.update_state(cpc_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "align_loss": self.align_loss_tracker.result(),
            "cpc_loss": self.cpc_loss_tracker.result(),
        }

    def call(
        self,
        inputs,
        training=None,
        mask=None
        # len_output,
        # eval_cp_ix,
        # start_ix=0,
        # cp_ix=-1,
        # model_mode="full",
        # skip_frame=False,
        # init_hidden=True,
    ):
        len_output = 20
        eval_cp_ix = len_output - 1
        start_ix = 0
        cp_ix = -1
        model_mode = "full"
        skip_frame = False
        init_hidden = True

        batch_size, num_frames, h, w, channels = inputs.shape
        dim_shape = (h, w, channels)

        gen_seq = [inputs[:, 0, ...]]
        x_in = inputs[:, 0, ...]

        seq_len = inputs.shape[1]
        cp_ix = seq_len - 1

        x_cp, global_z = self.get_global_descriptor(
            inputs, cp_ix=cp_ix
        )  # here global_z is h_cp

        skip_prob = self.skip_prob

        prev_i = 0
        max_skip_count = seq_len * skip_prob
        skip_count = 0
        probs = np.random.uniform(0, 1, len_output - 1)

        for i in range(1, len_output):
            if (
                probs[i - 1] <= skip_prob
                and i >= self.n_past
                and skip_count < max_skip_count
                and i != 1
                and i != (len_output - 1)
                and skip_frame
            ):
                skip_count += 1
                gen_seq.append(tf.zeros_like(x_in))
                continue

            time_until_cp = tf.fill([100, 1], (eval_cp_ix - i + 1) / eval_cp_ix)

            delta_time = tf.fill([100, 1], ((i - prev_i) / eval_cp_ix))

            prev_i = i

            h = self.encoder(x_in)

            if self.last_frame_skip or i == 1 or i < self.n_past:
                h, skip = h
            else:
                h, _ = h

            h_cpaw = tf.stop_gradient(tf.concat([h, global_z, time_until_cp, delta_time], axis=-1))

            if i < self.n_past:
                h_target = self.encoder(inputs[:, i, ...])[0]
                h_target_cpaw = tf.stop_gradient(tf.concat(
                    [h_target, global_z, time_until_cp, delta_time], axis=1
                ))

                zt, _, _ = self.posterior(h_target_cpaw)
                zt_p, _, _ = self.prior(h_cpaw)

                if model_mode == "posterior" or model_mode == "full":
                    self.frame_predictor(
                        tf.concat([h, zt, time_until_cp, delta_time], axis=-1)
                    )
                elif model_mode == "prior":
                    self.frame_predictor(
                        tf.concat([h, zt_p, time_until_cp, delta_time], axis=-1)
                    )

                x_in = inputs[:, i, ...]
                gen_seq.append(x_in)
            else:
                if i < num_frames:
                    h_target = self.encoder(inputs[:, i, ...])[0]
                    h_target_cpaw = tf.stop_gradient(tf.concat(
                        [h_target, global_z, time_until_cp, delta_time], axis=-1
                    ))
                else:
                    h_target_cpaw = h_cpaw

                zt, _, _ = self.posterior(h_target_cpaw)
                zt_p, _, _ = self.prior(h_cpaw)

                if model_mode == "posterior":
                    h = self.frame_predictor(
                        tf.concat([h, zt, time_until_cp, delta_time], axis=-1)
                    )
                elif model_mode == "prior" or model_mode == "full":
                    h = self.frame_predictor(
                        tf.concat([h, zt_p, time_until_cp, delta_time], axis=-1)
                    )

                x_in = tf.stop_gradient(self.decoder([h, skip]))
                gen_seq.append(x_in)

        return tf.stack(gen_seq, axis=1)

    def update_model(self, gradients, var_list):
        self.frame_predictor_optimizer.apply_gradients(zip(gradients, var_list))
        self.posterior_optimizer.apply_gradients(zip(gradients, var_list))
        self.encoder_optimizer.apply_gradients(zip(gradients, var_list))
        self.decoder_optimizer.apply_gradients(zip(gradients, var_list))
        #self.prior_optimizer.apply_gradients(zip(gradients, var_list))

    def update_prior(self, gradients, var_list):
        self.prior_optimizer.apply_gradients(zip(gradients, var_list))

    # def update_model_without_prior(self):
    #     self.frame_predictor_optimizer.step()
    #     self.posterior_optimizer.step()
    #     self.encoder_optimizer.step()
    #     self.decoder_optimizer.step()
