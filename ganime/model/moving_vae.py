from tensorflow.keras import Model

import tensorflow as tf
import tensorflow_probability as tfp


class MovingVAE(Model):
    def __init__(self, input_shape, encoded_size=64, base_depth=32):
        super().__init__()

        self.encoded_size = encoded_size
        self.base_depth = base_depth

        self.prior = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(encoded_size), scale=1),
            reinterpreted_batch_ndims=1,
        )

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),
                tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
                tf.keras.layers.Conv3D(
                    self.base_depth,
                    5,
                    strides=1,
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv3D(
                    self.base_depth,
                    5,
                    strides=2,
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv3D(
                    2 * self.base_depth,
                    5,
                    strides=1,
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv3D(
                    2 * self.base_depth,
                    5,
                    strides=2,
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                # tf.keras.layers.Conv3D(4 * encoded_size, 7, strides=1,
                #            padding='valid', activation=tf.nn.leaky_relu),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    tfp.layers.MultivariateNormalTriL.params_size(self.encoded_size),
                    activation=None,
                ),
                tfp.layers.MultivariateNormalTriL(
                    self.encoded_size,
                    activity_regularizer=tfp.layers.KLDivergenceRegularizer(self.prior),
                ),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=[self.encoded_size]),
                tf.keras.layers.Reshape([1, 1, 1, self.encoded_size]),
                tf.keras.layers.Conv3DTranspose(
                    self.base_depth,
                    (5, 4, 4),
                    strides=1,
                    padding="valid",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv3DTranspose(
                    2 * self.base_depth,
                    (5, 4, 4),
                    strides=(1, 2, 2),
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv3DTranspose(
                    2 * self.base_depth,
                    (5, 4, 4),
                    strides=2,
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv3DTranspose(
                    self.base_depth,
                    (5, 4, 4),
                    strides=(1, 2, 2),
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv3DTranspose(
                    self.base_depth,
                    (5, 4, 4),
                    strides=2,
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv3DTranspose(
                    self.base_depth,
                    (5, 4, 4),
                    strides=1,
                    padding="same",
                    activation=tf.nn.leaky_relu,
                ),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=5, strides=1, padding="same", activation=None
                ),
                tf.keras.layers.Flatten(),
                tfp.layers.IndependentBernoulli(
                    input_shape, tfp.distributions.Bernoulli.logits
                ),
            ]
        )

        self.model = tf.keras.Model(
            inputs=self.encoder.inputs, outputs=self.decoder(self.encoder.outputs[0])
        )

    def call(self, inputs):
        return self.model(inputs)
