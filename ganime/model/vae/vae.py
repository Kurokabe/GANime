import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

input_shape = (20, 64, 64, 1)

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = z_mean.shape[1:]
        epsilon = tf.keras.backend.random_normal(shape=(batch, *dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VAE(keras.Model):
    def __init__(self, latent_dim:int=32, num_embeddings:int=128, beta:float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")


    def get_encoder(self):
        encoder_inputs = keras.Input(shape=input_shape)
        x = layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"))(
            encoder_inputs
        )
        x = layers.TimeDistributed(layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"))(x)
        x = layers.TimeDistributed(layers.Conv2D(self.latent_dim, 1, padding="same"))(x)
        
        x = layers.TimeDistributed(layers.Flatten())(x)
        mu = layers.TimeDistributed(layers.Dense(self.num_embeddings))(x)
        logvar = layers.TimeDistributed(layers.Dense(self.num_embeddings))(x)
        z = Sampling()([mu, logvar])
        
        return keras.Model(encoder_inputs, [mu, logvar, z], name="encoder")


    def get_decoder(self):
        latent_inputs = keras.Input(shape=self.encoder.output[2].shape[1:])

        x = layers.TimeDistributed(layers.Dense(16 * 16 * 32, activation="relu"))(latent_inputs)
        x = layers.TimeDistributed(layers.Reshape((16, 16, 32)))(x)
        x = layers.TimeDistributed(layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))(
            x
        )
        x = layers.TimeDistributed(layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))(x)
        decoder_outputs = layers.TimeDistributed(layers.Conv2DTranspose(1, 3, padding="same"))(x)
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            mu, logvar, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(y, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta * kl_loss
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

    def call(self, inputs, training=False, mask=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        pred = self.decoder(z)
        return pred
