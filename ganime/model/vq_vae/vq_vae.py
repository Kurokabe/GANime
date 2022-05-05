import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

input_shape = (20, 64, 64, 1)

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
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class VQVAE(keras.Model):
    def __init__(self, train_variance:float, latent_dim:int=32, num_embeddings:int=128, **kwargs):
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = self.get_vqvae()

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")


    def get_encoder(self):
        encoder_inputs = keras.Input(shape=input_shape)
        x = layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"))(
            encoder_inputs
        )
        x = layers.TimeDistributed(layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"))(x)
        encoder_outputs = layers.TimeDistributed(layers.Conv2D(self.latent_dim, 1, padding="same"))(x)
        return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


    def get_decoder(self):
        latent_inputs = keras.Input(shape=self.get_encoder().output.shape[1:])
        x = layers.TimeDistributed(layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))(
            latent_inputs
        )
        x = layers.TimeDistributed(layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))(x)
        decoder_outputs = layers.TimeDistributed(layers.Conv2DTranspose(1, 3, padding="same"))(x)
        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def get_vqvae(self):
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.latent_dim, name="vector_quantizer")
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        inputs = keras.Input(shape=input_shape)
        encoder_outputs = self.encoder(inputs)
        quantized_latents = self.vq_layer(encoder_outputs)
        reconstructions = self.decoder(quantized_latents)
        return keras.Model(inputs, reconstructions, name="vq_vae")

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

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

    def call(self, inputs, training=False, mask=None):
        return self.vqvae(inputs)
