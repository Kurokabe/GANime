from typing import List, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf
from .discriminator.model import NLayerDiscriminator
from .losses.vqperceptual import PerceptualLoss
from .vqvae.quantize import VectorQuantizer
from .diffusion.encoder import Encoder
from .diffusion.decoder import Decoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Optimizer
from ganime.configs.model_configs import (
    VQVAEConfig,
    AutoencoderConfig,
    DiscriminatorConfig,
    LossConfig,
)


@tf.function
def hinge_d_loss(logits_real, logits_fake):
    loss_real = tf.reduce_mean(keras.activations.relu(1.0 - logits_real))
    loss_fake = tf.reduce_mean(keras.activations.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


@tf.function
def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        tf.reduce_mean(keras.activations.softplus(-logits_real))
        + tf.reduce_mean(keras.activations.softplus(logits_fake))
    )
    return d_loss


class VQGAN(keras.Model):
    def __init__(
        self,
        vqvae_config: VQVAEConfig,
        autoencoder_config: AutoencoderConfig,
        discriminator_config: DiscriminatorConfig,
        loss_config: LossConfig,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        """Create a VQ-GAN model.

        Args:
            vqvae (VQVAEConfig): The configuration of the VQ-VAE
            autoencoder (AutoencoderConfig): The configuration of the autoencoder
            discriminator (DiscriminatorConfig): The configuration of the discriminator
            loss_config (LossConfig): The configuration of the loss

        Raises:
            ValueError: The specified loss type is not supported.
        """
        super().__init__(**kwargs)
        self.codebook_weight = loss_config.vqvae.codebook_weight
        self.vqvae_config = vqvae_config
        self.autoencoder_config = autoencoder_config
        self.discriminator_config = discriminator_config
        self.loss_config = loss_config
        # self.num_embeddings = num_embeddings
        # self.embedding_dim = embedding_dim
        # self.codebook_weight = codebook_weight
        # self.beta = beta
        # self.z_channels = z_channels
        # self.ae_channels = ae_channels
        # self.ae_channels_multiplier = ae_channels_multiplier
        # self.ae_num_res_blocks = ae_num_res_blocks
        # self.ae_attention_resolution = ae_attention_resolution
        # self.ae_resolution = ae_resolution
        # self.ae_dropout = ae_dropout
        # self.disc_num_layers = disc_num_layers
        # self.disc_filters = disc_filters
        # self.disc_loss_str = disc_loss

        # Create the encoder - quant_conv - vector quantizer - post quant_conv - decoder
        self.encoder = Encoder(**autoencoder_config)

        self.quant_conv = layers.Conv2D(
            vqvae_config.embedding_dim, kernel_size=1, name="pre_quant_conv"
        )

        self.quantize = VectorQuantizer(
            vqvae_config.num_embeddings,
            vqvae_config.embedding_dim,
            beta=vqvae_config.beta,
        )

        self.post_quant_conv = layers.Conv2D(
            autoencoder_config.z_channels, kernel_size=1, name="post_quant_conv"
        )

        self.decoder = Decoder(**autoencoder_config)

        self.perceptual_loss = PerceptualLoss(reduction=tf.keras.losses.Reduction.NONE)

        # Setup discriminator and params
        self.discriminator = NLayerDiscriminator(
            filters=discriminator_config.filters,
            n_layers=discriminator_config.num_layers,
        )
        self.discriminator_iter_start = loss_config.discriminator.iter_start
        self.disc_loss = self._get_discriminator_loss(loss_config.discriminator.loss)
        self.disc_factor = loss_config.discriminator.factor
        self.discriminator_weight = loss_config.discriminator.weight
        # self.disc_conditional = disc_conditional

        # Setup loss trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")

        # Setup optimizer (will be given with the compile method)
        self.gen_optimizer: Optimizer = None
        self.disc_optimizer: Optimizer = None

        self.checkpoint_path = checkpoint_path

    def load_from_checkpoint(self, path):
        self.load_weights(path)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
            self.disc_loss_tracker,
        ]

    # def get_config(self):
    #     config = super().get_config()
    #     config.update(
    #         {
    #             "train_variance": self.train_variance,
    #             "vqvae_config": self.vqvae_config,
    #             # "autoencoder_config": self.autoencoder_config,
    #             "discriminator_config": self.discriminator_config,
    #             "loss_config": self.loss_config,
    #         }
    #     )
    #     return config

    def _get_discriminator_loss(self, disc_loss):
        if disc_loss == "hinge":
            loss = hinge_d_loss
        elif disc_loss == "vanilla":
            loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        return loss

    def build(self, input_shape):
        # Defer the shape initialization
        self.vqvae = self.get_vqvae(input_shape)
        self.discriminator.build(input_shape)

        if self.checkpoint_path is not None:
            self.load_from_checkpoint(self.checkpoint_path)
        super().build(input_shape)

    def get_vqvae(self, input_shape):
        inputs = keras.Input(shape=input_shape[1:])
        quant, indices = self.encode(inputs)
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

    def calculate_adaptive_weight(
        self,
        nll_loss: tf.Tensor,
        g_loss: tf.Tensor,
        tape: tf.GradientTape,
        trainable_vars: list,
        discriminator_weight: float,
    ) -> tf.Tensor:
        """Calculate the adaptive weight for the discriminator which prevents mode collapse (https://arxiv.org/abs/2012.03149).

        Args:
            nll_loss (tf.Tensor): Negative log likelihood loss (the reconstruction loss).
            g_loss (tf.Tensor): Generator loss (compared to the discriminator).
            tape (tf.GradientTape): Gradient tape used to compute the nll_loss and g_loss
            trainable_vars (list): List of trainable vars of the last layer (conv_out of the decoder)
            discriminator_weight (float): Weight of the discriminator

        Returns:
            tf.Tensor: Discriminator weights used for the discriminator loss to benefits best the generator or discriminator and avoiding mode collapse.
        """
        nll_grads = tape.gradient(nll_loss, trainable_vars)[0]
        g_grads = tape.gradient(g_loss, trainable_vars)[0]

        d_weight = tf.norm(nll_grads) / (tf.norm(g_grads) + 1e-4)
        d_weight = tf.stop_gradient(tf.clip_by_value(d_weight, 0.0, 1e4))
        return d_weight * discriminator_weight

    @tf.function
    def adapt_weight(
        self, weight: float, global_step: int, threshold: int = 0, value: float = 0.0
    ) -> float:
        """Adapt the weight depending on the global step. If the global_step is lower than the threshold, the weight is set to value. Used to reduce the weight of the discriminator during the first iterations.

        Args:
            weight (float): The weight to adapt.
            global_step (int): The global step of the optimizer
            threshold (int, optional): The threshold under which the weight will be set to `value`. Defaults to 0.
            value (float, optional): The value of the weight. Defaults to 0.0.

        Returns:
            float: The adapted weight
        """
        if global_step < threshold:
            weight = value
        return weight

    def _get_global_step(self, optimizer: Optimizer):
        """Get the global step of the optimizer."""
        return optimizer.iterations

    def compile(
        self,
        gen_optimizer,
        disc_optimizer,
    ):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        x, y = data

        # Train the generator
        with tf.GradientTape() as tape:  # Gradient tape for the final loss
            with tf.GradientTape(
                persistent=True
            ) as adaptive_tape:  # Gradient tape for the adaptive weights
                reconstructions = self(x, training=True)

                logits_fake = self.discriminator(reconstructions, training=False)

                g_loss = -tf.reduce_mean(logits_fake)
                nll_loss = self.perceptual_loss(y, reconstructions)

            d_weight = self.calculate_adaptive_weight(
                nll_loss,
                g_loss,
                adaptive_tape,
                self.decoder.conv_out.trainable_variables,
                self.discriminator_weight,
            )
            del adaptive_tape  # Since persistent tape, important to delete it

            disc_factor = self.adapt_weight(
                weight=self.disc_factor,
                global_step=self._get_global_step(self.gen_optimizer),
                threshold=self.discriminator_iter_start,
            )

            total_loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                # + self.codebook_weight * tf.reduce_mean(self.vqvae.losses)
                + self.codebook_weight * sum(self.vqvae.losses)
            )

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Discriminator
        with tf.GradientTape() as disc_tape:
            logits_real = self.discriminator(y, training=True)
            logits_fake = self.discriminator(reconstructions, training=True)

            disc_factor = self.adapt_weight(
                weight=self.disc_factor,
                global_step=self._get_global_step(self.disc_optimizer),
                threshold=self.discriminator_iter_start,
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        # Backpropagation.
        disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(nll_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        self.disc_loss_tracker.update_state(d_loss)

        # Log results.
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        x, y = data

        # Train the generator
        with tf.GradientTape(
            persistent=True
        ) as adaptive_tape:  # Gradient tape for the adaptive weights
            reconstructions = self(x, training=False)

            logits_fake = self.discriminator(reconstructions, training=False)

            g_loss = -tf.reduce_mean(logits_fake)
            nll_loss = self.perceptual_loss(y, reconstructions)

        d_weight = self.calculate_adaptive_weight(
            nll_loss,
            g_loss,
            adaptive_tape,
            self.decoder.conv_out.trainable_variables,
            self.discriminator_weight,
        )
        del adaptive_tape  # Since persistent tape, important to delete it

        disc_factor = self.adapt_weight(
            weight=self.disc_factor,
            global_step=self._get_global_step(self.gen_optimizer),
            threshold=self.discriminator_iter_start,
        )

        total_loss = (
            nll_loss
            + d_weight * disc_factor * g_loss
            # + self.codebook_weight * tf.reduce_mean(self.vqvae.losses)
            + self.codebook_weight * sum(self.vqvae.losses)
        )

        # Discriminator
        logits_real = self.discriminator(y, training=False)
        logits_fake = self.discriminator(reconstructions, training=False)

        disc_factor = self.adapt_weight(
            weight=self.disc_factor,
            global_step=self._get_global_step(self.disc_optimizer),
            threshold=self.discriminator_iter_start,
        )
        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(nll_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))
        self.disc_loss_tracker.update_state(d_loss)

        # Log results.
        return {m.name: m.result() for m in self.metrics}
