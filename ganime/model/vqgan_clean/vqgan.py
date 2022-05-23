from typing import List, Literal, Tuple

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
        train_variance: float,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        z_channels: int = 128,  # 256,
        codebook_weight: float = 1.0,
        ae_channels: int = 128,  # ae stands for autoencoder
        ae_channels_multiplier: List[int] = [1, 1, 2, 2],
        ae_num_res_blocks: int = 1,  # 2,
        ae_attention_resolution: List[int] = [16],
        ae_resolution: int = 128,  # 256,
        ae_dropout=0.0,
        disc_num_layers: int = 3,
        disc_factor: float = 1.0,
        disc_iter_start: int = 0,
        disc_conditional: bool = False,
        disc_weight: float = 0.3,
        disc_filters: int = 64,
        disc_loss: Literal["hinge", "vanilla"] = "hinge",
        **kwargs,
    ):
        """Create a VQ-GAN model.

        Args:
            train_variance (float): The variance of the training dataset
            num_embeddings (int): The number of embeddings of the VQ-VAE
            embedding_dim (int): The dimension of the embeddings
            beta (float, optional): The beta parameter used as a discount factor for the VQ-VAE loss. Defaults to 0.25.
            z_channels (int, optional): The number of channels at the end of the Vector Quantizer. Defaults to 128.
            codebook_weight (float, optional): The weight applied to the codebook loss. Defaults to 1.0.
            ae_channels (int, optional): The number of channels for the first and last layer of the autoencoder. Defaults to 128.
            ae_channels_multiplier (List[int], optional): The multiplier for each level of the autoencoder. Starts with `ae-channels` (128) then multiply it by `ae_channels_multiplier[0]` at the next layer and so on. Defaults to [1, 1, 2, 2].
            ae_num_res_blocks (int, optional): The number of residual blocks for the autoencoder. Defaults to 1.
            ae_attention_resolution (List[int], optional): List indicating at which resolution an attention block will be put. Defaults to [16].
            ae_resolution (int, optional): The starting resolution of the image. Defaults to 128.
            ae_dropout (float, optional): The dropout value of the resnet blocks. Defaults to 0.0.
            disc_num_layers (int, optional): The number of layer for the discriminator. Defaults to 3.
            disc_factor (float, optional): The factor for the discriminator. Defaults to 1.0.
            disc_iter_start (int, optional): The number of steps when the discriminator will start training. Defaults to 0.
            disc_conditional (bool, optional): Whether the VQ-GAN is conditioned or not. Defaults to False.
            disc_weight (float, optional): The weight of the discriminator loss. Defaults to 0.3.
            disc_filters (int, optional): The starting number of filters for the discriminator. Defaults to 64.
            disc_loss (Literal[&quot;hinge&quot;, &quot;vanilla&quot;], optional): The type of discriminator loss. Defaults to "hinge".

        Raises:
            ValueError: The specified loss type is not supported.
        """
        super().__init__(**kwargs)
        self.train_variance = train_variance
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook_weight = codebook_weight
        self.beta = beta
        self.z_channels = z_channels
        self.ae_channels = ae_channels
        self.ae_channels_multiplier = ae_channels_multiplier
        self.ae_num_res_blocks = ae_num_res_blocks
        self.ae_attention_resolution = ae_attention_resolution
        self.ae_resolution = ae_resolution
        self.ae_dropout = ae_dropout
        self.disc_num_layers = disc_num_layers
        self.disc_filters = disc_filters
        self.disc_loss_str = disc_loss

        # Create the encoder - quant_conv - vector quantizer - post quant_conv - decoder
        self.encoder = Encoder(
            channels=ae_channels,
            channels_multiplier=ae_channels_multiplier,
            num_res_blocks=ae_num_res_blocks,
            attention_resolution=ae_attention_resolution,
            resolution=ae_resolution,
            dropout=ae_dropout,
        )
        self.decoder = Decoder(
            channels=ae_channels,
            channels_multiplier=ae_channels_multiplier,
            num_res_blocks=ae_num_res_blocks,
            attention_resolution=ae_attention_resolution,
            resolution=ae_resolution,
            dropout=ae_dropout,
        )
        self.quantize = VectorQuantizer(num_embeddings, embedding_dim, beta=beta)

        self.quant_conv = layers.Conv2D(embedding_dim, kernel_size=1)
        self.post_quant_conv = layers.Conv2D(z_channels, kernel_size=1)

        self.perceptual_loss = PerceptualLoss(reduction=tf.keras.losses.Reduction.NONE)

        # Setup discriminator and params
        self.discriminator = NLayerDiscriminator(
            filters=disc_filters,
            n_layers=disc_num_layers,
        )
        self.discriminator_iter_start = disc_iter_start
        self.disc_loss = self._get_discriminator_loss(disc_loss)
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "train_variance": self.train_variance,
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "beta": self.beta,
                "z_channels": self.z_channels,
                "codebook_weight": self.codebook_weight,
                "ae_channels": self.ae_channels,
                "ae_channels_multiplier": self.ae_channels_multiplier,
                "ae_num_res_blocks": self.ae_num_res_blocks,
                "ae_attention_resolution": self.ae_attention_resolution,
                "ae_resolution": self.ae_resolution,
                "ae_dropout": self.ae_dropout,
                "disc_num_layers": self.disc_num_layers,
                "disc_factor": self.disc_factor,
                "disc_iter_start": self.discriminator_iter_start,
                "disc_conditional": self.disc_conditional,
                "disc_weight": self.discriminator_weight,
                "disc_filters": self.disc_filters,
                "disc_loss": self.disc_loss_str,
            }
        )
        return config

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
        super().build(input_shape)

    def get_vqvae(self, input_shape):
        inputs = keras.Input(shape=input_shape[1:])
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
