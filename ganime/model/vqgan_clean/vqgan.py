from typing import List, Literal, Optional, Tuple

import numpy as np
import tensorflow as tf

from ganime.model.vqgan_clean.losses.losses import Losses
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
        num_replicas: int = 1,
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
        self.perceptual_weight = loss_config.vqvae.perceptual_weight
        self.codebook_weight = loss_config.vqvae.codebook_weight
        self.vqvae_config = vqvae_config
        self.autoencoder_config = autoencoder_config
        self.discriminator_config = discriminator_config
        self.loss_config = loss_config
        self.num_replicas = num_replicas
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

        self.perceptual_loss = self.get_perceptual_loss(
            loss_config.perceptual_loss
        )  # PerceptualLoss(reduction=tf.keras.losses.Reduction.NONE)

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

        self.cross_entropy = Losses(self.num_replicas).bce_loss
        self.reconstruction_loss = self.get_reconstruction_loss("mae")

    def get_perceptual_loss(self, loss_type: str):
        if loss_type == "vgg16":
            return PerceptualLoss(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_type == "vgg19":
            return Losses(self.num_replicas).vgg_loss
        elif loss_type == "style":
            return Losses(self.num_replicas).style_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def get_reconstruction_loss(self, loss_type: str):
        if loss_type == "mse":
            return Losses(self.num_replicas).mse_loss
        elif loss_type == "mae":
            return Losses(self.num_replicas).mae_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

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
        # self.vqvae = self.get_vqvae(input_shape)

        if self.checkpoint_path is not None:
            self.load_from_checkpoint(self.checkpoint_path)
        super().build(input_shape)

    # def get_vqvae(self, input_shape):
    #     inputs = keras.Input(shape=input_shape[1:])
    #     quant, indices, loss = self.encode(inputs)
    #     reconstructed = self.decode(quant)
    #     return keras.Model(inputs, reconstructed, name="vq_vae")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantize(h)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def call(self, inputs, training=True, mask=None):
        quantized, encoding_indices, loss = self.encode(inputs)
        reconstructed = self.decode(quantized)
        return reconstructed, loss

    def predict(self, inputs):
        output, loss = self(inputs)
        output = (output + 1.0) * 127.5 / 255
        return output

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

    def get_vqvae_trainable_vars(self):
        return (
            self.encoder.trainable_variables
            + self.quant_conv.trainable_variables
            + self.quantize.trainable_variables
            + self.post_quant_conv.trainable_variables
            + self.decoder.trainable_variables
        )

    # def gradient_penalty(self, real, f):
    #     def interpolate(a):
    #         beta = tf.random.uniform(shape=tf.shape(a), minval=0.0, maxval=1.0)
    #         _, variance = tf.nn.moments(a, list(range(a.shape.ndims)))
    #         b = a + 0.5 * tf.sqrt(variance) * beta

    #         shape = tf.concat(
    #             (tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0
    #         )
    #         alpha = tf.random.uniform(shape=shape, minval=0.0, maxval=1.0)
    #         inter = a + alpha * (b - a)
    #         inter.set_shape(a.get_shape().as_list())

    #         return inter

    #     x = interpolate(real)
    #     pred = f(x)
    #     gradients = tf.gradients(pred, x)[0]
    #     slopes = tf.sqrt(
    #         tf.reduce_sum(tf.square(gradients), axis=list(range(1, x.shape.ndims)))
    #     )
    #     gp = tf.reduce_mean((slopes - 1.0) ** 2)
    #     return gp

    # def discriminator_loss(self, real_images, real_output, fake_output, discriminator):
    #     real_loss = self.cross_entropy(
    #         tf.ones_like(real_output), real_output
    #     )  # tf.losses.sigmoid_cross_entropy(tf.ones_like(real_output), real_output)
    #     fake_loss = self.cross_entropy(
    #         tf.zeros_like(fake_output), fake_output
    #     )  # tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_output), fake_output)
    #     gp = self.gradient_penalty(real_images, discriminator)
    #     total_loss = real_loss + fake_loss + 10.0 * gp
    #     return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.cross_entropy(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.cross_entropy(
            tf.zeros_like(disc_generated_output), disc_generated_output
        )

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss

    # def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
    #     x, y = data

    #     # Train the generator
    #     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  # Gradient tape for the final loss
    #         reconstructions, quantized_loss = self(x, training=True)

    #         logits_fake = self.discriminator(reconstructions, training=True)
    #         logits_true = self.discriminator(x, training=True)

    #         g_loss = self.generator_loss(logits_fake)#-tf.reduce_mean(logits_fake)
    #         disc_loss = self.discriminator_loss(x, logits_true, logits_fake, self.discriminator)

    #         nll_loss = self.perceptual_loss(y, reconstructions)

    #         disc_factor = self.adapt_weight(
    #             weight=self.disc_factor,
    #             global_step=self._get_global_step(self.gen_optimizer),
    #             threshold=self.discriminator_iter_start,
    #         )

    #         total_loss = (
    #             self.perceptual_weight * nll_loss
    #             + disc_factor * g_loss
    #             + self.codebook_weight * quantized_loss
    #         )

    #         d_loss = disc_factor * disc_loss

    #     # Backpropagation.
    #     gen_grads = gen_tape.gradient(total_loss, self.get_vqvae_trainable_vars())
    #     self.gen_optimizer.apply_gradients(zip(gen_grads, self.get_vqvae_trainable_vars()))

    #     disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
    #     self.disc_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

    #     # Loss tracking.
    #     self.total_loss_tracker.update_state(total_loss)
    #     self.reconstruction_loss_tracker.update_state(nll_loss)
    #     self.vq_loss_tracker.update_state(quantized_loss)
    #     self.disc_loss_tracker.update_state(d_loss)

    #     # Log results.
    #     return {m.name: m.result() for m in self.metrics}

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        x, y = data

        # Train the generator
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  # Gradient tape for the final loss
            with tf.GradientTape(
                persistent=True
            ) as adaptive_tape:  # Gradient tape for the adaptive weights
                reconstructions, quantized_loss = self(x, training=True)

                disc_real_input = tf.image.resize(
                    x, [256, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                )
                disc_gen_input = tf.image.resize(
                    reconstructions,
                    [256, 512],
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                )
                logits_real = self.discriminator(
                    (disc_real_input, disc_real_input),
                    training=True,
                )
                logits_fake = self.discriminator(
                    (disc_real_input, disc_gen_input),
                    training=True,
                )

                reconstruction_loss = self.reconstruction_loss(y, reconstructions)
                if self.perceptual_weight > 0.0:
                    perceptual_loss = self.perceptual_weight * self.perceptual_loss(
                        y, reconstructions
                    )
                else:
                    perceptual_loss = 0.0

                nll_loss = reconstruction_loss + perceptual_loss

                g_loss = -tf.reduce_mean(logits_fake)
                # g_loss = self.generator_loss(logits_fake)

            d_weight = self.calculate_adaptive_weight(
                nll_loss,
                g_loss,
                adaptive_tape,
                self.decoder.conv_out.trainable_variables,
                self.discriminator_weight,
            )
            del adaptive_tape  # Since persistent tape, important to delete it
            # d_weight = 1.0

            disc_factor = self.adapt_weight(
                weight=self.disc_factor,
                global_step=self._get_global_step(self.gen_optimizer),
                threshold=self.discriminator_iter_start,
            )

            total_loss = (
                nll_loss
                + d_weight * disc_factor * g_loss
                + self.codebook_weight * quantized_loss
            )

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            # d_loss = disc_factor * self.discriminator_loss(
            #     tf.concat((disc_real_input, disc_real_input), axis=-1),
            #     logits_real,
            #     logits_fake,
            #     self.discriminator,
            # )
            # d_loss = disc_factor * self.discriminator_loss(
            #     logits_real,
            #     logits_fake,
            # )

        # Backpropagation.
        grads = gen_tape.gradient(total_loss, self.get_vqvae_trainable_vars())
        self.gen_optimizer.apply_gradients(zip(grads, self.get_vqvae_trainable_vars()))

        # Backpropagation.
        disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(nll_loss)
        self.vq_loss_tracker.update_state(quantized_loss)
        self.disc_loss_tracker.update_state(d_loss)

        # Log results.
        return {m.name: m.result() for m in self.metrics}

    # def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
    #     x, y = data

    #     # Train the generator
    #     with tf.GradientTape() as tape:  # Gradient tape for the final loss
    #         with tf.GradientTape(
    #             persistent=True
    #         ) as adaptive_tape:  # Gradient tape for the adaptive weights
    #             reconstructions, quantized_loss = self(x, training=True)

    #             logits_fake = self.discriminator(reconstructions, training=False)

    #             g_loss = -tf.reduce_mean(logits_fake)
    #             nll_loss = self.perceptual_loss(y, reconstructions)

    #         d_weight = self.calculate_adaptive_weight(
    #             nll_loss,
    #             g_loss,
    #             adaptive_tape,
    #             self.decoder.conv_out.trainable_variables,
    #             self.discriminator_weight,
    #         )
    #         del adaptive_tape  # Since persistent tape, important to delete it

    #         disc_factor = self.adapt_weight(
    #             weight=self.disc_factor,
    #             global_step=self._get_global_step(self.gen_optimizer),
    #             threshold=self.discriminator_iter_start,
    #         )

    #         total_loss = (
    #             self.perceptual_weight * nll_loss
    #             + d_weight * disc_factor * g_loss
    #             + self.codebook_weight * quantized_loss
    #         )

    #         # total_loss = (
    #         #     nll_loss
    #         #     + d_weight * disc_factor * g_loss
    #         #     # + self.codebook_weight * tf.reduce_mean(self.vqvae.losses)
    #         #     + self.codebook_weight * sum(self.vqvae.losses)
    #         # )

    #     # Backpropagation.
    #     grads = tape.gradient(total_loss, self.get_vqvae_trainable_vars())
    #     self.gen_optimizer.apply_gradients(zip(grads, self.get_vqvae_trainable_vars()))

    #     # Discriminator
    #     with tf.GradientTape() as disc_tape:
    #         logits_real = self.discriminator(y, training=True)
    #         logits_fake = self.discriminator(reconstructions, training=True)

    #         disc_factor = self.adapt_weight(
    #             weight=self.disc_factor,
    #             global_step=self._get_global_step(self.disc_optimizer),
    #             threshold=self.discriminator_iter_start,
    #         )
    #         d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

    #     # Backpropagation.
    #     disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
    #     self.disc_optimizer.apply_gradients(
    #         zip(disc_grads, self.discriminator.trainable_variables)
    #     )

    #     # Loss tracking.
    #     self.total_loss_tracker.update_state(total_loss)
    #     self.reconstruction_loss_tracker.update_state(nll_loss)
    #     self.vq_loss_tracker.update_state(quantized_loss)
    #     self.disc_loss_tracker.update_state(d_loss)

    #     # Log results.
    #     return {m.name: m.result() for m in self.metrics}

    # def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
    #     x, y = data

    #     # Train the generator
    #     with tf.GradientTape(
    #         persistent=True
    #     ) as adaptive_tape:  # Gradient tape for the adaptive weights
    #         reconstructions, quantized_loss = self(x, training=False)

    #         logits_fake = self.discriminator(reconstructions, training=False)

    #         g_loss = -tf.reduce_mean(logits_fake)
    #         nll_loss = self.perceptual_loss(y, reconstructions)

    #     d_weight = self.calculate_adaptive_weight(
    #         nll_loss,
    #         g_loss,
    #         adaptive_tape,
    #         self.decoder.conv_out.trainable_variables,
    #         self.discriminator_weight,
    #     )
    #     del adaptive_tape  # Since persistent tape, important to delete it

    #     disc_factor = self.adapt_weight(
    #         weight=self.disc_factor,
    #         global_step=self._get_global_step(self.gen_optimizer),
    #         threshold=self.discriminator_iter_start,
    #     )

    #     total_loss = (
    #         nll_loss
    #         + d_weight * disc_factor * g_loss
    #         # + self.codebook_weight * tf.reduce_mean(self.vqvae.losses)
    #         + self.codebook_weight * quantized_loss
    #     )

    #     # Discriminator
    #     logits_real = self.discriminator(y, training=False)
    #     logits_fake = self.discriminator(reconstructions, training=False)

    #     disc_factor = self.adapt_weight(
    #         weight=self.disc_factor,
    #         global_step=self._get_global_step(self.disc_optimizer),
    #         threshold=self.discriminator_iter_start,
    #     )
    #     d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

    #     # Loss tracking.
    #     self.total_loss_tracker.update_state(total_loss)
    #     self.reconstruction_loss_tracker.update_state(nll_loss)
    #     self.vq_loss_tracker.update_state(quantized_loss)
    #     self.disc_loss_tracker.update_state(d_loss)

    #     # Log results.
    #     return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        x, y = data

        with tf.GradientTape(
            persistent=True
        ) as adaptive_tape:  # Gradient tape for the adaptive weights
            reconstructions, quantized_loss = self(x, training=False)

            disc_real_input = tf.image.resize(
                x, [256, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            disc_gen_input = tf.image.resize(
                reconstructions,
                [256, 512],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            logits_real = self.discriminator(
                (disc_real_input, disc_real_input),
                training=False,
            )
            logits_fake = self.discriminator(
                (disc_real_input, disc_gen_input),
                training=False,
            )

            reconstruction_loss = self.reconstruction_loss(y, reconstructions)
            if self.perceptual_weight > 0.0:
                perceptual_loss = self.perceptual_weight * self.perceptual_loss(
                    y, reconstructions
                )
            else:
                perceptual_loss = 0.0

            nll_loss = reconstruction_loss + perceptual_loss
            g_loss = -tf.reduce_mean(logits_fake)
            # g_loss = self.generator_loss(logits_fake)

        d_weight = self.calculate_adaptive_weight(
            nll_loss,
            g_loss,
            adaptive_tape,
            self.decoder.conv_out.trainable_variables,
            self.discriminator_weight,
        )
        del adaptive_tape  # Since persistent tape, important to delete it
        # d_weight = 1.0

        disc_factor = self.adapt_weight(
            weight=self.disc_factor,
            global_step=self._get_global_step(self.gen_optimizer),
            threshold=self.discriminator_iter_start,
        )

        total_loss = (
            nll_loss
            + d_weight * disc_factor * g_loss
            + self.codebook_weight * quantized_loss
        )

        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
        # d_loss = disc_factor * self.discriminator_loss(
        #     tf.concat((disc_real_input, disc_real_input), axis=-1),
        #     logits_real,
        #     logits_fake,
        #     self.discriminator,
        # )
        # d_loss = disc_factor * self.discriminator_loss(
        #     logits_real,
        #     logits_fake,
        # )

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(nll_loss)
        self.vq_loss_tracker.update_state(quantized_loss)
        self.disc_loss_tracker.update_state(d_loss)

        # Log results.
        return {m.name: m.result() for m in self.metrics}
