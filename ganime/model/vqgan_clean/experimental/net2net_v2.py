import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from ganime.configs.model_configs import GPTConfig, ModelConfig
from ganime.model.vqgan_clean.experimental.transformer import Transformer
from ganime.model.vqgan_clean.vqgan import VQGAN
from ganime.trainer.warmup.cosine import WarmUpCosine
from tensorflow import keras
from tensorflow.keras import Model, layers
from ganime.model.vqgan_clean.losses.losses import Losses
from ganime.trainer.warmup.base import create_warmup_scheduler


class Net2Net(Model):
    def __init__(
        self,
        transformer_config: GPTConfig,
        first_stage_config: ModelConfig,
        trainer_config,
        num_replicas: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.first_stage_model = VQGAN(**first_stage_config)
        self.transformer = Transformer(transformer_config)

        if "checkpoint_path" in transformer_config:
            print(f"Restoring weights from {transformer_config['checkpoint_path']}")
            self.load_weights(transformer_config["checkpoint_path"])

        losses = Losses(num_replicas=num_replicas)
        self.scce_loss = losses.scce_loss

        self.scheduled_lrs = create_warmup_scheduler(
            trainer_config, num_devices=num_replicas
        )

        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.scheduled_lrs, weight_decay=1e-4
        )
        self.compile(
            optimizer=optimizer,
            loss=self.loss_fn,
            # run_eagerly=True,
        )

        # Gradient accumulation
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in self.transformer.trainable_variables
        ]

        self.loss_tracker = keras.metrics.Mean(name="loss")

    def loss_fn(self, logits_true, logits_pred):
        frame_loss = self.scce_loss(logits_true, logits_pred)
        return frame_loss

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.transformer.trainable_variables)
        )

        # reset
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.transformer.trainable_variables[i], dtype=tf.float32)
            )

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.loss_tracker,
        ]

    @tf.function()
    def encode_to_z(self, x):
        quant_z, indices, quantized_loss = self.first_stage_model.encode(x)

        batch_size = tf.shape(quant_z)[0]

        indices = tf.reshape(indices, shape=(batch_size, -1))
        return quant_z, indices

    def call(self, inputs, training=None, mask=None):

        first_frame = inputs["first_frame"]
        last_frame = inputs["last_frame"]
        n_frames = inputs["n_frames"]

        return self.generate_video(first_frame, last_frame, n_frames)

    @tf.function()
    def predict_next_indices(self, inputs):
        logits = self.transformer(inputs)
        return logits

    @tf.function()
    def body(self, total_loss, frames, index, last_frame_indices):

        previous_frame_indices = self.encode_to_z(frames[:, index - 1, ...])[1]
        target_indices = self.encode_to_z(frames[:, index, ...])[1]
        # target_indices = tf.reshape(target_indices, shape=(-1,))

        with tf.GradientTape() as tape:
            logits = self.predict_next_indices(
                (last_frame_indices, previous_frame_indices)
            )

            frame_loss = self.loss_fn(target_indices, logits)

        # Calculate batch gradients
        gradients = tape.gradient(frame_loss, self.transformer.trainable_variables)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(tf.cast(gradients[i], tf.float32))

        index = tf.add(index, 1)
        total_loss = tf.add(total_loss, frame_loss)
        return total_loss, frames, index, last_frame_indices

    def cond(self, total_loss, frames, index, last_frame_indices):
        return tf.less(index, tf.shape(frames)[1])

    def train_step(self, data):
        first_frame = data["first_frame"]
        last_frame = data["last_frame"]
        frames = data["y"]
        n_frames = data["n_frames"]

        last_frame_indices = self.encode_to_z(last_frame)[1]
        total_loss = 0.0

        total_loss, _, _, _ = tf.while_loop(
            cond=self.cond,
            body=self.body,
            loop_vars=(tf.constant(0.0), frames, tf.constant(1), last_frame_indices),
        )

        self.apply_accu_gradients()
        self.loss_tracker.update_state(total_loss)
        return {m.name: m.result() for m in self.metrics}

    def cond_test_step(self, total_loss, frames, index, last_frame_indices):
        return tf.less(index, tf.shape(frames)[1])

    @tf.function()
    def body_test_step(self, total_loss, frames, index, predicted_logits):
        target_indices = self.encode_to_z(frames[:, index, ...])[1]
        # target_indices = tf.reshape(target_indices, shape=(-1,))
        logits = predicted_logits[index]

        frame_loss = self.loss_fn(target_indices, logits)

        index = tf.add(index, 1)
        total_loss = tf.add(total_loss, frame_loss)
        return total_loss, frames, index, predicted_logits

    def test_step(self, data):
        first_frame = data["first_frame"]
        last_frame = data["last_frame"]
        frames = data["y"]
        n_frames = data["n_frames"]

        predicted_logits, _, _ = self.predict_logits(first_frame, last_frame, n_frames)

        total_loss, _, _, _ = tf.while_loop(
            cond=self.cond_test_step,
            body=self.body_test_step,
            loop_vars=(tf.constant(0.0), frames, tf.constant(1), predicted_logits),
        )

        self.loss_tracker.update_state(total_loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function()
    def convert_logits_to_indices(self, logits, shape):
        probs = tf.keras.activations.softmax(logits)
        _, generated_indices = tf.math.top_k(probs)
        generated_indices = tf.reshape(
            generated_indices,
            shape,  # , self.first_stage_model.quantize.num_embeddings)
        )
        return generated_indices
        # quant = self.first_stage_model.quantize.get_codebook_entry(
        #     generated_indices, shape=shape
        # )

        # return self.first_stage_model.decode(quant)

    @tf.function()
    def predict_logits(self, first_frame, last_frame, n_frames):
        quant_first, indices_first = self.encode_to_z(first_frame)
        quant_last, indices_last = self.encode_to_z(last_frame)

        indices_previous = indices_first

        predicted_logits = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )

        index = tf.constant(1)
        while tf.less(index, tf.reduce_max(n_frames)):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(indices_previous, tf.TensorShape([None, None]))]
            )
            logits = self.predict_next_indices((indices_last, indices_previous))

            # generated_indices = self.convert_logits_to_indices(
            #     logits, tf.shape(indices_last)
            # )
            predicted_logits = predicted_logits.write(index, logits)
            indices_previous = self.convert_logits_to_indices(
                logits, tf.shape(indices_first)
            )
            index = tf.add(index, 1)

        return predicted_logits.stack(), tf.shape(quant_first), tf.shape(indices_first)

    @tf.function()
    def generate_video(self, first_frame, last_frame, n_frames):
        predicted_logits, quant_shape, indices_shape = self.predict_logits(
            first_frame, last_frame, n_frames
        )

        generated_images = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )
        generated_images = generated_images.write(0, first_frame)

        index = tf.constant(1)
        while tf.less(index, tf.reduce_max(n_frames)):
            indices = self.convert_logits_to_indices(
                predicted_logits[index], indices_shape
            )
            quant = self.first_stage_model.quantize.get_codebook_entry(
                indices,
                shape=quant_shape,
            )
            decoded = self.first_stage_model.decode(quant)
            generated_images = generated_images.write(index, decoded)
            index = tf.add(index, 1)

        stacked_images = generated_images.stack()
        videos = tf.transpose(stacked_images, (1, 0, 2, 3, 4))
        return videos
