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
from ganime.visualization.images import unnormalize_if_necessary


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

        self.scheduled_lrs = create_warmup_scheduler(
            trainer_config, num_devices=num_replicas
        )

        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.scheduled_lrs, weight_decay=1e-4
        )
        self.compile(
            optimizer=optimizer,
            # loss=self.loss_fn,
            # run_eagerly=True,
        )

        self.n_frames_before = trainer_config["n_frames_before"]

        # Gradient accumulation
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in self.transformer.trainable_variables
        ]
        self.accumulation_size = trainer_config["accumulation_size"]

        # Losses
        self.perceptual_loss_weight = trainer_config["perceptual_loss_weight"]
        losses = Losses(num_replicas=num_replicas)
        self.scce_loss = losses.scce_loss
        self.perceptual_loss = losses.perceptual_loss

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.scce_loss_tracker = keras.metrics.Mean(name="scce_loss")
        self.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")

        self.epoch = 0
        self.stop_ground_truth_after_epoch = trainer_config[
            "stop_ground_truth_after_epoch"
        ]

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
            self.total_loss_tracker,
            self.scce_loss_tracker,
            self.perceptual_loss_tracker,
        ]

    @tf.function()
    def encode_to_z(self, x):
        quant_z, indices, quantized_loss = self.first_stage_model.encode(x)

        batch_size = tf.shape(quant_z)[0]

        indices = tf.reshape(indices, shape=(batch_size, -1))
        return quant_z, indices

    def call(self, inputs, training=False, mask=None, return_losses=False):

        return self.predict_video(inputs, training, return_losses)

    def predict(self, data, sample=False, temperature=1.0):
        video = self.predict_video(
            data,
            training=False,
            return_losses=False,
            sample=sample,
            temperature=temperature,
        )
        video = unnormalize_if_necessary(video)
        return video

    def get_remaining_frames(self, inputs):
        if "remaining_frames" in inputs:
            remaining_frames = inputs["remaining_frames"]
        else:
            raise NotImplementedError
        remaining_frames = tf.cast(remaining_frames, tf.int64)
        return remaining_frames

    # @tf.function()
    def predict_video(
        self, inputs, training=False, return_losses=False, sample=False, temperature=1.0
    ):
        first_frame = inputs["first_frame"]
        last_frame = inputs["last_frame"]
        n_frames = tf.reduce_max(inputs["n_frames"])
        remaining_frames = self.get_remaining_frames(inputs)

        try:
            ground_truth = inputs["y"]
        except AttributeError:
            ground_truth = None

        previous_frames = tf.expand_dims(first_frame, axis=1)

        predictions = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )

        quant_last, indices_last = self.encode_to_z(last_frame)

        total_loss = tf.constant(0.0)
        scce_loss = tf.constant(0.0)
        perceptual_loss = tf.constant(0.0)

        current_frame_index = tf.constant(1)
        while tf.less(current_frame_index, n_frames):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (previous_frames, tf.TensorShape([None, None, None, None, 3]))
                ],
            )

            if ground_truth is not None:
                target_frame = ground_truth[:, current_frame_index]
            else:
                target_frame = None

            y_pred, losses = self.predict_next_frame(
                remaining_frames[:, current_frame_index],
                previous_frames,
                last_frame,
                indices_last,
                quant_last,
                target_frame=target_frame,
                training=training,
                sample=sample,
                temperature=temperature,
            )
            predictions = predictions.write(current_frame_index, y_pred)

            if training and self.epoch < self.stop_ground_truth_after_epoch:
                start_index = tf.math.maximum(
                    0, current_frame_index - self.n_frames_before
                )
                previous_frames = ground_truth[
                    :, start_index + 1 : current_frame_index + 1
                ]
            else:
                previous_frames = predictions.stack()
                previous_frames = tf.transpose(previous_frames, (1, 0, 2, 3, 4))
                previous_frames = previous_frames[:, -self.n_frames_before :]

            current_frame_index = tf.add(current_frame_index, 1)
            total_loss = tf.add(total_loss, losses[0])
            scce_loss = tf.add(scce_loss, losses[1])
            perceptual_loss = tf.add(perceptual_loss, losses[2])

        predictions = predictions.stack()
        predictions = tf.transpose(predictions, (1, 0, 2, 3, 4))

        total_loss = tf.divide(total_loss, tf.cast(n_frames, tf.float32))
        scce_loss = tf.divide(scce_loss, tf.cast(n_frames, tf.float32))
        perceptual_loss = tf.divide(perceptual_loss, tf.cast(n_frames, tf.float32))

        if return_losses:
            return predictions, total_loss, scce_loss, perceptual_loss
        else:
            return predictions

    def predict_next_frame(
        self,
        remaining_frames,
        previous_frames,
        last_frame,
        indices_last,
        quant_last,
        target_frame=None,
        training=False,
        sample=False,
        temperature=1.0,
    ):
        # previous frames is of shape (batch_size, n_frames, height, width, 3)
        previous_frames = tf.transpose(previous_frames, (1, 0, 2, 3, 4))
        # previous frames is now of shape (n_frames, batch_size, height, width, 3)

        indices_previous = tf.map_fn(
            lambda x: self.encode_to_z(x)[1],
            previous_frames,
            fn_output_signature=tf.int64,
        )

        # indices is of shape (n_frames, batch_size, n_z)
        indices_previous = tf.transpose(indices_previous, (1, 0, 2))
        # indices is now of shape (batch_size, n_frames, n_z)
        batch_size, n_frames, n_z = (
            tf.shape(indices_previous)[0],
            tf.shape(indices_previous)[1],
            tf.shape(indices_previous)[2],
        )
        indices_previous = tf.reshape(
            indices_previous, shape=(batch_size, n_frames * n_z)
        )

        if target_frame is not None:
            _, target_indices = self.encode_to_z(target_frame)
        else:
            target_indices = None

        if training:
            next_frame, losses = self.train_predict_next_frame(
                remaining_frames,
                indices_last,
                indices_previous,
                target_indices=target_indices,
                target_frame=target_frame,
                quant_shape=tf.shape(quant_last),
                indices_shape=tf.shape(indices_last),
            )
        else:
            next_frame, losses = self.predict_next_frame_body(
                remaining_frames,
                indices_last,
                indices_previous,
                target_indices=target_indices,
                target_frame=target_frame,
                quant_shape=tf.shape(quant_last),
                indices_shape=tf.shape(indices_last),
                sample=sample,
                temperature=temperature,
            )

        return next_frame, losses

    def predict_next_frame_body(
        self,
        remaining_frames,
        last_frame_indices,
        previous_frame_indices,
        quant_shape,
        indices_shape,
        target_indices=None,
        target_frame=None,
        sample=False,
        temperature=1.0,
    ):
        logits = self.transformer(
            (remaining_frames, last_frame_indices, previous_frame_indices)
        )
        next_frame = self.convert_logits_to_image(
            logits,
            quant_shape=quant_shape,
            indices_shape=indices_shape,
            sample=sample,
            temperature=temperature,
        )
        if target_indices is not None:
            scce_loss = self.scce_loss(target_indices, logits)
        else:
            scce_loss = 0.0

        if target_frame is not None:
            perceptual_loss = 1.0 * self.perceptual_loss(target_frame, next_frame)
        else:
            perceptual_loss = 0.0

        frame_loss = scce_loss + perceptual_loss

        # self.total_loss_tracker.update_state(frame_loss)
        # self.scce_loss_tracker.update_state(scce_loss)
        # self.perceptual_loss_tracker.update_state(perceptual_loss)

        return next_frame, (frame_loss, scce_loss, perceptual_loss)

    def train_predict_next_frame(
        self,
        remaining_frames,
        last_frame_indices,
        previous_frame_indices,
        quant_shape,
        indices_shape,
        target_indices,
        target_frame,
    ):
        with tf.GradientTape() as tape:
            next_frame, losses = self.predict_next_frame_body(
                remaining_frames=remaining_frames,
                last_frame_indices=last_frame_indices,
                previous_frame_indices=previous_frame_indices,
                target_indices=target_indices,
                quant_shape=quant_shape,
                indices_shape=indices_shape,
                target_frame=target_frame,
                sample=False,
            )
        frame_loss = losses[0]
        # Calculate batch gradients
        gradients = tape.gradient(frame_loss, self.transformer.trainable_variables)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(tf.cast(gradients[i], tf.float32))

        return next_frame, losses

    def convert_logits_to_image(
        self, logits, quant_shape, indices_shape, sample=False, temperature=1.0
    ):
        if sample:
            array = []
            for i in range(logits.shape[1]):
                sub_logits = logits[:, i]
                sub_logits = sub_logits / temperature
                # sub_logits, _ = tf.math.top_k(sub_logits, k=1)
                probs = tf.keras.activations.softmax(sub_logits)
                probs, probs_index = tf.math.top_k(probs, k=50)
                selection_index = tf.random.categorical(
                    tf.math.log(probs), num_samples=1
                )
                ix = tf.gather_nd(probs_index, selection_index, batch_dims=1)
                ix = tf.reshape(ix, (-1, 1))
                array.append(ix)
            generated_indices = tf.concat(array, axis=-1)
        else:
            probs = tf.keras.activations.softmax(logits)
            _, generated_indices = tf.math.top_k(probs)

        generated_indices = tf.reshape(
            generated_indices,
            indices_shape,
        )
        quant = self.first_stage_model.quantize.get_codebook_entry(
            generated_indices, shape=quant_shape
        )

        return self.first_stage_model.decode(quant)

    def train_step(self, data):

        batch_total_loss, batch_scce_loss, batch_perceptual_loss = 0.0, 0.0, 0.0
        for i in range(self.accumulation_size):
            sub_data = {
                key: value[
                    self.accumulation_size * i : self.accumulation_size * (i + 1)
                ]
                for key, value in data.items()
            }
            _, total_loss, scce_loss, perceptual_loss = self(
                sub_data, training=True, return_losses=True
            )
            batch_total_loss += total_loss
            batch_scce_loss += scce_loss
            batch_perceptual_loss += perceptual_loss

        self.apply_accu_gradients()
        self.total_loss_tracker.update_state(batch_total_loss)
        self.scce_loss_tracker.update_state(batch_scce_loss)
        self.perceptual_loss_tracker.update_state(batch_perceptual_loss)
        self.epoch += 1
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        _, total_loss, scce_loss, perceptual_loss = self(
            data, training=False, return_losses=True
        )

        self.total_loss_tracker.update_state(total_loss)
        self.scce_loss_tracker.update_state(scce_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        return {m.name: m.result() for m in self.metrics}
