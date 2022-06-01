import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from ganime.configs.model_configs import GPTConfig, ModelConfig
from ganime.model.vqgan_clean.transformer.mingpt import GPT
from ganime.model.vqgan_clean.vqgan import VQGAN
from tensorflow import keras
from tensorflow.keras import Model, layers


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a warmup cosine decay schedule."""

    def __init__(self, lr_start, lr_max, warmup_steps, total_steps):
        """
        Args:
            lr_start: The initial learning rate
            lr_max: The maximum learning rate to which lr should increase to in
                the warmup steps
            warmup_steps: The number of steps for which the model warms up
            total_steps: The total number of steps for the model training
        """
        super().__init__()
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        # Check whether the total number of steps is larger than the warmup
        # steps. If not, then throw a value error.
        if self.total_steps < self.warmup_steps:
            raise ValueError(
                f"Total number of steps {self.total_steps} must be"
                + f"larger or equal to warmup steps {self.warmup_steps}."
            )

        # `cos_annealed_lr` is a graph that increases to 1 from the initial
        # step to the warmup step. After that this graph decays to -1 at the
        # final step mark.
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        )

        # Shift the mean of the `cos_annealed_lr` graph to 1. Now the grpah goes
        # from 0 to 2. Normalize the graph with 0.5 so that now it goes from 0
        # to 1. With the normalized graph we scale it with `lr_max` such that
        # it goes from 0 to `lr_max`
        learning_rate = 0.5 * self.lr_max * (1 + cos_annealed_lr)

        # Check whether warmup_steps is more than 0.
        if self.warmup_steps > 0:
            # Check whether lr_max is larger that lr_start. If not, throw a value
            # error.
            if self.lr_max < self.lr_start:
                raise ValueError(
                    f"lr_start {self.lr_start} must be smaller or"
                    + f"equal to lr_max {self.lr_max}."
                )

            # Calculate the slope with which the learning rate should increase
            # in the warumup schedule. The formula for slope is m = ((b-a)/steps)
            slope = (self.lr_max - self.lr_start) / self.warmup_steps

            # With the formula for a straight line (y = mx+c) build the warmup
            # schedule
            warmup_rate = slope * tf.cast(step, tf.float32) + self.lr_start

            # When the current step is lesser that warmup steps, get the line
            # graph. When the current step is greater than the warmup steps, get
            # the scaled cos graph.
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )

        # When the current step is more that the total steps, return 0 else return
        # the calculated graph.
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


LEN_X_TRAIN = 9000
BATCH_SIZE = 128
N_EPOCHS = 300
TOTAL_STEPS = int(LEN_X_TRAIN / BATCH_SIZE * N_EPOCHS)
WARMUP_EPOCH_PERCENTAGE = 0.15
WARMUP_STEPS = int(TOTAL_STEPS * WARMUP_EPOCH_PERCENTAGE)


class Net2Net(Model):
    def __init__(
        self,
        transformer_config: GPTConfig,
        first_stage_config: ModelConfig,
        cond_stage_config: ModelConfig,
    ):
        super().__init__()
        self.transformer = GPT(**transformer_config)
        self.first_stage_model = VQGAN(**first_stage_config)
        self.cond_stage_model = self.first_stage_model  # VQGAN(**cond_stage_config)

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        # self.compile(
        #     "adam",
        #     loss=self.loss_fn,
        # )

        # Calculate the number of steps for warmup.

        # Initialize the warmupcosine schedule.
        self.scheduled_lrs = WarmUpCosine(
            lr_start=1e-5,
            lr_max=2.5e-4,
            warmup_steps=WARMUP_STEPS,
            total_steps=TOTAL_STEPS,
        )

        self.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=self.scheduled_lrs, weight_decay=1e-4
            ),
            loss=[self.loss_fn, None],
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

    def encode_to_z(self, x):
        quant_z, indices = self.first_stage_model.encode(x)

        batch_size = tf.shape(quant_z)[0]

        indices = tf.reshape(indices, shape=(batch_size, -1))
        return quant_z, indices

    def encode_to_c(self, c):
        quant_c, indices = self.cond_stage_model.encode(c)

        batch_size = tf.shape(quant_c)[0]

        indices = tf.reshape(indices, shape=(batch_size, -1))
        return quant_c, indices

    # def build(self, input_shape):
    #     self.first_stage_model.build(input_shape)
    #     self.cond_stage_model.build(input_shape)
    #     return super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        # x, c = inputs

        # # one step to produce the logits
        # _, z_indices = self.encode_to_z(x)
        # _, c_indices = self.encode_to_c(c)

        # cz_indices = tf.concat((c_indices, z_indices), axis=1)

        # target = z_indices
        # logits = self.transformer(
        #     cz_indices[:, :-1]  # , training=training
        # )  # don't know why -1

        # logits = logits[:, tf.shape(c_indices)[1] - 1 :]  # -1 here 'cause -1 above

        # logits = tf.reshape(logits, shape=(-1, logits.shape[-1]))
        # target = tf.reshape(target, shape=(-1,))

        # return logits, target

        X, y = inputs
        first_frame = X[:, 0]
        last_frame = X[:, -1]

        return self.process_video(first_frame, last_frame, y)

    def process_image(self, x, c, target_image=None):

        frame_loss = 0

        # one step to produce the logits
        quant_z, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        cz_indices = tf.concat((c_indices, z_indices), axis=1)

        logits = self.transformer(
            cz_indices[:, :-1]  # , training=training
        )  # don't know why -1

        # Remove the conditioned part
        logits = logits[:, tf.shape(c_indices)[1] - 1 :]  # -1 here 'cause -1 above

        logits = tf.reshape(logits, shape=(-1, logits.shape[-1]))

        if target_image is not None:
            _, target_indices = self.encode_to_z(target_image)
            target_indices = tf.reshape(target_indices, shape=(-1,))

            frame_loss = tf.reduce_mean(
                self.loss_fn(y_true=target_indices, y_pred=logits)
            )

        image = self.get_image(logits, tf.shape(quant_z))

        return image, frame_loss

    def process_video(self, first_frame, last_frame, target):
        x = first_frame
        c = last_frame

        loss = 0
        generated_video = [x]
        for i in range(19):  # TODO change 19 to the number of frame in the video
            generated_image, frame_loss = self.process_image(
                x, c, target_image=target[:, i, ...]
            )
            x = generated_image
            generated_video.append(generated_image)
            loss += frame_loss
        return tf.stack(generated_video, axis=1), loss

    def train_step(self, data):

        X, y = data
        first_frame = X[:, 0]
        last_frame = X[:, -1]

        with tf.GradientTape() as tape:
            generated_video, loss = self.process_video(first_frame, last_frame, y)
        grads = tape.gradient(
            loss,
            self.transformer.trainable_variables,
        )
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        # Log results.
        return {m.name: m.result() for m in self.metrics}

    def get_image(self, logits, shape):
        probs = tf.keras.activations.softmax(logits)
        _, generated_indices = tf.math.top_k(probs)
        generated_indices = tf.reshape(
            generated_indices, (-1,)  # , self.first_stage_model.quantize.num_embeddings)
        )
        quant = self.first_stage_model.quantize.get_codebook_entry(
            generated_indices, shape=shape
        )
        return self.first_stage_model.decode(quant)

    def test_step(self, data):
        loss = 0
        # for i in range(y.shape[1]):
        logits, target = self.call(data, training=True)
        loss += tf.reduce_mean(self.loss_fn(y_true=target, y_pred=logits))
        self.loss_tracker.update_state(loss)

        # Log results.
        return {m.name: m.result() for m in self.metrics}

    def decode_to_img(self, index, zshape):
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            tf.reshape(index, -1), shape=zshape
        )
        x = self.first_stage_model.decode(quant_z)
        return x
