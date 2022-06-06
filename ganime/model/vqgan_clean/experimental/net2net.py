from ganime.configs.model_configs import ModelConfig
from tensorflow.keras import Model

from transformers import TFGPT2Model
import tensorflow as tf
from tensorflow import keras

import tensorflow_addons as tfa

from ganime.model.vqgan_clean.vqgan import VQGAN
import numpy as np


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


LEN_X_TRAIN = 8000
BATCH_SIZE = 16
N_EPOCHS = 500
TOTAL_STEPS = int(LEN_X_TRAIN / BATCH_SIZE * N_EPOCHS)
WARMUP_EPOCH_PERCENTAGE = 0.15
WARMUP_STEPS = int(TOTAL_STEPS * WARMUP_EPOCH_PERCENTAGE)

from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return f"GPU memory occupied: {info.used//1024**2} MB."


class Net2Net(Model):
    def __init__(self, first_stage_config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.first_stage_model = VQGAN(**first_stage_config)
        self.transformer = TFGPT2Model.from_pretrained("gpt2")
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")

        self.scheduled_lrs = WarmUpCosine(
            lr_start=1e-5,
            lr_max=2.5e-4,
            warmup_steps=WARMUP_STEPS,
            total_steps=TOTAL_STEPS,
        )

        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.scheduled_lrs, weight_decay=1e-4
        )
        self.compile(
            optimizer=optimizer,
            loss=[self.loss_fn, None],
        )

        # self.predict_next_recompute = tf.recompute_grad(self.predict_next_frame)

        # Gradient accumulation
        # self.n_gradients = tf.constant(20, dtype=tf.int32)
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in self.transformer.trainable_variables
        ]

    def apply_accu_gradients(self):
        # tf.print("before apply")
        # print("before apply")
        # apply accumulated gradients
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.transformer.trainable_variables)
        )

        # tf.print("after apply")
        # print("after apply")
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

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
        ]
    )
    def encode_to_z(self, x):
        quant_z, indices, quantized_loss = self.first_stage_model.encode(x)

        batch_size = tf.shape(quant_z)[0]

        indices = tf.reshape(indices, shape=(batch_size, -1))
        return quant_z, indices

    def call(self, inputs, training=None, mask=None):
        return self.process_video(inputs)

    @tf.function(
        # input_signature=[
        #     tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        #     tf.TensorSpec(shape=[None], dtype=tf.int32),
        # ]
    )
    def get_image(self, logits, shape):

        probs = tf.keras.activations.softmax(logits)
        _, generated_indices = tf.math.top_k(probs)
        generated_indices = tf.reshape(
            generated_indices,
            (-1,),  # , self.first_stage_model.quantize.num_embeddings)
        )
        quant = self.first_stage_model.quantize.get_codebook_entry(
            generated_indices, shape=shape
        )

        return self.first_stage_model.decode(quant)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
        ]
    )
    def predict_next_frame(self, previous_frame, end_frame):
        quant_z, z_indices = self.encode_to_z(previous_frame)
        quant_c, c_indices = self.encode_to_z(end_frame)

        cz_indices = tf.concat((c_indices, z_indices), axis=1)
        logits = self.transformer(cz_indices[:, :-1])  # don't know why -1

        logits = logits.last_hidden_state
        # print(logits)

        # Remove the conditioned part
        logits = logits[:, tf.shape(c_indices)[1] - 1 :]  # -1 here 'cause -1 above

        logits = tf.reshape(logits, shape=(-1, tf.shape(logits)[-1]))
        # next_frame = self.get_image(logits[:, :256], tf.shape(quant_z))

        return logits

    @tf.function()
    def process_video(self, first_frame, end_frame, n_frames, target):
        total_loss = 0.0
        previous_frame = first_frame

        # get trainable variables
        # train_vars = self.transformer.trainable_variables
        # # Create empty gradient list (not a tf.Variable list)
        # accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]

        for i in tf.range(tf.math.reduce_max(n_frames)):

            # for i in range(1, 20):
            target_frame = target[:, i, :, :, :]

            quant_z, target_indices = self.encode_to_z(target_frame)

            with tf.GradientTape() as tape:
                logits = self.predict_next_frame(previous_frame, end_frame)
                # logits = self.predict_next_recompute(self, previous_frame, end_frame)
                frame_loss = tf.reduce_mean(self.loss_fn(target_indices, logits))

            total_loss += frame_loss
            # Calculate batch gradients
            gradients = tape.gradient(frame_loss, self.transformer.trainable_variables)
            # Accumulate batch gradients
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(
                    tf.cast(gradients[i], tf.float32)
                )

            previous_frame = self.get_image(logits, tf.shape(quant_z))
            previous_frame = tf.reshape(previous_frame, tf.shape(first_frame))

        self.apply_accu_gradients()
        self.loss_tracker.update_state(total_loss)

        # return total_loss

    def train_step(self, data):
        frames = data["video"]
        n_frames = data["n_frames"]

        first_frame = frames[:, 0, :, :, :]
        end_frame = frames[:, -1, :, :, :]

        self.process_video(first_frame, end_frame, n_frames, target=frames)

        return {m.name: m.result() for m in self.metrics}
