import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from ganime.configs.model_configs import GPTConfig, ModelConfig
from ganime.model.vqgan_clean.vqgan import VQGAN
from ganime.trainer.warmup.cosine import WarmUpCosine
from tensorflow import keras
from tensorflow.keras import Model, layers
from transformers import TFGPT2Model, GPT2Config
from tensorflow.keras import mixed_precision

class Net2Net(Model):
    def __init__(
        self,
        transformer_config: GPTConfig,
        first_stage_config: ModelConfig,
        trainer_config,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.first_stage_model = VQGAN(**first_stage_config)

        # self.encoder, self.decoder = self.load_vqgan("../../../checkpoints/tflite/encoder_quant_f16.tflite", "../../../checkpoints/tflite/decoder_quant_f16.tflite")

        # from tensorflow.keras import mixed_precision

        # self.policy = mixed_precision.Policy("mixed_float16")

        # configuration = GPT2Config(**transformer_config)
        # self.transformer = TFGPT2Model(configuration)#.from_pretrained("gpt2", **self.transformer_config)
        # configuration = GPT2Config(**transformer_config)
        self.transformer = TFGPT2Model.from_pretrained(
            "gpt2-medium"
        )  # , **transformer_config)
        if "checkpoint_path" in transformer_config:
            print(f"Restoring weights from {transformer_config['checkpoint_path']}")
            self.load_weights(transformer_config["checkpoint_path"])

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")

        self.scheduled_lrs = self.create_warmup_scheduler(trainer_config)

        # optimizer = mixed_precision.LossScaleOptimizer(tfa.optimizers.AdamW(
        #     learning_rate=self.scheduled_lrs, weight_decay=1e-4
        # ))
        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.scheduled_lrs, weight_decay=1e-4
        )
        self.compile(
            optimizer=optimizer,
            loss=self.loss_fn, 
            run_eagerly=True
        )

        # self.predict_next_recompute = tf.recompute_grad(self.predict_next_frame)

        # Gradient accumulation
        # self.n_gradients = tf.constant(20, dtype=tf.int32)
        self.gradient_accumulation = [
            tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
            for v in self.transformer.trainable_variables
        ]

    def load_vqgan(self, encoder_path: str, decoder_path: str):
        encoder = tf.lite.Interpreter(model_path=str(encoder_path))
        decoder = tf.lite.Interpreter(model_path=str(decoder_path))

        return encoder, decoder

    def create_warmup_scheduler(self, trainer_config):
        len_x_train = trainer_config["len_x_train"]
        batch_size = trainer_config["batch_size"]
        n_epochs = trainer_config["n_epochs"]

        total_steps = int(len_x_train / batch_size * n_epochs)
        warmup_epoch_percentage = trainer_config["warmup_epoch_percentage"]
        warmup_steps = int(total_steps * warmup_epoch_percentage)

        scheduled_lrs = WarmUpCosine(
            lr_start=trainer_config["lr_start"],
            lr_max=trainer_config["lr_max"],
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        return scheduled_lrs

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

    @tf.function(
        # input_signature=[
        #     tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
        # ]
    )
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

    def test_step(self, data):
        first_frame = data["first_frame"]
        last_frame = data["last_frame"]
        frames = data["y"]
        n_frames = data["n_frames"]

        predicted_logits, _, _ = self.predict_logits(first_frame, last_frame, n_frames)

        total_loss = 0.0

        for i in range(1, tf.reduce_max(n_frames).numpy()):
            target_indices = self.encode_to_z(frames[:, i, ...])[1]
            target_indices = tf.reshape(target_indices, shape=(-1,))
            logits = predicted_logits[i]

            frame_loss = tf.cast(
                tf.reduce_mean(self.loss_fn(target_indices, logits)),
                dtype=tf.float32,
            )

            total_loss += frame_loss

        self.loss_tracker.update_state(total_loss)
        return {m.name: m.result() for m in self.metrics}

    @tf.function()
    def predict_next_indices(self, inputs, example_indices):
        logits = self.transformer(inputs)
        logits = logits.last_hidden_state
        logits = tf.cast(logits, dtype=tf.float32)
        # Remove the conditioned part
        logits = logits[
            :, tf.shape(example_indices)[1] - 1 :
        ]  # -1 here 'cause -1 above
        logits = tf.reshape(logits, shape=(-1, tf.shape(logits)[-1]))
        return logits

    def train_step(self, data):
        first_frame = data["first_frame"]
        last_frame = data["last_frame"]
        frames = data["y"]
        n_frames = data["n_frames"]

        # if self.first_stage_model.built is True and self.transformer.built is False:
        #     print("Setting mixed precision", self.policy.compute_dtype)
        #     mixed_precision.set_global_policy(self.policy)

        # first_frame_indices = self.encode_to_z(first_frame)[1]
        last_frame_indices = self.encode_to_z(last_frame)[1]
        total_loss = 0.0

        # previous_frame_indices = first_frame_indices
        for i in range(1, tf.math.reduce_max(n_frames).numpy()): 
            previous_frame_indices = self.encode_to_z(frames[:, i - 1, ...])[1]
            cz_indices = tf.concat((last_frame_indices, previous_frame_indices), axis=1)
            target_indices = self.encode_to_z(frames[:, i, ...])[1]
            target_indices = tf.reshape(target_indices, shape=(-1,))

            with tf.GradientTape() as tape:
                logits = self.predict_next_indices(
                    cz_indices[:, :-1], last_frame_indices
                )  # don't know why -1

                frame_loss = tf.cast(
                    tf.reduce_mean(self.loss_fn(target_indices, logits)),
                    dtype=tf.float32,
                )

            #     scaled_loss = self.optimizer.get_scaled_loss(frame_loss)
            total_loss += frame_loss

            # scaled_gradients = tape.gradient(scaled_loss, self.transformer.trainable_variables)
            # gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

            # Calculate batch gradients
            gradients = tape.gradient(frame_loss, self.transformer.trainable_variables)

            # Accumulate batch gradients
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(
                    tf.cast(gradients[i], tf.float32)
                )

            # previous_frame_indices = self.convert_logits_to_indices(
            #     logits, tf.shape(last_frame_indices)
            # )
            # previous_frame_indices = tf.reshape(target_indices, tf.shape(last_frame_indices))

        self.apply_accu_gradients()
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

    def predict_logits(self, first_frame, last_frame, n_frames):
        quant_first, indices_first = self.encode_to_z(first_frame)
        quant_last, indices_last = self.encode_to_z(last_frame)

        indices_previous = indices_first

        predicted_logits = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )

        for i in range(1, tf.math.reduce_max(n_frames).numpy()):
            cz_indices = tf.concat((indices_last, indices_previous), axis=1)
            logits = self.predict_next_indices(cz_indices[:, :-1], indices_last)

            # generated_indices = self.convert_logits_to_indices(
            #     logits, tf.shape(indices_last)
            # )
            predicted_logits = predicted_logits.write(i, logits)
            indices_previous = self.convert_logits_to_indices(
                logits, tf.shape(indices_last)
            )

        return predicted_logits.stack(), tf.shape(quant_first), tf.shape(indices_first)

    def generate_video(self, first_frame, last_frame, n_frames):
        predicted_logits, quant_shape, indices_shape = self.predict_logits(
            first_frame, last_frame, n_frames
        )

        generated_images = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )
        generated_images = generated_images.write(0, first_frame)

        for i in range(1, len(predicted_logits)):
            indices = self.convert_logits_to_indices(predicted_logits[i], indices_shape)
            quant = self.first_stage_model.quantize.get_codebook_entry(
                indices,
                shape=quant_shape,
            )
            decoded = self.first_stage_model.decode(quant)
            generated_images = generated_images.write(i, decoded)

        stacked_images = generated_images.stack()
        videos = tf.transpose(stacked_images, (1, 0, 2, 3, 4))
        return videos
