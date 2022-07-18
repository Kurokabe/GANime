from tensorflow.keras import layers
from tensorflow.keras import Model
from transformers import TFGPT2Model, TFGPT2LMHeadModel, GPT2Config
import tensorflow as tf


class Transformer(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = TFGPT2Model.from_pretrained("gpt2-medium")

    def call(self, inputs, training=True, mask=None):
        remaining_frames, last_frame_indices, previous_frame_indices = inputs
        # last_frame_indices = tf.expand_dims(last_frame_indices, axis=1)
        # previous_frame_indices = tf.expand_dims(previous_frame_indices, axis=1)
        # tf.print("last_frame_indices", tf.shape(last_frame_indices))
        # tf.print("previous_frame_indices", tf.shape(previous_frame_indices))
        remaining_frames = tf.expand_dims(remaining_frames, axis=1)
        shape_to_keep = tf.shape(last_frame_indices)[1]
        h = tf.concat([last_frame_indices, previous_frame_indices], axis=1)

        transformer_input = h[:, :-1]
        mask = tf.ones_like(transformer_input) * tf.cast(
            tf.cast(remaining_frames, dtype=tf.bool), dtype=remaining_frames.dtype
        )

        h = self.transformer(
            transformer_input,
            remaining_frames_ids=remaining_frames,
            training=training,
            attention_mask=mask,
        )
        # h = h.logits
        h = h.last_hidden_state
        h = h[:, -shape_to_keep:]
        # tf.print("h", tf.shape(h))
        return h
