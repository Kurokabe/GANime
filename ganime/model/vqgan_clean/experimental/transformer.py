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
        last_frame_indices, previous_frame_indices = inputs
        # last_frame_indices = tf.expand_dims(last_frame_indices, axis=1)
        # previous_frame_indices = tf.expand_dims(previous_frame_indices, axis=1)
        shape_to_keep = tf.shape(previous_frame_indices)[1]
        h = tf.concat([last_frame_indices, previous_frame_indices], axis=1)

        h = self.transformer(h[:, :-1], training=training)
        # h = h.logits
        h = h.last_hidden_state
        h = h[:, -shape_to_keep:]
        return h