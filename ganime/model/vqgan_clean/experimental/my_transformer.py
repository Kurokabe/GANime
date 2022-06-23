from tensorflow.keras import layers
from tensorflow.keras import Model
from transformers import TFGPT2Model
import tensorflow as tf


class MyTransformer(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = TFGPT2Model.from_pretrained("gpt2-medium")

    def call(self, inputs):
        num_remaining_frames, last_frame, previous_frame = inputs

        h = tf.concat([num_remaining_frames, last_frame, previous_frame], axis=1)
        h = self.transformer(h[:, :-1])
        h = h.last_hidden_state
        truncate_idx = tf.shape(last_frame)[1]
        # outputs = tf.slice(h, -truncate_idx, truncate_idx)
        outputs = h[:, -truncate_idx:]
        return outputs

    # def get_model(self):
    #     inputs_remaining_frames = layers.Input(
    #         shape=(None, None), dtype=tf.int32, name="remaining_frames"
    #     )
    #     inputs_last = layers.Input(
    #         shape=(None, None), dtype=tf.int32, name="last_frame_indices"
    #     )
    #     inputs_previous = layers.Input(
    #         shape=(None, None), dtype=tf.int32, name="previous_frame_indices"
    #     )
    #     inputs = layers.Concatenate()(
    #         [inputs_remaining_frames, inputs_last, inputs_previous]
    #     )
    #     results = self.transformer.layers[-1](inputs)
    #     outputs = layers.Lambda(
    #         lambda output: output[: -tf.shape(inputs_last.shape)[1] :]
    #     )(results)
    #     return Model(
    #         inputs=[inputs_last, inputs_previous, inputs_remaining_frames],
    #         outputs=outputs,
    #     )
