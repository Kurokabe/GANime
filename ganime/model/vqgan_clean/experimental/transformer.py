from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
from transformers import TFPreTrainedModel


class Transformer(Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.remaining_frames_method = self.get_remaining_frames_method(config)
        self.transformer = self.load_transformer(self.remaining_frames_method)

    def get_remaining_frames_method(self, config) -> str:
        """Get the method to use for remaining frames.
        Check if the method is set inside the configuration, otherwise use concat as the default.
        """
        if "remaining_frames_method" in config:
            return config["remaining_frames_method"]
        else:
            return "concat"

    def load_transformer(self, method) -> TFPreTrainedModel:
        print("using method ", method)
        if method == "own_embeddings":
            from ganime.model.vqgan_clean.experimental.gpt2_embedding import (
                TFGPT2LMHeadModel,
            )

            transformer = TFGPT2LMHeadModel.from_pretrained("gpt2-large")

        else:
            from transformers import TFGPT2LMHeadModel

            transformer = TFGPT2LMHeadModel.from_pretrained("gpt2-large")
        return transformer

    def concatenate_inputs(
        self, remaining_frames, last_frame_indices, previous_frame_indices
    ) -> tf.Tensor:
        if self.remaining_frames_method == "concat":
            return tf.concat(
                [remaining_frames, last_frame_indices, previous_frame_indices], axis=1
            )
        else:
            return tf.concat([last_frame_indices, previous_frame_indices], axis=1)

    def call_transformer(
        self, transformer_input, remaining_frames, training, attention_mask
    ):
        if self.remaining_frames_method == "concat":
            return self.transformer(
                transformer_input, training=training, attention_mask=attention_mask
            )
        elif self.remaining_frames_method == "token_type_ids":
            return self.transformer(
                transformer_input,
                token_type_ids=remaining_frames,
                training=training,
                attention_mask=attention_mask,
            )
        elif self.remaining_frames_method == "own_embeddings":
            return self.transformer(
                transformer_input,
                remaining_frames_ids=remaining_frames,
                training=training,
                attention_mask=attention_mask,
            )
        else:
            raise ValueError(
                f"Unknown remaining_frames_method {self.remaining_frames_method}"
            )

    def call(self, inputs, training=True, mask=None):
        remaining_frames, last_frame_indices, previous_frame_indices = inputs
        remaining_frames = tf.expand_dims(remaining_frames, axis=1)
        shape_to_keep = tf.shape(last_frame_indices)[1]

        h = self.concatenate_inputs(
            remaining_frames, last_frame_indices, previous_frame_indices
        )

        # transformer_input = h[:, :-1]
        transformer_input = h
        mask = tf.ones_like(transformer_input) * tf.cast(
            tf.cast(remaining_frames, dtype=tf.bool), dtype=remaining_frames.dtype
        )

        h = self.call_transformer(transformer_input, remaining_frames, training, mask)
        h = h.logits
        # h = self.transformer.transformer.wte(h, mode="linear")
        h = h[:, -shape_to_keep:]
        return h
