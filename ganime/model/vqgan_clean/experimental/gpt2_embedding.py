# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TF 2.0 OpenAI GPT-2 model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice


from transformers.activations_tf import get_tf_activation
from transformers.modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFSequenceClassifierOutputWithPast,
)
from transformers.modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFConv1D,
    TFModelInputType,
    TFPreTrainedModel,
    TFSequenceClassificationLoss,
    TFSequenceSummary,
    TFSharedEmbeddings,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from transformers.tf_utils import shape_list, stable_softmax
from transformers.utils import (
    DUMMY_INPUTS,
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import GPT2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]


class TFAttention(tf.keras.layers.Layer):
    def __init__(self, nx, config, scale=False, is_cross_attention=False, **kwargs):
        super().__init__(**kwargs)

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implementation]
        assert n_state % config.n_head == 0
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions

        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = TFConv1D(
                n_state * 2,
                nx,
                initializer_range=config.initializer_range,
                name="c_attn",
            )
            self.q_attn = TFConv1D(
                n_state, nx, initializer_range=config.initializer_range, name="q_attn"
            )
        else:
            self.c_attn = TFConv1D(
                n_state * 3,
                nx,
                initializer_range=config.initializer_range,
                name="c_attn",
            )

        self.c_proj = TFConv1D(
            n_state, nx, initializer_range=config.initializer_range, name="c_proj"
        )
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        pass

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """
        1's in the lower triangle, counting from the lower right corner. Same as tf.matrix_band_part(tf.ones([nd, ns]),
        -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

    def _attn(
        self, q, k, v, attention_mask, head_mask, output_attentions, training=False
    ):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], dtype=w.dtype)  # scale attention_scores
            w = w / tf.math.sqrt(dk)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask

            # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
            _, _, nd, ns = shape_list(w)
            b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
            b = tf.reshape(b, [1, 1, nd, ns])
            w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            attention_mask = tf.cast(attention_mask, dtype=w.dtype)
            w = w + attention_mask

        w = stable_softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))  # (batch, head, seq_length, head_features)

    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
        training=False,
    ):

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(x)
            kv_out = self.c_attn(encoder_hidden_states)
            key, value = tf.split(kv_out, 2, axis=2)
            attention_mask = encoder_attention_mask
        else:
            x = self.c_attn(x)
            query, key, value = tf.split(x, 3, axis=2)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        # to cope with keras serialization
        if use_cache:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None,)

        attn_outputs = self._attn(
            query,
            key,
            value,
            attention_mask,
            head_mask,
            output_attentions,
            training=training,
        )
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class TFMLP(tf.keras.layers.Layer):
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(
            n_state, nx, initializer_range=config.initializer_range, name="c_fc"
        )
        self.c_proj = TFConv1D(
            nx, n_state, initializer_range=config.initializer_range, name="c_proj"
        )
        self.act = get_tf_activation(config.activation_function)
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)

    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2


class TFBlock(tf.keras.layers.Layer):
    def __init__(self, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * nx
        self.ln_1 = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_1"
        )
        self.attn = TFAttention(nx, config, scale, name="attn")
        self.ln_2 = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_2"
        )

        if config.add_cross_attention:

            self.crossattention = TFAttention(
                nx, config, scale, name="crossattention", is_cross_attention=True
            )
            self.ln_cross_attn = tf.keras.layers.LayerNormalization(
                epsilon=config.layer_norm_epsilon, name="ln_cross_attn"
            )

        self.mlp = TFMLP(inner_dim, config, name="mlp")

    def call(
        self,
        x,
        layer_past,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        use_cache,
        output_attentions,
        training=False,
    ):
        a = self.ln_1(x)
        output_attn = self.attn(
            a,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        outputs = output_attn[1:]
        x = x + a

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )

            ca = self.ln_cross_attn(x)
            output_cross_attn = self.crossattention(
                ca,
                layer_past=None,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=False,
                output_attentions=output_attentions,
                training=training,
            )
            ca = output_cross_attn[0]  # output_attn: a, present, (cross_attentions)
            x = x + ca
            outputs = (
                outputs + output_cross_attn[2:]
            )  # add cross attentions if we output attention weights

        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m

        outputs = [x] + outputs
        return outputs  # x, present, (attentions, cross_attentions)


@keras_serializable
class TFGPT2MainLayer(tf.keras.layers.Layer):
    config_class = GPT2Config

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict

        self.num_hidden_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.n_positions = config.n_positions
        self.initializer_range = config.initializer_range

        self.wte = TFSharedEmbeddings(
            config.vocab_size,
            config.hidden_size,
            initializer_range=config.initializer_range,
            name="wte",
        )

        self.wte_remaining_frames = TFSharedEmbeddings(
            config.vocab_size,
            config.hidden_size,
            initializer_range=config.initializer_range,
            name="wte_remaining_frames",
        )
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [
            TFBlock(config, scale=True, name=f"h_._{i}") for i in range(config.n_layer)
        ]
        self.ln_f = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_epsilon, name="ln_f"
        )

    def build(self, input_shape):
        with tf.name_scope("wpe"):
            self.wpe = self.add_weight(
                name="embeddings",
                shape=[self.n_positions, self.n_embd],
                initializer=get_initializer(self.initializer_range),
            )
        self.wte_remaining_frames.build(input_shape)

        super().build(input_shape)

    def get_input_embeddings(self):
        return self.wte

    def get_remaining_frames_embeddings(self):
        return self.wte_remaining_frames

    def set_input_embeddings(self, value):
        self.wte.weight = value
        self.wte.vocab_size = shape_list(value)[0]

    def set_remaining_frames_embeddings(self, value):
        self.wte_remaining_frames.weight = value
        self.wte_remaining_frames.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        remaining_frames_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        past: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = shape_list(past[0][0])[-2]

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(past_length, input_shape[-1] + past_length), axis=0
            )

        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask_shape = shape_list(attention_mask)
            attention_mask = tf.reshape(
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            one_cst = tf.constant(1.0)
            attention_mask = tf.cast(attention_mask, dtype=one_cst.dtype)
            attention_mask = tf.multiply(
                tf.subtract(one_cst, attention_mask), tf.constant(-10000.0)
            )

        # Copied from `modeling_tf_t5.py` with -1e9 -> -10000
        if self.config.add_cross_attention and encoder_attention_mask is not None:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, mask_seq_length, mask_seq_length]
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            encoder_attention_mask = tf.cast(
                encoder_attention_mask, dtype=encoder_hidden_states.dtype
            )
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[
                    :, None, None, :
                ]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = tf.math.equal(encoder_extended_attention_mask,
            #                                         tf.transpose(encoder_extended_attention_mask, perm=(-1, -2)))

            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask
            ) * -10000.0
        else:
            encoder_extended_attention_mask = None

        encoder_attention_mask = encoder_extended_attention_mask

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
            # head_mask = tf.constant([0] * self.num_hidden_layers)

        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids, mode="embedding")

        position_embeds = tf.gather(self.wpe, position_ids)

        if token_type_ids is not None:
            token_type_ids = tf.reshape(
                token_type_ids, [-1, shape_list(token_type_ids)[-1]]
            )
            token_type_embeds = self.wte(token_type_ids, mode="embedding")
        else:
            token_type_embeds = tf.constant(0.0)

        if remaining_frames_ids is not None:
            remaining_frames_ids = tf.reshape(
                remaining_frames_ids, [-1, shape_list(remaining_frames_ids)[-1]]
            )
            remaining_frames_embeds = self.wte_remaining_frames(
                remaining_frames_ids, mode="embedding"
            )
        else:
            remaining_frames_embeds = tf.constant(0.0)

        position_embeds = tf.cast(position_embeds, dtype=inputs_embeds.dtype)
        token_type_embeds = tf.cast(token_type_embeds, dtype=inputs_embeds.dtype)
        remaining_frames_embeds = tf.cast(
            remaining_frames_embeds, dtype=inputs_embeds.dtype
        )
        hidden_states = (
            inputs_embeds
            + position_embeds
            + token_type_embeds
            + remaining_frames_embeds
        )
        hidden_states = self.drop(hidden_states, training=training)

        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    tf.reshape(hidden_states, output_shape),
                )

            outputs = block(
                hidden_states,
                layer_past,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
                training=training,
            )

            hidden_states, present = outputs[:2]
            if use_cache:
                presents = presents + (present,)

            if output_attentions:
                all_attentions = all_attentions + (outputs[2],)
                if (
                    self.config.add_cross_attention
                    and encoder_hidden_states is not None
                ):
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = tf.reshape(hidden_states, output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = (
                input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            )
            all_attentions = tuple(
                tf.reshape(t, attention_output_shape) for t in all_attentions
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class TFGPT2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    base_model_prefix = "transformer"
    # names with a '.' represents the authorized unexpected/missing layers when a TF model is loaded from a PT model
    _keys_to_ignore_on_load_unexpected = [
        r"h.\d+.attn.bias",
        r"h.\d+.crossattention.bias",
    ]

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        dummy = {"input_ids": tf.constant(DUMMY_INPUTS)}
        # Add `encoder_hidden_states` to make the cross-attention layers' weights initialized
        if self.config.add_cross_attention:
            batch_size, seq_len = tf.constant(DUMMY_INPUTS).shape
            shape = (batch_size, seq_len) + (self.config.hidden_size,)
            h = tf.random.uniform(shape=shape)
            dummy["encoder_hidden_states"] = h

        return dummy

    @tf.function(
        input_signature=[
            {
                "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
                "attention_mask": tf.TensorSpec(
                    (None, None), tf.int32, name="attention_mask"
                ),
            }
        ]
    )
    def serving(self, inputs):
        output = self.call(inputs)

        return self.serving_output(output)


@dataclass
class TFGPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (`tf.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (`tf.Tensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (`List[tf.Tensor]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads,
            sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: tf.Tensor = None
    mc_logits: tf.Tensor = None
    past_key_values: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None


GPT2_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>

    TF 2.0 models accepts two formats as inputs:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional arguments.

    This second option is useful when using [`tf.keras.Model.fit`] method which currently requires having all the
    tensors in the first argument of the model call function: `model(inputs)`.

    If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the
    first positional argument :

    - a single Tensor with `input_ids` only and nothing else: `model(inputs_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    </Tip>

    Parameters:
        config ([`GPT2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past` is `None` else `past[0].shape[-2]` (`sequence_length` of
            input past key value states). Indices of input sequence tokens in the vocabulary.

            If `past` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.

            Indices can be obtained using [`GPT2Tokenizer`]. See [`PreTrainedTokenizer.__call__`] and
            [`PreTrainedTokenizer.encode`] for details.

            [What are input IDs?](../glossary#input-ids)
        past (`List[tf.Tensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past` output below). Can be used to speed up sequential decoding. The token ids which have their past
            given to this model should not be passed as input ids as they have already been computed.
        attention_mask (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the
            config will be used instead.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. This argument can be used only in eager mode, in graph mode the value in the config will be
            used instead.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. This argument can be used in
            eager mode, in graph mode the value will always be set to True.
        training (`bool`, *optional*, defaults to `False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class TFGPT2Model(TFGPT2PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPT2MainLayer(config, name="transformer")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: Optional[TFModelInputType] = None,
        remaining_frames_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        past: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        token_type_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        position_ids: Optional[Union[np.ndarray, tf.Tensor]] = None,
        head_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        inputs_embeds: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_hidden_states: Optional[Union[np.ndarray, tf.Tensor]] = None,
        encoder_attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        r"""
        encoder_hidden_states  (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        past (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`)
            contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past` are used, the user can optionally input only the last `decoder_input_ids` (those that don't have
            their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*, defaults to `True`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past`). Set to `False` during training, `True` during generation
        """

        outputs = self.transformer(
            input_ids=input_ids,
            remaining_frames_ids=remaining_frames_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    def serving_output(self, output):
        pkv = (
            tf.convert_to_tensor(output.past_key_values)
            if self.config.use_cache
            else None
        )
        hs = (
            tf.convert_to_tensor(output.hidden_states)
            if self.config.output_hidden_states
            else None
        )
        attns = (
            tf.convert_to_tensor(output.attentions)
            if self.config.output_attentions
            else None
        )
        cross_attns = (
            tf.convert_to_tensor(output.cross_attentions)
            if self.config.output_attentions
            and self.config.add_cross_attention
            and output.cross_attentions is not None
            else None
        )

        return TFBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output.last_hidden_state,
            past_key_values=pkv,
            hidden_states=hs,
            attentions=attns,
            cross_attentions=cross_attns,
        )
