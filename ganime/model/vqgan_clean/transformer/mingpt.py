import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers
from ganime.utils.recompute_grad import recompute_grad


class TransformerBlock(layers.Layer):
    def __init__(self, n_embedding, n_head, attention_percentage_drop):
        super().__init__()
        self.att = layers.MultiHeadAttention(n_head, n_embedding)
        self.ffn = Sequential(
            [
                layers.Dense(n_embedding * 4, activation="relu"),
                layers.Dense(n_embedding),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(attention_percentage_drop)
        self.dropout2 = layers.Dropout(attention_percentage_drop)

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """
        Mask the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TransformerBlockV2(layers.Layer):
    def __init__(self, n_embedding, n_head, attention_percentage_drop):
        super().__init__()
        self.att = layers.MultiHeadAttention(n_head, n_embedding)
        self.mlp = Sequential(
            [
                layers.Dense(n_embedding * 4),
                layers.Activation("gelu"),
                layers.Dense(n_embedding),
                layers.Dropout(attention_percentage_drop),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """
        Mask the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        h = inputs
        h = self.layernorm1(h)
        h = self.att(h, h, attention_mask=causal_mask)

        h = inputs + h
        h = h + self.mlp(self.layernorm2(h))
        return h


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, block_size, vocab_size, n_embedding, embedding_percentage_drop):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=n_embedding)
        self.pos_emb = layers.Embedding(input_dim=block_size, output_dim=n_embedding)
        self.dropout = layers.Dropout(embedding_percentage_drop)

    def call(self, x, training=None, mask=None):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return self.dropout(x + positions, training=training)


class GPT(Model):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer,
        n_head,
        n_embedding,
        embedding_percentage_drop,
        attention_percentage_drop,
    ):
        super().__init__()
        self.block_size = block_size
        self.embedding_layer = TokenAndPositionEmbedding(
            block_size=block_size,
            vocab_size=vocab_size,
            n_embedding=n_embedding,
            embedding_percentage_drop=embedding_percentage_drop,
        )
        self.blocks = [
            recompute_grad(
                TransformerBlock(
                    n_embedding=n_embedding,
                    n_head=n_head,
                    attention_percentage_drop=attention_percentage_drop,
                )
            )
            for _ in range(n_layer)
        ]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.outputs = layers.Dense(vocab_size)

        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.compile(
        #     "adam",
        #     loss=loss_fn,
        # )  # No loss and optimization based on word embeddings from transformer block

    # def build(self, input_shape):
    #     self.input_shape = input_shape

    def summary(self):
        x = layers.Input(shape=self.input_shape[1:])
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=(raw_shape), ragged=True)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def call(self, inputs, training=True, mask=None):
        token_embeddings = self.embedding_layer(inputs)

        h = token_embeddings
        for block in self.blocks:
            h = block(h)
        h = self.layer_norm(h)
        logits = self.outputs(h)
        return logits
