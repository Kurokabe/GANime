import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        # This parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings)  # , dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "num_embeddings": self.num_embeddings,
                "beta": self.beta,
            }
        )
        return config

    def call(self, x):
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)

        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        loss = commitment_loss + codebook_loss
        # self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized, encoding_indices, loss

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1, output_type=tf.int32)
        return encoding_indices

    def get_codebook_entry(self, indices, shape):
        encodings = tf.one_hot(indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, shape)
        return quantized

# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import backend as K

# class VectorQuantizer(layers.Layer):
#     def __init__(self, num_embeddings, embedding_dim, beta,
#                  initializer='uniform', epsilon=1e-10, **kwargs):
#         self.embedding_dim = embedding_dim
#         self.num_embeddings = num_embeddings
#         self.beta = beta
#         self.initializer = initializer
#         super().__init__(**kwargs)

#     def build(self, input_shape):
#         # Add embedding weights.
#         self.w = self.add_weight(name='embedding',
#                                   shape=(self.embedding_dim, self.num_embeddings),
#                                   initializer=self.initializer,
#                                   trainable=True)

#         # Finalize building.
#         super().build(input_shape)

#     def call(self, x):
#         # Flatten input except for last dimension.
#         flat_inputs = K.reshape(x, (-1, self.embedding_dim))

#         # Calculate distances of input to embedding vectors.
#         distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
#                      - 2 * K.dot(flat_inputs, self.w)
#                      + K.sum(self.w ** 2, axis=0, keepdims=True))

#         # Retrieve encoding indices.
#         encoding_indices = K.argmax(-distances, axis=1)
#         encodings = K.one_hot(encoding_indices, self.num_embeddings)
#         encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
#         quantized = self.quantize(encoding_indices)

#         commitment_loss = self.beta * tf.reduce_mean(
#             (tf.stop_gradient(quantized) - x) ** 2
#         )
#         codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
#         loss = commitment_loss + codebook_loss
#         # self.add_loss(commitment_loss + codebook_loss)

#         # Straight-through estimator.
#         quantized = x + tf.stop_gradient(quantized - x)

#         # Metrics.
#         #avg_probs = K.mean(encodings, axis=0)
#         #perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))
        
#         return quantized, encoding_indices, loss

#     @property
#     def embeddings(self):
#         return self.w

#     def quantize(self, encoding_indices):
#         w = K.transpose(self.embeddings.read_value())
#         return tf.nn.embedding_lookup(w, encoding_indices)