import math
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Model


class MultiHeadSelfAttention(layers.Layer):
    # TODO: Update docstrings
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    

class ClusterableWeightsCA(tfmot.clustering.keras.ClusteringAlgorithm):
    """This class provides a special lookup function for the the weights 'w'.
    It reshapes and tile centroids the same way as the weights. This allows us
    to find pulling indices efficiently.
    """

    def get_pulling_indices(self, weight):
        clst_num = self.cluster_centroids.shape[0]
        tiled_weights = tf.tile(tf.expand_dims(weight, axis=2), [1, 1, clst_num])
        tiled_cluster_centroids = tf.tile(
            tf.reshape(self.cluster_centroids, [1, 1, clst_num]),
            [weight.shape[0], weight.shape[1], 1],
        )

        # We find the nearest cluster centroids and store them so that ops can build
        # their kernels upon it
        pulling_indices = tf.argmin(
            tf.abs(tiled_weights - tiled_cluster_centroids), axis=2
        )

        return pulling_indices


class PrunableClusterableLayer(
    tf.keras.layers.Layer,
    tfmot.sparsity.keras.PrunableLayer,
    tfmot.clustering.keras.ClusterableLayer,
):
    def get_prunable_weights(self):
        # Prune bias also, though that usually harms model accuracy too much.
        return [("kernel", self.kernel)]

    def get_clusterable_weights(self):
        # Cluster kernel and bias. This is just an example, clustering
        # bias usually hurts model accuracy.
        return [("kernel", self.kernel), ("bias", self.bias)]

    def get_clusterable_algorithm(self, weight_name):
        """Returns clustering algorithm for the custom weights 'w'."""
        if weight_name == "kernel":
            return ClusterableWeightsCA
        else:
            # We don't cluster other weights.
            return None


class ConvEmbedding(PrunableClusterableLayer):
    def __init__(self, num_filters, **kwargs):
        super(ConvEmbedding, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.conv1d = layers.Conv1D(
            filters=num_filters, kernel_size=1, activation="relu"
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
            }
        )
        return config

    def call(self, inputs):
        embedding = self.conv1d(inputs)

        return embedding


class PositionalEncoding(PrunableClusterableLayer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super(PositionalEncoding, self).__init__(dtype=dtype, **kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims

        if max_dims % 2 == 1:
            max_dims += 1  # max_dims must be even
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_steps": self.max_steps,
                "max_dims": self.max_dims,
            }
        )
        return config

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, : shape[-2], : shape[-1]]


class RelativePositionEmbedding(tf.keras.layers.Layer):
    """Creates a positional embedding.

    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and formulized in
    "Attention is All You Need", section 3.5.
    (https://arxiv.org/abs/1706.03762).

    Arguments:
    hidden_size: Size of the hidden layer.
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position.
    """

    def __init__(self, hidden_size, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        # We compute the positional encoding in float32 even if the model uses
        # float16, as many of the ops used, like log and exp, are numerically
        # unstable in float16.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def call(self, inputs, length=None):
        """Implements call() for the layer.

        Args:
          inputs: An tensor whose second dimension will be used as `length`. If
            `None`, the other `length` argument must be specified.
          length: An optional integer specifying the number of positions. If both
            `inputs` and `length` are spcified, `length` must be equal to the second
            dimension of `inputs`.

        Returns:
          A tensor in shape of [length, hidden_size].
        """
        shape = tf.shape(inputs)
        length = shape[1]
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = self._hidden_size // 2
        min_timescale, max_timescale = self._min_timescale, self._max_timescale
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (tf.cast(num_timescales, tf.float32) - 1)
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment
        )
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        position_embeddings = tf.concat(
            [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1
        )
        return inputs + position_embeddings


class TransformerBlock(PrunableClusterableLayer):
    # TODO: Update docstrings
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
            }
        )
        return config

    def call(self, inputs, training):

        # Sublayer 1
        attn_output = self.att(inputs)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(
            inputs + attn_output
        )  # Residual connection, (batch_size, input_seq_len, d_model)

        # Sublayer 2
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # Residual connection, # (batch_size, input_seq_len, d_model)

        return out2  # (batch_size, input_seq_len, d_model)
    

class T2Model(keras.Model):
    # TODO: Update docstrings
    """Time-Transformer with Multi-headed.
    embed_dim --> Embedding size for each token
    num_heads --> Number of attention heads
    ff_dim    --> Hidden layer size in feed forward network inside transformer
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        ff_dim,
        num_filters,
        num_classes,
        num_layers,
        droprate,
        num_aux_feats=0,
        add_aux_feats_to="M",
        **kwargs,
    ):
        super(T2Model, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.droprate = droprate
        self.num_aux_feats = num_aux_feats
        self.add_aux_feats_to = add_aux_feats_to

        self.num_classes = num_classes
        if self.add_aux_feats_to == "L":
            self.sequence_length = input_dim[1] + self.num_aux_feats
        else:
            self.sequence_length = input_dim[
                1
            ]  # input_dim.shape = (batch_size, input_seq_len, d_model)

        self.embedding = ConvEmbedding(
            num_filters=self.num_filters, input_shape=input_dim
        )

        # <-- Additional layers when adding Z features here -->

        self.pos_encoding = PositionalEncoding(
            max_steps=self.sequence_length, max_dims=self.embed_dim
        )

        self.encoder = [
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
            for _ in range(num_layers)
        ]

        self.pooling = layers.GlobalAveragePooling1D()
        self.dropout1 = layers.Dropout(self.droprate)

        # self.fc             = layers.Dense(self.embed_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
        # self.dropout2       = layers.Dropout(self.droprate)

        self.classifier = layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs, training=None):

        # If not a list then inputs are of type tensor: tf.is_tensor(inputs) == True
        if tf.is_tensor(inputs):
            x = self.embedding(inputs)
            x = self.pos_encoding(x)

            for layer in self.encoder:
                x = layer(x, training)

            x = self.pooling(x)
            if training:
                x = self.dropout1(x, training=training)

            # x = self.fc(x)
            # if training:
            #     x = self.dropout2(x, training=training)

            classifier = self.classifier(x)

        # if (isinstance(inputs, list)) and (self.add_aux_feats_to == "M"):
        # Else this implies input is a list; a list of tensors, i.e. multiple inputs
        else:
            if isinstance(inputs, dict):
                x = inputs["input_1"]
                z = inputs["input_2"]
            else:
                # X in L x M
                x = inputs[0]
                # Additional Z features
                z = inputs[1]
                # >>> z.shape
                # TensorShape([None, 2])
            if self.add_aux_feats_to == "M":
                z = tf.tile(z, [1, 100])
                # >>> z.shape
                # TensorShape([None, 200])
                z = tf.keras.layers.Reshape([100, 2])(z)
                # >>> z.shape
                # TensorShape([None, 100, 2])
                x = tf.keras.layers.Concatenate(axis=2)([x, z])
                # >>> x.shape
                # TensorShape([None, 100, 8)])
            else:  # Else self.add_aux_feats_to == 'L'
                z = tf.tile(z, [1, 6])
                # >>> z.shape
                # TensorShape([None, 12])
                z = tf.keras.layers.Reshape([2, 6])(z)
                # >>> z.shape
                # TensorShape([None, 2, 6])
                x = tf.keras.layers.Concatenate(axis=1)([x, z])
                # >>> x.shape
                # TensorShape([None, 102, 6)])

            # Transforms X in L x (M + Z) -> X in L x d if self.add_aux_feats_to == "M" or
            # transforms X in (L + 2) x M -> X in L x d if self.add_aux_feats_to == "L"
            x = self.embedding(x)

            x = self.pos_encoding(x)  # X <- X + P, where X in L x d

            for layer in self.encoder:
                x = layer(x, training)

            x = self.pooling(x)
            if training:
                x = self.dropout1(x, training=training)

            # Additional layers when adding Z features
            # x = tf.keras.layers.Concatenate(axis=1)([inputs[1], x])

            # x = self.fc(x)
            # if training:
            #     x = self.dropout2(x, training=training)

            classifier = self.classifier(x)

        return classifier
    
    def model(self):
        x = tf.keras.layers.Input(shape=(1500,1))
        return Model(inputs=[x], outputs=self.call(x))

    def build_graph(self, input_shapes):
        if isinstance(
            input_shapes, tuple
        ):  # A list would imply there is multiple inputs
            # Code lifted from example:
            # https://github.com/tensorflow/tensorflow/issues/29132#issuecomment-504679288
            input_shape_nobatch = input_shapes[1:]
            # self.build(input_shapes)
            inputs = keras.Input(shape=input_shape_nobatch)
        else:
            input_shape_nobatch = input_shapes[0][1:]
            Z_input_shape_nobatch = input_shapes[1][1:]
            inputs = [
                tf.keras.Input(shape=input_shape_nobatch),
                tf.keras.Input(shape=Z_input_shape_nobatch),
            ]

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)


import random
random.seed(0)

import pickle
with open("./preproccessed_data.pickle", 'rb') as f:
    data = pickle.load(f)
    
train = data['train']
train_target = data['train_target']
validation = data['validation']
validation_target = data['validation_target']
test = data['test']
test_target = data['test_target']

# search with a subset of the data
# from sklearn.model_selection import train_test_split
# train, _, train_target, _ = train_test_split(train, train_target, test_size=0.8, stratify=train_target, random_state=0)
# validation, _, validation_target, _ = train_test_split(validation, validation_target, test_size=0.8, stratify=validation_target, random_state=0)


model = T2Model(input_dim=(1500,1),
embed_dim=128,
num_heads=8,
ff_dim=32,
num_filters=128,
num_classes=4,
num_layers=10,
droprate=0.2
)


from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# print("model parameters: " , model.count_params()/1000)
# print(model.summary()) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
model_name = "best_transformer.pickle"
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100)
model.fit(train, train_target, epochs=200, batch_size=128, callbacks=[es, chk], verbose=2, validation_data=(validation,validation_target))


