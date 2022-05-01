# %%
import numpy as np
from ganime.data.base import load_dataset
from ganime.model.p2p.p2p_v2 import P2P
import tensorflow as tf
from tqdm import tqdm

tf.get_logger().setLevel("ERROR")

# %%
strategy = tf.distribute.MirroredStrategy()

# %%
train_ds, test_ds, input_shape = load_dataset(
    "moving_mnist", "../../data", batch_size=100
)  # * strategy.num_replicas_in_sync)

# %%
train_ds_iterator = train_ds.as_numpy_iterator()
x = next(train_ds_iterator)

# %%
with strategy.scope():
    model = P2P()

# %%
model.encoder.summary(line_length=200)

# %%
model.decoder.summary(line_length=200)

# %%
model.frame_predictor.summary(line_length=200)

# %%
with strategy.scope():
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-8
        )
    )
history = model.fit(train_ds, epochs=500)

# %%
from ganime.visualization.videos import display_videos

generated = model(x[0])

# %%
display_videos(generated, n_rows=1, n_cols=5)

# %%
display_videos(x[1], n_rows=1, n_cols=5)


# %%
model.save("p2p_v2")
