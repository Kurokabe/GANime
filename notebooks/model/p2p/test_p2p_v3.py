

# %%
import numpy as np
from ganime.data.base import load_dataset
from ganime.model.p2p.p2p_v3 import P2P
from ganime.visualization.videos import display_videos
import tensorflow as tf
from tqdm import tqdm
tf.get_logger().setLevel('ERROR')

# %%
train_ds, test_ds, input_shape = load_dataset("moving_mnist", "../../data", batch_size=64)

# %%
strategy = tf.distribute.MirroredStrategy()

# %%
with strategy.scope():
    model = P2P(g_dim=128, z_dim=64)

# %%
model.encoder.summary(line_length=200)

# %%
model.decoder.summary(line_length=200)

# %%
with strategy.scope():
    model.compile(optimizer=tf.keras.optimizers.Adam())
history = model.fit(train_ds, epochs=1500)

# %%
train_ds_iterator = train_ds.as_numpy_iterator()
x = next(train_ds_iterator)

# %%
generated = model(x[0])

# %%
display_videos(generated, n_rows=1, n_cols=5)

# %%
display_videos(x[1], n_rows=1, n_cols=5)

# %%
test_ds_iterator = test_ds.as_numpy_iterator()
x = next(test_ds_iterator)

# %%
generated = model(x[0])
display_videos(generated, n_rows=1, n_cols=5)

# %%
display_videos(x[1], n_rows=1, n_cols=5)

# %%
model.save("B-VAE_LSTM_longer")

# %%



