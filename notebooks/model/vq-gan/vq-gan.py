#!/usr/bin/env python
# coding: utf-8

# In[3]:


import omegaconf
import numpy as np
import matplotlib.pyplot as plt
from ganime.data.experimental import ImageDataset, VideoDataset
from ganime.model.vqgan_clean.vqgan import VQGAN
from ganime.visualization.videos import display_videos
from ganime.visualization.images import display_images
from ganime.model.vqgan_clean.transformer.mingpt import GPT
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import os
from pyprojroot.pyprojroot import here

# tf.get_logger().setLevel('ERROR')


# In[4]:


for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


# In[5]:


num_workers = len(tf.config.list_physical_devices("GPU"))
batch_size = 128


# In[6]:


cfg = omegaconf.OmegaConf.load(here("configs/moving_mnist_image.yaml"))


# In[7]:


dataset_length = 20 * 10000
num_batch = dataset_length / batch_size


# In[8]:


def preprocess(element):
    element = tf.reshape(element, (tf.shape(element)[0], tf.shape(element)[1], 3))
    element = tf.cast(element, tf.float32) / 255.0
    return element, element


# In[9]:


dataset = ImageDataset("../../../data/mnist_tfrecords").load()
dataset = (
    dataset.shuffle(dataset_length, reshuffle_each_iteration=True)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)


# In[10]:


train_size = int(num_batch * 0.8)
validation_size = int(num_batch * 0.1)
test_size = int(num_batch * 0.1)


# In[11]:


train_ds = dataset.take(train_size)
validation_ds = dataset.skip(train_size).take(validation_size)
test_ds = dataset.skip(train_size + validation_size).take(validation_size)


# In[12]:


train_sample_data = next(train_ds.as_numpy_iterator())
validation_sample_data = next(validation_ds.as_numpy_iterator())


# In[13]:


from ganime.utils.callbacks import TensorboardImage, get_logdir

logdir = get_logdir("../../../logs/ganime/", experiment_name="new_moving_mnist_image")
# Define the basic TensorBoard callback.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_image_callback = TensorboardImage(
    logdir, train_sample_data, validation_sample_data
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_total_loss",
    min_delta=0.001,
    patience=50,
    restore_best_weights=True,
)
checkpointing = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(logdir, "checkpoint", "checkpoint"),
    monitor="val_total_loss",
    save_best_only=True,
    save_weights_only=True,
)
callbacks = [
    tensorboard_callback,
    tensorboard_image_callback,
    early_stopping,
    checkpointing,
]


# In[14]:


strategy = tf.distribute.MirroredStrategy()


# In[18]:


with strategy.scope():
    vqgan = VQGAN(**cfg["model"])


# In[19]:


with strategy.scope():
    vqgan.compile(
        gen_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        disc_optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    )


# In[22]:


history = vqgan.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=500,
    callbacks=callbacks,
)


# In[ ]:
