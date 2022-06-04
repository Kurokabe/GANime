#!/usr/bin/env python
# coding: utf-8


# In[3]:


import omegaconf
import numpy as np
import matplotlib.pyplot as plt
from ganime.data.experimental import ImageDataset, VideoDataset
from ganime.visualization.videos import display_videos
from ganime.visualization.images import display_images
from ganime.model.vqgan_clean.net2net import Net2Net
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from tqdm import tqdm
from pyprojroot.pyprojroot import here
#tf.get_logger().setLevel('ERROR')


# In[4]:


for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


# In[5]:


strategy = tf.distribute.MultiWorkerMirroredStrategy()


# In[6]:


cfg = omegaconf.OmegaConf.load(here("configs/moving_mnist_image_transformer.yaml"))
#cfg = omegaconf.OmegaConf.load(here("configs/default_transformer.yaml"))
batch_size = 16


# In[7]:


dataset_length = 10000
num_batch = dataset_length / batch_size


# In[8]:


def preprocess(element):
    element = tf.reshape(element, (tf.shape(element)[0], tf.shape(element)[1], tf.shape(element)[2], 3))
    element = tf.cast(element, tf.float32) / 255.0
    first_frame = element[0:1,...]
    last_frame = element[-1:,...]
    
    y = element[1:,...]
    
    first_last_frame = tf.concat([first_frame, last_frame], axis=0)
    
    return first_last_frame, y


# In[9]:


dataset = VideoDataset("../../../data/moving_mnist_tfrecords").load()
dataset = dataset.shuffle(dataset_length, reshuffle_each_iteration=True).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


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


train_sample_data[1].shape


# In[14]:


with strategy.scope():
    model = Net2Net(**cfg["model"])


# In[15]:


lrs = [model.scheduled_lrs(i) for i in range(int(num_batch) * 100)]
xs = np.linspace(0, 300, len(lrs))
plt.plot(xs, lrs)


# In[16]:


from ganime.utils.callbacks import TensorboardVideo, get_logdir
import os

logdir = get_logdir("../../../logs/ganime/", experiment_name="new_transformer_mnist_video")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_video_callback = TensorboardVideo(logdir, train_sample_data, validation_sample_data)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=50,
    restore_best_weights=True,
)
checkpointing = tf.keras.callbacks.ModelCheckpoint(os.path.join(logdir, "checkpoint", "checkpoint"), monitor='val_loss', save_best_only=True, save_weights_only=True)
callbacks = [tensorboard_callback, early_stopping, checkpointing, tensorboard_video_callback]


# In[17]:


#with strategy.scope():
#    model.compile(optimizer=tfa.optimizers.AdamW(
#        learning_rate=1e-3, weight_decay=1e-4
#    ))


# display_images(train_ds[0][0], 1, 3)
# plt.show()

# display_images(train_ds[0][1], 1, 3)
# plt.show()

# In[18]:


with strategy.scope():
    model.first_stage_model.build(input_shape=(None, *train_sample_data[0].shape[2:]))
    model.cond_stage_model.build(input_shape=(None, *train_sample_data[0].shape[2:]))


# In[19]:


with strategy.scope():
    video = model(train_sample_data[0])


# In[20]:


model.summary()


# In[ ]:


model.fit(train_ds, validation_data=validation_ds, epochs=500, callbacks=callbacks)




