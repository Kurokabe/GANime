#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5, 6"
os.environ["NCCL_DEBUG"]="WARN"
#os.environ["NCCL_P2P_LEVEL"]="NODE"


# In[2]:


import sys
sys.path.append("../../../")


# In[3]:


# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

# In[4]:


import omegaconf
import numpy as np
import matplotlib.pyplot as plt
from ganime.data.experimental import ImageDataset, VideoDataset
from ganime.model.vqgan_clean.vqgan import VQGAN
from ganime.visualization.videos import display_videos
from ganime.visualization.images import display_images
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import os
from pyprojroot.pyprojroot import here
#tf.get_logger().setLevel('ERROR')


# In[5]:


for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


# In[6]:


strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


# In[7]:


cfg = omegaconf.OmegaConf.load(here("configs/kny_image_vgg19.yaml"))


# In[8]:


num_workers = len(tf.config.list_physical_devices("GPU"))
batch_size = cfg["trainer"]["batch_size"] 
global_batch_size = batch_size * strategy.num_replicas_in_sync
n_epochs = cfg["trainer"]["n_epochs"] 
sample_batch_size = 8


# In[9]:


dataset_length = 34045 # KNY
#dataset_length = 20*10000 # MNIST
num_batch = dataset_length / batch_size


# In[10]:


def preprocess(element):
    element = tf.reshape(element, (tf.shape(element)[0], tf.shape(element)[1], 3))
    element = tf.cast(element, tf.float32) / 255.0
    return element, element


# In[11]:


dataset = ImageDataset("../../../data/kny/images_tfrecords_light").load()
dataset = dataset.shuffle(dataset_length, reshuffle_each_iteration=True, seed=10).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)


# In[12]:


train_size = int(dataset_length * 0.2)
validation_size = int(dataset_length * 0.1)
test_size = int(dataset_length * 0.1)


# In[13]:


train_ds = dataset.take(train_size)
validation_ds = dataset.skip(train_size).take(validation_size)#.padded_batch(global_batch_size).map(postprocess)
test_ds = dataset.skip(train_size + validation_size).take(validation_size)#.padded_batch(global_batch_size).map(postprocess)


# In[14]:


train_sample_data = next(train_ds
                          .batch(sample_batch_size)
                          .prefetch(tf.data.AUTOTUNE).as_numpy_iterator())
validation_sample_data = next(validation_ds.batch(sample_batch_size).as_numpy_iterator())


# In[15]:


train_ds = (train_ds.batch(global_batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))
validation_ds = (validation_ds.batch(global_batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))
test_ds = (test_ds.batch(global_batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))


# In[16]:


from ganime.utils.callbacks import TensorboardImage, get_logdir

logdir = get_logdir("../../../logs/ganime/vqgan", experiment_name="vgg19_loss_final")
# Define the basic TensorBoard callback.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
tensorboard_image_callback = TensorboardImage(logdir, train_sample_data, validation_sample_data)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_total_loss",
    min_delta=0.0001,
    patience=100,
    restore_best_weights=True,
)
checkpointing = tf.keras.callbacks.ModelCheckpoint(os.path.join(logdir, "checkpoint", "checkpoint"), monitor='val_total_loss', save_best_only=True, save_weights_only=True)
callbacks = [tensorboard_callback, tensorboard_image_callback, checkpointing]


# In[17]:


# train_mean, train_var, train_std = dataset_statistics(train_ds)


# In[18]:


from ganime.visualization.images import display_true_pred


# In[19]:


display_images(train_sample_data[0])
plt.show()


# In[20]:


with strategy.scope():
    vqgan = VQGAN(**cfg["model"])


# In[21]:


with strategy.scope():
    gen_optimizer = tf.keras.optimizers.Adam(
                learning_rate=cfg["trainer"]["gen_lr"],
                beta_1=cfg["trainer"]["gen_beta_1"],
                beta_2=cfg["trainer"]["gen_beta_2"],
                clipnorm=cfg["trainer"]["gen_clip_norm"],
    )
    disc_optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg["trainer"]["disc_lr"],
        beta_1=cfg["trainer"]["disc_beta_1"],
        beta_2=cfg["trainer"]["disc_beta_2"],
        clipnorm=cfg["trainer"]["disc_clip_norm"],
    )
    vqgan.compile(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)


# In[ ]:


history = vqgan.fit(train_ds, validation_data=validation_ds, epochs=n_epochs, callbacks=callbacks)


# In[ ]:


with strategy.scope():
    x = train_sample_data[0]
    generated = vqgan(x[:10])[0]


# In[ ]:


vqgan.summary()


# In[ ]:


from ganime.model.vqgan_clean.losses.losses import Losses
from ganime.model.vqgan_clean.losses.vqperceptual import PerceptualLoss


# In[ ]:


losses = Losses(1)


# In[ ]:


real = x
pred = generated


# In[ ]:


losses.perceptual_loss(real.copy(), pred)


# In[ ]:


losses.vgg_loss(real, pred)


# In[ ]:


losses.style_loss(real, pred)


# In[ ]:


PerceptualLoss()(real, pred)


# In[ ]:


display_images(pred)
plt.show()


# In[ ]:


display_images(real)
plt.show()


# In[ ]:




