#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

# In[3]:



from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


# In[4]:


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
#tf.get_logger().setLevel('ERROR')


# In[5]:


for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)


# In[6]:


num_workers = len(tf.config.list_physical_devices("GPU"))
batch_size = 128


# In[7]:


cfg = omegaconf.OmegaConf.load(here("configs/moving_mnist_image.yaml"))


# In[8]:


dataset_length = 20*10000
num_batch = dataset_length / batch_size


# In[9]:


def preprocess(element):
    element = tf.reshape(element, (tf.shape(element)[0], tf.shape(element)[1], 3))
    element = tf.cast(element, tf.float32) / 255.0
    return element, element


# In[10]:


dataset = ImageDataset("../../../data/mnist_tfrecords").load()
dataset = dataset.shuffle(dataset_length, reshuffle_each_iteration=True).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# In[11]:


train_size = int(num_batch * 0.8)
validation_size = int(num_batch * 0.1)
test_size = int(num_batch * 0.1)


# In[12]:


train_ds = dataset.take(train_size)
validation_ds = dataset.skip(train_size).take(validation_size)
test_ds = dataset.skip(train_size + validation_size).take(validation_size)


# In[13]:


train_sample_data = next(train_ds.as_numpy_iterator())
validation_sample_data = next(validation_ds.as_numpy_iterator())


# In[14]:


from ganime.utils.callbacks import TensorboardImage, get_logdir

logdir = get_logdir("../../../logs/ganime/", experiment_name="moving_mnist_image_f16")
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
callbacks = [tensorboard_callback, tensorboard_image_callback, early_stopping, checkpointing]


# In[15]:


strategy = tf.distribute.MirroredStrategy()


# In[16]:


# train_mean, train_var, train_std = dataset_statistics(train_ds)


# In[17]:


from ganime.visualization.images import display_true_pred


# In[18]:


display_images(train_sample_data[0])
plt.show()


# In[19]:


with strategy.scope():
    vqgan = VQGAN(**cfg["model"])


# In[20]:


with strategy.scope():
    gen_optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=1e-4))#, clipvalue=1.0, clipnorm=0.5, epsilon=1e-4))
    disc_optimizer = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=5e-5))#, clipvalue=1.0, clipnorm=0.5, epsilon=1e-4))
    vqgan.compile(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer)


# In[ ]:


history = vqgan.fit(train_ds, validation_data=validation_ds, epochs=1500, callbacks=callbacks)


# In[ ]:


x = train_sample_data[0]
generated = vqgan(x[:10])[0]


# In[31]:


display_images(generated)
plt.show()


# In[ ]:


display_images(x)
plt.show()


# In[ ]:


x2 = train_ds[30][0]
generated2 = vqgan(x2[:10])


# In[ ]:


display_images(generated2)
plt.show()


# In[ ]:


display_images(x2)
plt.show()


# In[ ]:




