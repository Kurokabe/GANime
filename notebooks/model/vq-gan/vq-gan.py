#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from ganime.data.base import load_dataset
from ganime.utils.statistics import dataset_statistics
from ganime.model.vqgan.vqgan import VQGAN
from ganime.visualization.videos import display_images, display_videos
import tensorflow as tf
from tqdm import tqdm

tf.get_logger().setLevel("ERROR")

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices("GPU")[0], True
)


# In[4]:


train_ds, test_ds, input_shape = load_dataset("mnist", "../../../data", batch_size=16)


# In[5]:


train_mean, train_var, train_std = dataset_statistics(train_ds)


# In[8]:


vqgan = VQGAN(num_embeddings=256, embedding_dim=128, train_variance=train_var)


# In[11]:


vqgan.compile(optimizer=tf.keras.optimizers.Adam())
history = vqgan.fit(train_ds, epochs=2)


# In[ ]:
