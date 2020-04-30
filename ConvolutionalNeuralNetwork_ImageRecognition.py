#!/usr/bin/env python
# coding: utf-8

# ## Convolutional Neural Network

# In[14]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
import matplotlib.pyplot as plt


# In[15]:


from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test,y_test) = cifar10.load_data()


# In[19]:


y_test


# In[20]:


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[21]:


#Normalization
X_test = X_test/255.0
X_train = X_train/255.0


# In[56]:


plt.imshow(X_test[12])


# In[45]:


#buliding a convolutional network
model = Sequential()
#First Layer
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', 
                                 input_shape = [32,32,3], padding = 'same' ))
#Second Layer
model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding= 'same'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding= 'valid'))


# In[46]:


#Third Layer
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))


# In[47]:


#Fourth Layer
model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding= 'same'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding= 'valid'))


# In[48]:


model.add(tf.keras.layers.Flatten())


# In[50]:


model.add(Dense(units = 64, activation = 'relu' ))
model.add(Dense(units = 10, activation = 'softmax'))


# In[51]:


model.summary()


# In[52]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adam", metrics=["sparse_categorical_accuracy"])


# In[53]:


model.fit(X_train, y_train, epochs=5, njob)


# In[54]:


model.evaluate(X_test, y_test)


# In[ ]:




