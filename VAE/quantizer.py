# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:54:13 2020

@author: Dell
"""

import datetime
from PIL import Image
from tensorflow.keras import models, datasets, layers, Model
import numpy as np
from matplotlib import image, pyplot
import tensorflow as tf

# %% load dataset

# (train_images,_) , (test_images,_) = datasets.mnist.load_data()

# %% qunatize dataset

train_images = list()
train_labels = list()

# Amount of quantization for choosing a specific dataset
# bit_quant = 4

for i in range(1,60001):
    train_labels.append(image.imread(f'<Path to Original Dataset>'%i))
    train_images.append(image.imread(f'<Path to Quantized Dataset>'%(bit_quant,i)))
    
train_images = np.array(train_images)
train_labels = np.array(train_labels)
train_images, train_labels = train_images / 255.0, train_labels / 255.0

train_images = train_images.reshape((60000,28,28,1))
train_labels = train_labels.reshape((60000,28,28,1))

# %% histogram?

hist = dict()

for image in train_images:
    for i in range(28):
        for j in range(28):
            try:
                hist[np.uint8(np.squeeze(image[i,j]) * 255.0)] += 1
            except KeyError:
                hist[np.uint8(np.squeeze(image[i,j]) * 255.0)] = 1

for key in hist.keys():
    hist[key] = hist[key] / (60000 * 28 * 28)
    
# %% Model #2

class NoiseReducer(Model): 
  def __init__(self):

    super(NoiseReducer, self).__init__() 

    self.encoder = models.Sequential()
    self.encoder.add(layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2)) 
    self.encoder.add(layers.Conv2D(8, (3,3), activation='relu', padding='same', strides=2))
    
    self.decoder = models.Sequential()
    self.decoder.add(layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same', input_shape=(7,7,8)))
    self.decoder.add(layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'))
    self.decoder.add(layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')) 
  
  def call(self, x): 
    encoded = self.encoder(x) 
    decoded = self.decoder(encoded) 
    return decoded

# %% get obj #2

autoencoder = NoiseReducer()
autoencoder.compile(optimizer="adam", loss="mse")
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
autoencoder.fit(train_images, 
                train_labels, 
                epochs=10, 
                shuffle=True, 
                validation_split=0.1,
                callbacks=[tensorboard_callback])

# %% test-out an output from the autoencoder

filter_img = tf.squeeze(autoencoder.decoder(autoencoder.encoder(train_images[0]).numpy())).numpy() * 255.0