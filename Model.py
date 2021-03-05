from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Conv3D, MaxPooling3D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import GlobalMaxPool2D
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Reshape, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.python.keras.utils import np_utils, generic_utils

import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
from sklearn import preprocessing


nb_classes = 2
dropout_rate = 0.2

model = Sequential()

strides = (1,1,1)
kernel_size = (3, 3, 3)

model.add(Conv3D(32, kernel_size, strides=strides, activation='relu', padding='same', input_shape=(30, 96, 144, 3)))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)
#model.add(Dropout(dropout_rate))

model.add(Conv3D(64, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)
#model.add(Dropout(dropout_rate))

model.add(Conv3D(128, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
print(model.output_shape)
#model.add(Dropout(dropout_rate))

model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
#model.add(Dropout(dropout_rate))

model.add(Conv3D(256, kernel_size, strides=strides, activation='relu',padding='same'))
print(model.output_shape)
model.add(BatchNormalization())
#model.add(Dropout(dropout_rate))

model.add(MaxPooling3D(pool_size=(1,12,18)))
print(model.output_shape)

model.add(Reshape((30, 256)))
print(model.output_shape)
model.add(LSTM(256, return_sequences=True))
print(model.output_shape)
model.add(LSTM(256))
print(model.output_shape)

model.add(Dense(256, activation='relu'))
print(model.output_shape)
#model.add(Dropout(dropout_rate))

model.add(Dense(nb_classes, activation='sigmoid'))
print(model.output_shape)