#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:33:04 2019
@author: Leonard
"""

##
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model

# The original GSV images have 640x640, which we downscale to 224x224
input_shape = (224, 224, 3)

#Instantiate a VGG16 clone
clone = Sequential([
        
        ## CONV PART
Conv2D(64, (3, 3), input_shape=input_shape, padding="same", activation="relu"),
Conv2D(64, (3, 3), activation="relu", padding="same"),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(128, (3, 3), activation="relu", padding="same"),
Conv2D(128, (3, 3), activation="relu", padding="same",),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(256, (3, 3), activation="relu", padding="same",),
Conv2D(256, (3, 3), activation="relu", padding="same",),
Conv2D(256, (3, 3), activation="relu", padding="same",),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation="relu", padding="same",),
Conv2D(512, (3, 3), activation="relu", padding="same",),
Conv2D(512, (3, 3), activation="relu", padding="same",),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation="relu", padding="same",),
Conv2D(512, (3, 3), activation="relu", padding="same",),
Conv2D(512, (3, 3), activation="relu", padding="same",),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        ## CLASSIFICATION PART
Flatten(),
Dense(4096, activation="relu"),
Dense(4096, activation="relu"),
Dense(1000, activation="softmax")
])

#show structure of the network
clone.summary()
#save structure
plot_model(clone, show_shapes=True,  to_file='VGG_full.png')


#Instantiate only the convolutional part and use our own FC classifier
SA_classifier = Sequential([
        ## CONV PART
Conv2D(64, (3, 3), input_shape=input_shape, padding="same", activation="relu"),
Conv2D(64, (3, 3), activation="relu", padding="same"),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(128, (3, 3), activation="relu", padding="same"),
Conv2D(128, (3, 3), activation="relu", padding="same",),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(256, (3, 3), activation="relu", padding="same",),
Conv2D(256, (3, 3), activation="relu", padding="same",),
Conv2D(256, (3, 3), activation="relu", padding="same",),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation="relu", padding="same",),
Conv2D(512, (3, 3), activation="relu", padding="same",),
Conv2D(512, (3, 3), activation="relu", padding="same",),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation="relu", padding="same",),
Conv2D(512, (3, 3), activation="relu", padding="same",),
Conv2D(512, (3, 3), activation="relu", padding="same",),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        ## CLASSIFICATION PART
Flatten(),
Dense(256, activation='relu'),
Dropout(0.5),
Dense(256, activation='relu'),
Dropout(0.5),
Dense(256, activation='relu'),
Dropout(0.5),
Dense(1, activation='sigmoid')
])

#show structure of the network
SA_classifier.summary()
#save structure
plot_model(SA_classifier, show_shapes=True,  to_file='VGG_modified_SA.png')
