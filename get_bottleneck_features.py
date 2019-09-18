
#### Script to reproduce the Street Art classifier

## Used libraries


import numpy as np
from math import ceil
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras import applications, optimizers


#### Pre-Settings


train_data_dir = "data\\training"
validation_data_dir = "data\\validation"
img_width, img_height = 224, 224
nb_train_samples, nb_validation_samples = 520, 100
epochs, batch_size = 100, 8

#### Get the bottleneck features


datagen = ImageDataGenerator(rescale=1., featurewise_center=True) 
#original VGG16 image preprocessing mean
datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32)

#instantiate the VGG16 network, dont use the top (fully connected) layers
model = applications.VGG16(include_top=False, weights='imagenet')
#generate training data from the gsv images
generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,#generator should only yield data batches without labels
    shuffle=False)
predict_size_train = int(ceil(nb_train_samples / batch_size))
# predict_generator returns the output of a model, given
# a generator that yields batches of data
bottleneck_features_train = model.predict_generator(generator, predict_size_train)
with open('bottleneck_features_train.npy', 'wb') as features_train_file:
    np.save(features_train_file, bottleneck_features_train)

#generate validation data from the gsv images
generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,#same here
    shuffle=False)
predict_size_validation = int(ceil(nb_validation_samples / batch_size))
bottleneck_features_validation = model.predict_generator(generator, predict_size_validation)
with open('bottleneck_features_validation.npy', "wb") as features_validation_file:
    np.save(features_validation_file, bottleneck_features_validation)


## Train the small top model 
    
    
with open('bottleneck_features_train.npy', 'rb') as features_train_file:
    train_data = np.load(features_train_file)
    # we now have to label the data again 80:0, 80:1
train_labels = np.array(
    [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

with open('bottleneck_features_validation.npy', 'rb') as features_validation_file:
    validation_data = np.load(features_validation_file)
    
validation_labels = np.array(
    [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

# Instantiate the top model, which can be seen as the classification model 
model = Sequential()#sequential because we add layer by layer
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0),
              loss='binary_crossentropy', metrics=['accuracy'])
# train the model
model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))
# save its weights
model.save_weights("bottleneck_weights")
# save the whole model
model.save("model_bn_weighted.h5")
import pandas as pd
h = pd.DataFrame(model.history.history)
h["Epoch"] = list(range(1, 201))
h["train_acc"] = h["acc"]
h["train_loss"] = h["loss"]

plot = h.plot(y=["train_acc", "val_acc"], x="Epoch")
fig = plot.get_figure()
fig.savefig("network_accuracy.png")

plot = h.plot(y=["train_loss", "val_loss"], x="Epoch")
fig = plot.get_figure()
fig.savefig("network_loss.png")

    
