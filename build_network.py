## Script to build the full model

from keras.models import Sequential, load_model, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, optimizers


config_dir = "/Users/Leonard/Desktop/NN_imp/SA_classifier/config/"
train_data_dir = "/Users/Leonard/Desktop/NN_imp/SA_classifier/data/training"
validation_data_dir = "/Users/Leonard/Desktop/NN_imp/SA_classifier/data/validation"
top_model_path = config_dir+"model_bn_weighted.h5"
top_model_weights_path = config_dir+"bottleneck_weights.h5"

img_width, img_height = 150, 150
nb_train_samples, nb_validation_samples = 400, 40
epochs, batch_size = 50, 16


# Instantiate the VGG16 network
base_model = applications.VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(img_width, img_height, 3))

# Instantiate the top model
top_model = load_model(top_model_path)



# add the model on top of the convolutional base
full_model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# save the full model
full_model.save(config_dir+"SA_classifier.h5")
