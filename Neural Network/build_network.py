from keras.models import Sequential, load_model, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, optimizers



top_model_path = "model_bn_weighted.h5"
top_model_weights_path = "bottleneck_weights.h5"

img_width, img_height = 224, 224


# Instantiate the VGG16 network
base_model = applications.VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(img_width, img_height, 3))

# Instantiate the top model
top_model = load_model(top_model_path)



# add the model on top of the convolutional base
full_model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# save the full model
full_model.save("SA_classifier.h5")
