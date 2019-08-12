
from keras.models import load_model
import requests
from keras.preprocessing.image import img_to_array as to_array, load_img 
import numpy as np, os

def load_classifier():
    model = load_model('/Users/Leonard/Desktop/NN_imp/SA_classifier/config/SA_classifier.h5')
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model


def predict_img(model, img_path): 
    img_width, img_height = 150, 150
    img = load_img(img_path, target_size=(img_width, img_height))
    x = to_array(img)
    x = np.expand_dims(x, axis=0)
    return model.predict(x)
    


def predict_batch(model, img_dir):
    img_width, img_height = 150, 150
    images = os.listdir(img_dir)
    images = [ to_array(load_img(img_dir+img, target_size=(img_width, img_height))) for img in images ]
    images = np.vstack([ np.expand_dims(img, axis=0) for img in images ])
    return model.predict(images)

def img_to_disk(images, destination):
    if type(images) != list:
        images = [images]
    for i in range(len(images)):
        response = requests.get(images[i])
        with open(destination+str(i)+".jpg", 'wb') as f:
            f.write(response.content)

    