
from keras.models import load_model
import requests
from PIL import Image
from io import BytesIO
from keras.preprocessing.image import img_to_array as to_array, load_img
import pandas as pd
import numpy as np, os



def load_classifier():
    model = load_model('/Users/Leonard/Desktop/NN_imp/SA_classifier/config/SA_classifier.h5')
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return model


def predict_img(model, img=None, img_path=None): 
    img_width, img_height = 150, 150
    img = load_img(img_path, target_size=(img_width, img_height)) if img_path!=None else img
    x = to_array(img)
    x = np.expand_dims(x, axis=0)
    return model.predict(x)
    


def predict_batch(model, img_dir, batch_size):
    img_width, img_height = 150, 150
    images = os.listdir(img_dir)
    images = [ to_array(load_img(img_dir+img, target_size=(img_width, img_height))) for img in images ]
    images = np.vstack([ np.expand_dims(img, axis=0) for img in images ])
    return model.predict(images, batch_size=batch_size)

def img_to_disk(images, destination, randomize=False):
    if type(images) != list:
        images = [images]
    if not randomize:
        for url in images:
            response = requests.get(url)
            with open(destination+url.lstrip("https://").replace("/", "_")+".jpg", 'wb') as f:
                f.write(response.content)
    else:
        import random
        for i in range(len(images)):
            r = random.randint(55, 555)+random.randint(666, 777) 
            response = requests.get(images[i])
            with open(destination+str(i+r)+".jpg", 'wb') as f:
                f.write(response.content)
                
                
def to_img(url, size="original"):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img.resize(size) if size != "original" else img

                
def classify(data, to_csv=True):
    model = load_classifier()
    model = load_classifier()
    urls = list(data["photo"])
    predictions = []
    for url in urls:
        img = to_img(url=url, size=(150, 150))
        pred = predict_img(model=model, img=img)
        predictions.append(pred[0][0])
    data["pred"] = predictions
    data.to_csv("result_data.csv") if to_csv else None
    return data
            
        
        
        
    
    

    
