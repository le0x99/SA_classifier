from Engine import Environment
from keras.models import load_model
from keras import optimzers

model = load_model("SA_classifier.h5")
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Ad,
              metrics=['accuracy'])
