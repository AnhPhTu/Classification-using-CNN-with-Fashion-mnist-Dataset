import tensorflow as tf
from tensorflow import keras
import numpy as np
#load model
model = keras.models.load_model('my_model.h5')
#class
Class_name=["T-shirt/Top", "Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

import cv2
img = cv2.imread('Coat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray=cv2.resize(gray, (28,28))
a = np.array(gray)
a=a/255.0
a=1-a
a = a.reshape(1,28,28,1)
pred  = np.argmax(model.predict(a))

Answer=Class_name[pred]
print(Answer)