# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf

# get image from Nodejs
import sys
# Takes first name and last name via command
# line arguments and then display them
image_path = sys.argv[1]

# predict image

model = load_model('model_trained_10class.hdf5')

food_list = ['apple_pie', 'cup_cakes', 'donuts', 'french_fries',
             'fried_rice', 'ice_cream', 'pizza', 'pho', 'steak', 'spring_rolls']

# image_path="pho.jpg"

img = image.load_img(image_path, target_size=(299, 299))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.

pred = model.predict(img)
index = np.argmax(pred)
food_list.sort()
pred_value = food_list[index]

print(pred_value)
