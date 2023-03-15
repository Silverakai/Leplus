# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:08:12 2022

@author: riyad
"""
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np


from tensorflow import keras
model = keras.models.load_model('alphabet.h5')

im = Image.open(r'data\ger3.jpg')
cube = Image.open(r'data\cube.png')
width, height = im.size

left = 70
top = 10
right = 100
bottom = height

im1 = im.crop((left, top, right, bottom))
#im1.show()

cube.paste(im1,(40,40))
cube.show()

cube.save("combine.png")

lol = cv2.imread('combine.png', cv2.IMREAD_GRAYSCALE)

imgr = cv2.resize(lol, (28,28), interpolation = cv2.INTER_LINEAR)
imgr = cv2.bitwise_not(imgr)

#lettre = np.argmax(model.predict(my_number.reshape(1,28,28,1)), axis=-1)
lettre = np.argmax(model.predict(imgr.reshape(1,28,28,1)), axis=-1)


list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

print(list[lettre[0]])