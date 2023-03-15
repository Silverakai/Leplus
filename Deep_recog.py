import cv2
import matplotlib.pyplot as plt
import numpy as np


from tensorflow import keras
model = keras.models.load_model('alphabet.h5')


lol = cv2.imread(r'data\letter1.png', cv2.IMREAD_GRAYSCALE)

imgr = cv2.resize(lol, (28,28), interpolation = cv2.INTER_LINEAR)
imgr = cv2.bitwise_not(imgr)

#lettre = np.argmax(model.predict(my_number.reshape(1,28,28,1)), axis=-1)
lettre = np.argmax(model.predict(imgr.reshape(1,28,28,1)), axis=-1)


list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

print(list[lettre[0]])

#---------
modele = keras.models.load_model('numero.h5')

lolo = cv2.imread(r'data\letter3.png', cv2.IMREAD_GRAYSCALE)

imgre = cv2.resize(lolo, (28,28), interpolation = cv2.INTER_LINEAR)
imgre = cv2.bitwise_not(imgre)

plt.imshow(imgre, cmap = 'gray')

number = np.argmax(modele.predict(imgre.reshape(1,28,28,1)), axis=-1)


liste= ["0","1","2","3","4","5","6","7","8","9"]

print(liste[number[0]])