import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
 
# Cargamos la imagen
path = "content/img4.jpg"
img_height = 400
img_width = 600
original = cv2.imread(path)
cv2.imshow("original", original)
#cv2_imshow(original)
 
# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
 
# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5,5), 0)
 
cv2.imshow("suavizado", gauss)
#cv2_imshow(gauss)
 
# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150)
 
cv2.imshow("canny", canny)
#cv2_imshow(canny)
 
# Buscamos los contornos
(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts =  sorted(contornos, key=cv2.contourArea, reverse=True)[:1]

# Mostramos el número de monedas por consola
print("He encontrado {} objetos".format(len(contornos)))
 
cv2.drawContours(original,cnts,-1,(0,0,255), 2)
cv2.imshow("contornos", original)
#cv2_imshow(original)
 
cv2.waitKey(0)