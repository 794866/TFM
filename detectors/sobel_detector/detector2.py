import math

import PIL
import cv2
import numpy as np

# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL.Image import Image
from PIL.ImageShow import show

path_success = "/canny_detector/content/9.png"
path2 = "/home/nbellorin/Imágenes/test/10.png"

img_height = 400
img_width = 600
image = cv2.imread(path_success)  # '''PATH'''
dim = (600, 400)
# resize image
image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
dst=cv2.Laplacian(gray, cv2.CV_8UC1)

# %% HoughLines

# Matriz de ceros con mismas dimenciones que dst
out=np.zeros_like(dst)

# Input:
#   imagen,
#   pixeles de resolución,
#   np.pi/180 -> 1 grado de resolución angular
#   mínimo 50 puntos por línea
#  **Nota importante** los puntos de interés son los blancos,
#   ojo con usar puntos negros en fondo blanco, no funcionará
lines = cv2.HoughLines(dst, 1, np.pi / 180, 50, None, 0, 0)
# Salida:
#   radio rho, y angulo theta
#  ver: https://www.geogebra.org/graphing/vmwe4kcb

# Dibujar rectas [fuente:https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html]
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(out, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)

# %% Considerar solo los puntos de las líneas
result=cv2.bitwise_and(dst,dst,mask = out)
#show(result)


# %% Poner más gruesa y conectar lineas débiles

# opción 1
e_im = cv2.dilate(result, None, iterations=5)
d_im = cv2.erode(e_im, None, iterations=3)
# opción 2
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# e_im = cv2.dilate(result, kernel, iterations=3)
# d_im = cv2.erode(e_im, kernel, iterations=2)

#show(d_im)

cv2.imshow('original', d_im)
cv2.waitKey(0)
cv2.destroyAllWindows()


def ordenar_puntos(puntos):
  n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
  y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
  x1_order = y_order[:2]
  x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
  x2_order = y_order[2:4]
  x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

  return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

canny = cv2.Canny(d_im,10,150)
canny = cv2.dilate(canny, None, iterations=1)
cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#print(sorted(cnts, key=cv2.contourArea, reverse=True)[:1])
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

for c in cnts:
  epsilon = 0.01 * cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, epsilon, True)

  # if len(approx)==4:
  cv2.drawContours(image, [approx], 0, (0, 255, 255), 2)

  puntos = ordenar_puntos(approx)
  cv2.circle(image, tuple(puntos[0]), 7, (255, 0, 0), 2)
  cv2.circle(image, tuple(puntos[1]), 7, (0, 255, 0), 2)
  cv2.circle(image, tuple(puntos[2]), 7, (0, 0, 255), 2)
  cv2.circle(image, tuple(puntos[3]), 7, (255, 255, 0), 2)
  print("PUNTOS", puntos)

  pts1 = np.float32(puntos)
  pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
  M = cv2.getPerspectiveTransform(pts1, pts2)
  prsp = cv2.warpPerspective(image, M, (img_width, img_height))
  cv2.imshow('perspectiva', prsp)

  # texto = pytesseract.image_to_string(dst, lang='spa')
  # print('texto: ', texto)

cv2.imshow('original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()