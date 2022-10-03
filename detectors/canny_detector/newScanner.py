import cv2
import cv2 as cv
import numpy as np

# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path_success = "/canny_detector/content/9.png"
path_fail = "/canny_detector/content/8.png"


def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

img_height = 400
img_width = 600
image = cv2.imread(path_success)  # '''PATH'''
dim = (640, 480)
# resize image
image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 10, 150)
canny = cv2.dilate(canny, None, iterations=1)
cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
#print(sorted(cnts, key=cv2.contourArea, reverse=True)[:1])
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

cnt = cnts[0]
M = cv.moments(cnt)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

area = cv.contourArea(cnt)

epsilon = 0.1* cv.arcLength (cnt, True )
aprox = cv.approxPolyDP (cnt,epsilon, True )
k = cv.isContourConvex(cnt)


x,y,w,h = cv.boundingRect (cnt)
cv.rectangle (image,(x,y),(x+w,y+h),(0,255,0),2)

rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)

box = np.int0(box)
cv.drawContours(image,[box],0,(0,0,255),2)

#puntos = ordenar_puntos(box)

pts1 = np.float32(box)
pts2 = np.float32([[0,0],[img_width,0],[0,img_height],[img_width,img_height]])
MN = cv2.getPerspectiveTransform(pts1,pts2)
prsp = cv2.warpPerspective(image,MN,(img_width,img_height))
cv2.imshow('perspectiva', prsp)

cv2.imshow('original', image)
#
cv2.waitKey(0)
cv2.destroyAllWindows()