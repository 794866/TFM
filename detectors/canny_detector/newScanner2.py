import cv2
import numpy as np

# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

"""
    CONFIGURATIONS
        canny = cv2.Canny(gray, 100, 200, apertuimagereSize = 5, L2gradient = True) #canny = cv2.Canny(image,100,200, L2gradient = True)
        
        configuration success to first 4 images
        canny = cv2.Canny(image, 100, 200, apertureSize = 5, L2gradient = True) #canny = cv2.Canny(image,100,200, L2gradient = True)
        
        canny = cv2.Canny(image,0,0, L2gradient = True)
        canny = cv2.Canny(gray, 10, 150)
        canny = cv2.Canny(image, 10, 150)
"""

path_success = "/home/nbellorin/PycharmProjects/procesamientoImagenes/canny_detector/content/9.png"


def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]


img_height = 480
img_width = 900
image = cv2.imread(path_success)  # '''PATH'''
dim = (640, 480)
# resize image
image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((5, 5), np.uint8)

#ESCALA DE GRISES
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(image, 10, 150, L2gradient = True)
canny = cv2.dilate(canny, None, iterations=1)

cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]
# print(sorted(cnts, key=cv2.contourArea, reverse=True)[:1])
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

for c in cnts:
    epsilon = 0.0278888 * cv2.arcLength(c, False)
    approx = cv2.approxPolyDP(c, epsilon, True)

    print("puntos: ",len(approx))
    if len(approx)==4:
        puntos = ordenar_puntos(approx)
        # cv2.circle(image, tuple(puntos[0]), 7, (255, 0, 0), 2)
        # cv2.circle(image, tuple(puntos[1]), 7, (0, 255, 0), 2)
        # cv2.circle(image, tuple(puntos[2]), 7, (0, 0, 255), 2)
        # cv2.circle(image, tuple(puntos[3]), 7, (255, 255, 0), 2)
        print("PUNTOS", puntos)

        pts1 = np.float32(puntos)
        pts2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        prsp = cv2.warpPerspective(image, M, (img_width, img_height))
        cv2.imshow('perspectiva', prsp)
        cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
        cv2.imshow('origin', image)


        # texto = pytesseract.image_to_string(dst, lang='spa')
        # print('texto: ', texto)

#cv2.imshow('original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
#CONFIGURATIONS
#canny = cv2.Canny(gray, 100, 200, apertuimagereSize = 5, L2gradient = True) #canny = cv2.Canny(image,100,200, L2gradient = True)

#configuration success to first 4 images
canny = cv2.Canny(image, 100, 200, apertureSize = 5, L2gradient = True) #canny = cv2.Canny(image,100,200, L2gradient = True)

canny = cv2.Canny(image,0,0, L2gradient = True)
canny = cv2.Canny(gray, 10, 150)
canny = cv2.Canny(image, 10, 150)
"""
