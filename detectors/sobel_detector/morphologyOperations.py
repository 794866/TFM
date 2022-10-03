import numpy as np
import cv2
import os


def CargarImagen(path):
    ruta = os.getcwd()
    path_success = path
    nombreArchivo = r'/canny_detector/content/9.png'
    rutaAbrir = os.path.join(ruta, nombreArchivo)
    imagen = cv2.imread(path_success)

    h, w, c = imagen.shape
    factor = 0.85

    nAlto, nAncho = int(factor * h), int(factor * w)
    imagenT = cv2.resize(imagen, (nAncho, nAlto))

    #cv2.imshow('img', imagenT)
    #cv2.moveWindow('img', 0, 0)

    return imagenT


def Laplacian(imagen):
    img = imagen.copy()
    h, w, c = img.shape

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    cv2.imshow('laplacian', laplacian)
    cv2.moveWindow('laplacian', w, 0)

    return None


def Sobel(imagen):
    img = imagen.copy()
    h, w, c = img.shape

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobelC = cv2.bitwise_or(sobelX, sobelY)

    cv2.imshow('sobel x', sobelX)
    cv2.moveWindow('sobel x', w, 0)

    cv2.imshow('sobel y', sobelY)
    cv2.moveWindow('sobel y', w, h)

    cv2.imshow('sobel c', sobelC)
    cv2.moveWindow('sobel c', w, 2 * h)

    canny = cv2.Canny(sobelC, 50, 150)
    cv2.imshow('canny', canny)

    # _,cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# OpenCV 3
    cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV 4

    return None

def SobelAndCanny(imagen):
    img = imagen.copy()
    h, w, c = img.shape

    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobelC = cv2.bitwise_or(sobelX, sobelY)

    # cv2.imshow('sobel x', sobelX)
    # cv2.moveWindow('sobel x', w, 0)
    #
    # cv2.imshow('sobel y', sobelY)
    # cv2.moveWindow('sobel y', w, h)

    img_height = 480
    img_width = 600
    imag = cv2.resize(sobelC, (img_width, img_height), interpolation=cv2.INTER_AREA)
    #cv2.imshow('sobel imag', imag)
    #cv2.moveWindow('sobel c', w, 2 * h)

    # canny = cv2.Canny(sobelC,150,250,apertureSize = 5, L2gradient = True)
    # cv2.imshow('canny', canny)
    # canny = cv2.dilate(canny, None, iterations=1)

    canny = cv2.Canny(sobelC, 0, 150)

    ''' Aplica operaciones morfológicas para tratar de corregir lineas discontinuas
        - diltation = agregar grosor a las lineas.                        
        - closing = limpiar puntos dentro de las lineas de los contornos.
    '''
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(canny, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('morphology', closing)

    kernelSizes = [(3, 3), (5, 5), (7, 7)]
    for kernelSize in kernelSizes:
        # construct a rectangular kernel and apply a "morphological
        # gradient" operation to the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)
    #cv2.imshow('morphology gradient', gradient)


    image = cv2.resize(sobelC, (img_width, img_height), interpolation=cv2.INTER_AREA)
    #cv2.imshow('canny image', image)
    cnts = cv2.findContours(gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # OpenCV 4
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    def ordenar_puntos(puntos):
        n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
        y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
        x1_order = y_order[:2]
        x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
        x2_order = y_order[2:4]
        x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

        return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

    for c in cnts:
        
        epsilon = 0.090 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        print("aprox", approx)

        if len(approx)==4:

            area = cv2.contourArea(c)
            print("Area ", area)

            perimetro = cv2.arcLength(c, True)
            print("perimetro ", perimetro)


            #cv2.drawContours(img, [approx], 0, (0, 255, 255), 2)
            #cv2.imshow('origin', img)

            puntos = ordenar_puntos(approx)
            # cv2.circle(img, tuple(puntos[0]), 7, (255, 0, 0), 2)
            # cv2.circle(img, tuple(puntos[1]), 7, (0, 255, 0), 2)
            # cv2.circle(img, tuple(puntos[2]), 7, (0, 0, 255), 2)
            # cv2.circle(img, tuple(puntos[3]), 7, (255, 255, 0), 2)

            pts1 = np.float32(puntos)
            pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            prsp = cv2.warpPerspective(img, M, (w, h))

            cv2.imshow('perspectiva', prsp)
            cv2.drawContours(img, [approx], 0, (0, 255, 255), 2)
            cv2.imshow('origin', img)

    return None

def Canny(imagen):
    img = imagen.copy()
    h, w, c = img.shape

    imgGris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGris, (5, 5), 0)

    imgCanny = cv2.Canny(imgBlur, 30, 150)

    cv2.imshow('img canny', imgCanny)
    cv2.moveWindow('img canny', w, 0)



if __name__ == '__main__':
    path_success = "/home/nbellorin/Descargas/pasarElDetector/pass1.jpeg"
    img = CargarImagen(path_success)
    #Laplacian(img)
    #Sobel(img)
    #Canny(img)
    #DetectarBordeMoneda(img)
    SobelAndCanny(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
