import numpy as np
import cv2

'''
Deteccion de bordes mediante tecnica sobel
'''

exe = True
while exe:
    path_success = "/home/nbellorin/PycharmProjects/procesamientoImagenes/canny_detector/content/9.png"
    img_height = 480
    img_width = 600
    image = cv2.imread(path_success)  # '''PATH'''
    dim = (600, 400)
    # resize image
    frame = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_AREA)


    #RECOMENDABLE REALIZAR UN PROCESO DE FILTRADO
    #CONVERSION DE DATO DE TIPO ENTERO 8bit EN FLOTANTES
    frame_float=frame.astype(float)
    #KERNEL DE SOBEL
    Hsx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Hsy=np.transpose(Hsx)
    #BORDES  EN  LAS  DIRECCIONES  HORIZONTALES  Y  VERTICALES
    bordex=cv2.filter2D(frame_float,-1,Hsx)
    bordey=cv2.filter2D(frame_float,-1,Hsy)
     #CALCULO DE LA MAGNITUD DEL GRADIENTE
    Mxy=bordex**2+bordey**2 #OPERACION PIXEL POR PIXEL
    Mxy=np.sqrt(Mxy)
    #NORMALIZACION
    Mxy=Mxy/np.max(Mxy)
    #SEGMENTACION
    mask=np.where(Mxy>0.1,255,0)
    mask=np.uint8(mask)
    cv2.imshow('BORDES',mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exe = False
