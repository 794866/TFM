# -*- coding: utf-8 -*-
"""Copia de kfold_training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K7AJ-O1oUZbUSYAH5j-JPgDRk_doy-qd
"""

import os
import shutil
from tensorflow.keras.utils import to_categorical
import time
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split


from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from tensorflow.keras.models import load_model
import re

from matplotlib import pyplot as plt

from numpy.random import seed
import tensorflow as tf
from tensorflow.keras import backend as k
import os
import PIL
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from google.colab import drive
drive.mount('/content/gdrive')

pathToImages = '/content/gdrive/MyDrive/TFM/data/set'
dirname = os.path.join(os.getcwd(), pathToImages)
imgpath = dirname + os.sep 
 
images = []
directories = []
dircount = []
prevRoot=''
cant=0

img_height = 400
img_width = 600
channel = 3
 
print("leyendo imagenes de ",imgpath)
 
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            #image = plt.imread(filepath)
            image = PIL.Image.open(filepath).convert('RGB')
            image = image.resize((img_width,img_height),PIL.Image.ANTIALIAS)
            plt.imshow(image)
            #cv2.waitkey(0)
            image_array = img_to_array(image)
            images.append(image_array)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)

 
dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

from keras.utils import np_utils
labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))
 
deportes=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    deportes.append(name[len(name)-1])
    indice=indice+1


X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
X = X.astype('float32')
targets=np.array(labels)#.astype('float32')
y = np_utils.to_categorical(targets) 
num_classes= y.shape[1]

del images
 
train_generator = ImageDataGenerator(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest')

valid_generator = ImageDataGenerator(rescale = 1. / 255)

# Commented out IPython magic to ensure Python compatibility.
''' Funcion para visualizar los resultados '''    
def plot_graphs(cnn):
    import matplotlib.pyplot as plt
#     %matplotlib inline
    plt.figure().set_size_inches(20, 8)
    
    # Accuracy vs Validation Accuracy
    plt.subplot(1, 2, 1)
    #plt.figure(0)
    plt.plot(cnn.history['accuracy'],'r')
    plt.plot(cnn.history['val_accuracy'],'g')
    plt.xticks(np.arange(0, epochs, 10))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy / Loss")
    plt.title("Accuracy vs Loss")
    plt.legend(['acc','val_acc'])
    plt.ylim(0, 1.01)
    #plt.show()

    # Loss vs Validation Loss
    plt.subplot(1, 2, 2)
    #plt.figure(1)
    plt.plot(cnn.history['loss'],'b', linestyle = '--')
    plt.plot(cnn.history['val_loss'],'y', linestyle = '--')
    plt.xticks(np.arange(0, epochs, 10))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])
    plt.show()

def createModel():
    model= Sequential()
            #n mapas #filtro
    model.add(Conv2D(16,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64,(5,5),activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(528,activation='relu'))
    model.add(Dense(342,activation='relu'))
    model.add(Dense(128,activation='relu'))
    #capa de salida (Si fuera binaria
    #model.add(Dense(1,activation='sigmoid')) #binaria
    #model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) #binaria
    #capa de salida
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    return model

start = time.time()

#Definimos el K fold, importante, creo que automaticamente da un 70/30 para train y test
kfold= StratifiedKFold(n_splits=5,shuffle=True)
cvscores = []
resultados = []

for train,test in kfold.split(X,labels):
    #definimos el modelo keras

    #De los datos para entrenamiento, saco para la validacion
    X_train,X_validation,y_train,y_validation = train_test_split(X[train],y[train],test_size=0.20) 
    # Entrenar
    # model.fit(X[train],y[train],epochs=10,batch_size=10,verbose=1) <---- si no tuviera conjunto para validacion
    # verbose=0 no muestra paso por paso
    batch_size = 32
    epochs = 10
    model = createModel()
    model_t = model.fit_generator(train_generator.flow(X_train, y_train, batch_size=batch_size, shuffle = True), steps_per_epoch=len(X_train) // batch_size, epochs=epochs, verbose=0, validation_data=valid_generator.flow(X_validation, y_validation, shuffle = True), validation_steps=len(X_validation) // batch_size)

    #model.fit(X_train,y_train,validation_data=(X_validation,y_validation),epochs=10,batch_size=20,verbose=0)# training / validation
    # Evaluar
    scores=model.evaluate(X[test],y[test],verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
    cvscores.append(scores[1]*100)

    from sklearn.metrics import confusion_matrix, accuracy_score
    predictions = model.predict(X[test], steps=len(X[test]), verbose=0)
    y_pred = np.argmax(predictions, axis=-1)
    y_true=np.argmax(y[test], axis=-1)

    cm = confusion_matrix(y_true, y_pred)

    plot_graphs(model_t)
    
    previsoes = model.predict(X[test])
    previsoes = (previsoes > 0.5) # Se a saida for > 0.5 -> classe 1 sen??o classe 0
    accuracy = accuracy_score(y[test], previsoes)
    print('Accuracy: ', accuracy)

    resultados.append(accuracy)

print('confussion matrix:')
print(cm)

print("%.2f%% (+/- %.2f%%)" %(np.mean(cvscores),np.std(cvscores)))
#print(model.metrics_names)

done = time.time()
elapsed = done - start

print("Time: %0.2f sec"%(elapsed))

print('listo')

path_to_save =  '/content/gdrive/MyDrive/TFM/modelos/'
model.save(path_to_save+'new_model_trained.h5')

import seaborn as sns
import matplotlib.pyplot as plt

class_names = ['error', 'front', 'back', 'passport']
cm = [[28, 3, 0, 1],
 [1, 57, 0, 0],
 [0, 0, 57, 0],
 [0, 0, 0, 36]]

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(class_names, fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(class_names, fontsize = 10)
plt.yticks(rotation=0)

plt.title('Refined Confusion Matrix', fontsize=20)

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(np.asarray(cm), class_names)

from tensorflow.keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
import PIL

from google.colab import drive
drive.mount('/content/gdrive')

#sport_model.save(path+"firstModel_mnist.h5py")
path =  '/content/gdrive/MyDrive/TFM/'

#modelt = load_model(path+"modelos/kfold_model_trained_97_26_Acc.h5")
modelt = load_model(path+"modelos/model_trained.h5")
#modelt = custom_vgg_model

print(deportes)

##Esta imagen no forma parte del dataset
path = "/content/gdrive/MyDrive/TFM/data/set/"
names = deportes

filepath = path+"back/Selecci??n_279.png"
image = PIL.Image.open(filepath)
image = image.resize((600,400),PIL.Image.ANTIALIAS)
xt = np.asarray(image)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)
preds1 = model.predict(xt)

print(names[np.argmax(preds1)])
plt.imshow(cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()



filepath = path+"front/Selecci??n_075.png"
image = PIL.Image.open(filepath)
image = image.resize((600,400),PIL.Image.ANTIALIAS)
xt = np.asarray(image)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)
preds1 = model.predict(xt)

print(names[np.argmax(preds1)])
plt.imshow(cv2.cvtColor(np.asarray(image),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


imaget_path = path+"back/Selecci??n_279.png"
img = cv2.imread(imaget_path)
imaget=cv2.resize(cv2.imread(imaget_path), (600, 400), interpolation = cv2.INTER_AREA)
xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)
preds = model.predict(xt)

print(names[np.argmax(preds)])
plt.imshow(cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

imaget_path = path+"prueba/3.png"
img = cv2.imread(imaget_path)
imaget=cv2.resize(cv2.imread(imaget_path), (600, 400), interpolation = cv2.INTER_AREA)
xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)
preds = model.predict(xt)

print(names[np.argmax(preds)])
plt.imshow(cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

imaget_path = path+"prueba/error.jpeg"
img = cv2.imread(imaget_path)
imaget=cv2.resize(cv2.imread(imaget_path), (600, 400), interpolation = cv2.INTER_AREA)
xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)
preds = modelt.predict(xt)

print(names[np.argmax(preds)])
plt.imshow(cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()