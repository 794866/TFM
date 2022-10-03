from os import path

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from os import listdir

loadPath = '/home/nbellorin/PycharmProjects/procesamientoImagenes/spyderScripts/dataGenerated/licensia'
savePath = '/home/nbellorin/PycharmProjects/procesamientoImagenes/spyderScripts/dataGenerated/generated/licensia'

def firstMehod():
    print("loading example images ...")
    directory = loadPath
    for name in listdir(directory):
        filename = path.join(directory, name)
        image = load_img(filename)

        image.mode  # Comprobamos el modo de la imagen: P
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image_id = name.split('.')[0]

        aug = ImageDataGenerator(
            rotation_range=2,
            zoom_range=0.3,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
        total = 0

        print(" generating images...")
        imageGen = aug.flow(image, batch_size=1, save_to_dir=savePath,
                            save_prefix="image", save_format="jpg")

        for image in imageGen:

            total += 1

            if total == 10:
                break

def secondMethod():
    # # Importing necessary functions
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator,
    # array_to_img, img_to_array, load_img

    # Initialising the ImageDataGenerator class.
    # We will pass in the augmentation parameters in the constructor.
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5))

    directory = loadPath
    for name in listdir(directory):
        filename = path.join(directory, name)
        image = load_img(filename)
        image.mode  # Comprobamos el modo de la imagen: P
        # Loading a sample image
        img = image
        # Converting the input sample image to an array
        x = img_to_array(img)
        # Reshaping the input image
        x = x.reshape((1,) + x.shape)

        # Generating and saving 5 augmented samples
        # using the above defined parameters.
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=savePath,
                                  save_prefix='image', save_format='jpg'):
            i += 1
            if i > 10:
                break

def thirdMethod():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,  # horizontal shift
        height_shift_range=0.2,  # vertical shift
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.5, 1.5))

    directory = loadPath
    for name in listdir(directory):
        filename = path.join(directory, name)
        image = load_img(filename)
        image.mode  # Comprobamos el modo de la imagen: P
        # Loading a sample image
        img = image
        # Converting the input sample image to an array
        x = img_to_array(img)
        # Reshaping the input image
        x = x.reshape((1,) + x.shape)

        # Generating and saving 5 augmented samples
        # using the above defined parameters.
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=savePath,
                                  save_prefix='image', save_format='jpg'):
            i += 1
            if i > 100:
                break

firstMehod()
# secondMethod()
#thirdMethod()
