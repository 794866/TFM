def imageProcessor():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.preprocessing.image import load_img
    import numpy as np
    import argparse

    pathLoad = '/home/nbellorin/PycharmProjects/procesamientoImagenes/imageGenerator/dataGenerated/original/'
    pathSave = '/home/nbellorin/PycharmProjects/procesamientoImagenes/imageGenerator/dataGenerated/generatedPycharm/'

    print("loading example images ...")
    image = load_img(pathLoad + '3.1.png')
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    total = 0

    print(" generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir = pathSave,
                        save_prefix="image", save_format="png")

    for image in imageGen:
        total += 1
        if total == 100:
            break