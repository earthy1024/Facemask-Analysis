import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import PIL
import pathlib
from zipfile import ZipFile


def createModel():

    train = keras.preprocessing.image_dataset_from_directory(directory="Images/", validation_split=0.2, subset="training"
    , seed=123, interpolation='bilinear', follow_links=False)
    validation = keras.preprocessing.image_dataset_from_directory(directory="Images/", validation_split=0.2, subset="validation"
    , seed=123, interpolation='bilinear', follow_links=False)

    names = train.class_names

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(256,
                                                                      256,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    epochs=3
    history = model.fit(train, validation_data=validation, epochs=epochs)
    return model

def getClassification(image, model):

    img_arr = keras.preprocessing.image.img_to_array(image)
    img_arr = tf.expand_dims(img_arr, 0)
    predictions = model.predict(img_arr)
    score = tf.nn.softmax(predictions[0])
    names = ["Masked", "Poorly Masked", "Unmasked"]
    return names[np.argmax(score)]



