# input images' dimensions = 500x500
"""1_kurus": "1 Kurus,Turkish Lira,turkey",
    "5_kurus": "5 Kurus,Turkish Lira,turkey",
    "10_kurus": "10 Kurus,Turkish Lira,turkey",
    "25_kurus": "25 Kurus,Turkish Lira,turkey",
    "50_kurus": "50 Kurus,Turkish Lira,turkey",
    "1_lira": "1 Lira,Turkish Lira,turkey","""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(6, activation="softmax")


])
from keras.optimizers import SGD

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0015), metrics=['accuracy'])

train_generator = ImageDataGenerator().flow_from_directory(

    "train",
    target_size=(500, 500),
    classes=['1_kurus', '5_kurus','10_kurus','25_kurus', '50_kurus', '1_lira'],
    batch_size=5,

)

validation_generator = ImageDataGenerator().flow_from_directory(
    "validation",
    target_size=(500, 500),
    classes=['1_kurus', '5_kurus', '10_kurus', '25_kurus', '50_kurus', '1_lira'],
    batch_size=4,
)

history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=20,
    verbose=1,
    validation_data=validation_generator

)
