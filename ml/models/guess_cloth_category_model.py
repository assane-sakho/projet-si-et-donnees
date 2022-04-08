import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import convolve
from joblib import dump, load
import os.path

dumped_model_path = 'dumped_models/guess_cloth_category_model.joblib'

'''
    Check if the model is already dumped
'''
def model_is_dumped():
    return os.path.exists(dumped_model_path)

'''
    Train the model
'''
def train ():
    if model_is_dumped() == False : #check if the model is already trained
        data_dir='/data' #directory used to store images
        batch_size = 3
        img_height = 200
        img_width = 200
        seed = 42
        validation_split = 0.2

        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split = validation_split,
            subset="training",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        val_data = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split = validation_split,
            subset="validation",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        class_names = val_data.class_names
        
        print(class_names)

        num_classes = len(class_names)

        model = tf.keras.Sequential([
            layers.experimental.preprocessing.Rescaling(1./255),
            layers.Conv2D(128,4, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64,4, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32,4, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16,4, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64,activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],)

        logdir="logs"

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=logdir,
                                                        embeddings_data=train_data)

        model.fit( 
            train_data,
            validation_data=val_data,
            epochs=2,
            callbacks=[tensorboard_callback]
        )    

        model.summary()
        dump(model, dumped_model_path) 
    return 'ok'

def predict (file_to_predict):
    for file_ in file_to_predict:
        image_to_predict = cv2.imread(file_,cv2.IMREAD_COLOR)
        plt.imshow(cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2RGB))
        plt.show()
        img_to_predict = np.expand_dims(cv2.resize(image_to_predict,(200,200)), axis=0) 
        res = np.argmax(model.predict(img_to_predict), axis=-1)
        print(model.predict(img_to_predict))
        i = 0
        cloth_result = []
        for vetement in class_names:
            if res == i:
                print(vetement) 
                cloth_result.append(vetement)
            i = i + 1

        return '\n'.join(cloth_result)
