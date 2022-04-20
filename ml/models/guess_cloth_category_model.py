import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from joblib import dump, load
import os.path

dumped_model_path = 'dumped_models/guess_cloth_category_model.joblib'
dumped_model_class_name_path = 'dumped_models/guess_cloth_category_class_name.joblib'

'''
    Check if the model is already dumped
'''
def is_model_dumped():
    return os.path.exists(dumped_model_path)


'''
    Train the model
'''
def train_category(force_train = False):
    if is_model_dumped() == False or force_train == True :  # check if the model is already trained
        data_dir = '/data'  # directory used to store images
        batch_size = 3
        img_height = 200
        img_width = 200
        seed = 42
        validation_split = 0.2

        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=(img_height, img_width),
            batch_size=batch_size,
        )

        val_data = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=validation_split,
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
            layers.Conv2D(128, 4, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 4, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 4, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 4, activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss=tf.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'],)

        logdir = "logs"

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images=logdir,
                                                           embeddings_data=train_data)

        model.fit(
            train_data,
            validation_data=val_data,
            epochs=2,
            callbacks=[tensorboard_callback]
        )

        model.summary()
        
        #persist model & class names
        dump(model, dumped_model_path)
        dump(class_names, dumped_model_class_name_path)

    return 'ok'


def predict_category(file_to_predict):
    if is_model_dumped():
        img = cv2.imdecode(np.fromstring(file_to_predict.read(),np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (224, 224))
        img_to_predict = np.expand_dims(cv2.resize(img, (200, 200)), axis=0)

        #load data from files
        model = load(dumped_model_path)
        class_names = load(dumped_model_class_name_path)

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
