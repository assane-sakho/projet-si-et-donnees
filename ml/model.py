import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
import tensorflow as tf
from tensorflow import keras
from scipy.ndimage import convolve

def tt ():
    data_dir='/data'
    batch_size = 3
    img_height = 200
    img_width = 200

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    )

    val_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = val_data.class_names
    print(class_names)
    return '\n'.join(class_names)