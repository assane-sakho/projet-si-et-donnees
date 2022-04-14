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
def train(force_train = False):
        return ''

def predict(file_to_predict):
    return 'cuir, rouge'
