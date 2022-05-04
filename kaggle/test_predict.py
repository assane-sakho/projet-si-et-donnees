import warnings
import os
import gc
import sys
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from tqdm import tqdm

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

print('b')

DATA_DIR = 'data'
ROOT_DIR = Path('.')

# For demonstration purpose, the classification ignores attributes (only categories),
# and the image size is set to 512, which is the same as the size of submission masks
NUM_CATS = 46
IMAGE_SIZE = 512

COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'

# Set Config

print('c')

class FashionConfig(Config):
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # a memory error occurs when IMAGES_PER_GPU is too high
    
    BACKBONE = 'resnet50'
    
    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE    
    IMAGE_RESIZE_MODE = 'none'
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    #DETECTION_NMS_THRESHOLD = 0.0
    
    # STEPS_PER_EPOCH should be the number of instances 
    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;
    # however, due to the time limit, I set them so that this kernel can be run in 9 hours
    STEPS_PER_EPOCH = 2
    VALIDATION_STEPS = 3
    
config = FashionConfig()

# Make Datasets
with open("/data/label_descriptions.json") as f:
    label_descriptions = json.load(f)

label_names = [x['name'] for x in label_descriptions['categories']]
print(label_names)
print('aaaa')


glob_list = glob.glob('./fashion20220503T1556/mask_rcnn_fashion_0002.h5')
model_path = glob_list[0] if glob_list else ''

class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config,
                          model_dir='')

# assert model_path != '', "Provide path to trained weights"
# print("Loading weights from ", model_path)
# model.load_weights(model_path, by_name=True)


sample_df = pd.read_csv("/data/sample_submission.csv")
# sample_df.head()

def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
    return img

# # Convert data to run-length encoding

def to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle

# Since the submission system does not permit overlapped masks, we have to fix them
def refine_masks(masks, rois):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    for m in range(masks.shape[-1]):
        mask_pos = np.where(masks[:, :, m]==True)
        if np.any(mask_pos):
            y1, x1 = np.min(mask_pos, axis=1)
            y2, x2 = np.max(mask_pos, axis=1)
            rois[m, :] = [y1, x1, y2, x2]
    return masks, rois


# sub_list = []
# missing_count = 0
# for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
#     try:
#         image = resize_image('/data/test/' + row['ImageId'])
#         result = model.detect([image])[0]
#         if result['masks'].size > 0:
#             masks, _ = refine_masks(result['masks'], result['rois'])
#             for m in range(masks.shape[-1]):
#                 mask = masks[:, :, m].ravel(order='F')
#                 rle = to_rle(mask)
#                 label = result['class_ids'][m] - 1
#                 sub_list.append([row['ImageId'], ' '.join(list(map(str, rle))), label])
#         else:
#             # The system does not allow missing ids, this is an easy way to fill them 
#             sub_list.append([row['ImageId'], '1 1', 23])
#             missing_count += 1
#     except:
#         print("skipped, ", row['ImageId'])
        
# submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
# print("Total image results: ", submission_df['ImageId'].nunique())
# print("Missing Images: ", missing_count)
# submission_df.head()

# submission_df.to_csv("submission.csv", index=False)

for i in range(89, 155):
    image_id = sample_df.sample()['ImageId'].values[0]
    image_path = '/data/test/' + image_id
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = model.detect([resize_image(image_path)])
    r = result[0]
    
    if r['masks'].size > 0:
        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)
        # rle = to_rle(mask)
        label = r['class_ids']
        print(label)
        print('\n')

        for m in range(r['masks'].shape[-1]):
            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 
                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y_scale = img.shape[0]/IMAGE_SIZE
        x_scale = img.shape[1]/IMAGE_SIZE   
        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
        
        masks, rois = refine_masks(masks, rois)
    else:
        masks, rois = r['masks'], r['rois']
        print('no')