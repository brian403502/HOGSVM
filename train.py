from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image 
from numpy import *

def sliding_window(image):
    for y in range(0, image.shape[0], 128):
        for x in range(0, image.shape[1], 64):
            yield image[y: y + 128, x:x + 64]


orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

pos_im_path = './INRIAPerson/INRIAPerson/train_64x128_H96/pos'
neg_im_path= './INRIAPerson/INRIAPerson/Train/neg'

data= []
labels = []


for dir in glob.glob(pos_im_path+'/*.jpg'):
    print(dir)
    img = Image.open(dir)
    img = img.resize((64,128))
    gray = img.convert('L')
    gray = np.array(gray)
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

for dir in glob.glob(pos_im_path+'/*.png'):
    img = Image.open(dir)
    img = img.resize((64,128))
    gray = img.convert('L')
    gray = np.array(gray)
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)
    
for dir in glob.glob(neg_im_path+'/*.png'):
    img= Image.open(dir)
    gray= img.convert('L')
    gray = np.array(gray)
    for images in sliding_window(gray):
        if images.shape[0] != 128 or images.shape[1] != 64:
            continue
        images = cv2.resize(images,(64,128))
        fd = hog(images, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
        data.append(fd)
        labels.append(0)

for dir in glob.glob(neg_im_path+'/*.jpg'):
    img= Image.open(dir)
    gray= img.convert('L')
    gray = np.array(gray)
    for images in sliding_window(gray):
        if images.shape[0] != 128 or images.shape[1] != 64:
            continue
        images = cv2.resize(images,(64,128))
        fd = hog(images, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
        data.append(fd)
        labels.append(0)


le = LabelEncoder()
labels = le.fit_transform(labels)

print(" Training Linear SVM classifier...")
model = LinearSVC(max_iter=30000)
model.fit(data, labels)

joblib.dump(model, 'model_name.npy')
