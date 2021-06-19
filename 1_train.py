from skimage.feature import hog as hog2
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
import utils
from PIL import Image

def sliding_window(image):
    for y in range(0, image.shape[0], 128):
        for x in range(0, image.shape[1], 64):
            yield image[y: y + 128, x:x + 64]

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

pos_im_path = './INRIAPerson/INRIAPerson/96X160H96/Train/pos'
neg_im_path= './INRIAPerson/INRIAPerson/Train/neg'
annotation_path = './INRIAPerson/INRIAPerson/Train/annotations/edited/annotation.txt'

data= []
labels = []

for dir in glob.glob(pos_im_path+'/*.jpg'):
    img = cv2.imread(dir)
    img = img[16:16+128,13:13+64]
    img = img_as_ubyte(img)
    fd = hog.compute(img)
    data.append(fd.flatten())
    labels.append(1)

for dir in glob.glob(pos_im_path+'/*.png'):
    img = Image.open(dir)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = img[16:16+128,16:16+64]
    img = img_as_ubyte(img)
    fd = hog.compute(img)
    data.append(fd.flatten())
    labels.append(1)
    
for dir in glob.glob(neg_im_path+'/*.png'):
    img = cv2.imread(dir)
    img = img_as_ubyte(img)
    h = hog.compute(img,winStride=(128,128))
    h = h.reshape(-1,3780)
    for i in range(0,h.shape[0]):
        fd = h[i]
        data.append(fd)
        labels.append(-1)

for dir in glob.glob(neg_im_path+'/*.jpg'):
    img = cv2.imread(dir)
    img = img_as_ubyte(img)
    h = hog.compute(img,winStride=(128,128))
    h = h.reshape(-1,3780)
    for i in range(0,h.shape[0]):
        fd = h[i]
        data.append(fd)
        labels.append(-1)

print(len(data))

le = LabelEncoder()
labels = le.fit_transform(labels)

print(" Training Linear SVM classifier...")
model = SVC(kernel='linear',probability=True)
model.fit(data, labels)

joblib.dump(model, 'hog_model.npy')
