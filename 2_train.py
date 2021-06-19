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
import cascade
import numpy as np
import cv2
import glob
import utils
from PIL import Image
import pickle
import random

def random_crop(image):
    y = random.randrange(0, image.shape[0]-127)
    x = random.randrange(0, image.shape[1]-63)
    return image[y: y + 128, x:x + 64]


def sliding_window(image):
    imgs = []
    for y in range(0, image.shape[0]-127, 128):
        for x in range(0, image.shape[1]-63, 64):
            imgs.append(image[y: y + 128, x:x + 64])
    return random.sample(imgs, 3)

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

pos_im_path = './INRIAPerson/INRIAPerson/96X160H96/Train/pos'
neg_im_path= './INRIAPerson/INRIAPerson/Train/neg'
annotation_path = './INRIAPerson/INRIAPerson/Train/annotations/edited/annotation.txt'

pos_img_list = []
neg_img_list = []

for dir in glob.glob(pos_im_path+'/*.jpg'):
    img = cv2.imread(dir)
    img = img[16:16+128,13:13+64]
    img = img_as_ubyte(img)
    pos_img_list.append(img)

for dir in glob.glob(pos_im_path+'/*.png'):
    img = Image.open(dir)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = img[16:16+128,16:16+64]
    img = img_as_ubyte(img)
    pos_img_list.append(img)
    
for dir in glob.glob(neg_im_path+'/*.png'):
    img = cv2.imread(dir)
    # image = random_crop(img)
    # image = img_as_ubyte(image)
    # neg_img_list.append(image)
    for image in sliding_window(img):
        image = img_as_ubyte(image)
        neg_img_list.append(image)

for dir in glob.glob(neg_im_path+'/*.jpg'):
    img = cv2.imread(dir)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # image = random_crop(img)
    # image = img_as_ubyte(image)
    # neg_img_list.append(image)
    for image in sliding_window(img):
        image = img_as_ubyte(image)
        neg_img_list.append(image)

print(len(pos_img_list),len(neg_img_list))

print(" Training classifiers...")
model = cascade.EntireClassifier()
model.train(pos_img_list, neg_img_list)

with open("cascade.pkl","wb") as fw:
    pickle.dump(model, fw)