from skimage.feature import hog
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score, precision_recall_curve, plot_precision_recall_curve
from skimage import img_as_ubyte
from imutils.object_detection import non_max_suppression
import utils
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import cv2 as cv
import cascade
import pickle
import random


def random_crop(image):
    y = random.randrange(0, image.shape[0]-127)
    x = random.randrange(0, image.shape[1]-63)
    return image[y: y + 128, x:x + 64]

test_pos_im_path = './INRIAPerson/INRIAPerson/70X134H96/Test/pos'
test_neg_im_path = './INRIAPerson/INRIAPerson/Test/neg'
annotation_path = './INRIAPerson/INRIAPerson/Test/annotations/edited/annotation.txt'
with open("cascade.pkl","rb") as f:
    model = pickle.load(f)

img_list = []
label_list = []

for dir in glob.glob(test_pos_im_path+'/*.jpg'):
    img = cv.imread(dir)
    img = img[3:3+128,3:3+64]
    img = img_as_ubyte(img)
    img_list.append(img)
    label_list.append(1)

for dir in glob.glob(test_pos_im_path+'/*.png'):
    img = Image.open(dir)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img = img[3:3+128,3:3+64]
    img = img_as_ubyte(img)
    img_list.append(img)
    label_list.append(1)
    
for dir in glob.glob(test_neg_im_path+'/*.png'):
    img = cv.imread(dir)
    image = random_crop(img)
    image = img_as_ubyte(image)
    img_list.append(image)
    label_list.append(-1)

for dir in glob.glob(test_neg_im_path+'/*.jpg'):
    img = cv.imread(dir)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    image = random_crop(img)
    image = img_as_ubyte(image)
    img_list.append(image)
    label_list.append(-1)

print(len(img_list),len(label_list))
pred_list = np.array(model.classify_list(img_list))
label_list = np.array(label_list)
print(pred_list.shape, label_list.shape)
utils.report_accuracy(pred_list, label_list)