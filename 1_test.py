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


def sliding_window(image):
    for y in range(0, image.shape[0], 128):
        for x in range(0, image.shape[1], 64):
            yield image[y: y + 128, x:x + 64]

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

testdata = []
testlabels = []
test_pos_im_path = './INRIAPerson/INRIAPerson/70X134H96/Test/pos'
test_neg_im_path = './INRIAPerson/INRIAPerson/Test/neg'
annotation_path = './INRIAPerson/INRIAPerson/Test/annotations/edited/annotation.txt'
model = joblib.load('hog_model.npy')


gt_ids = []
gt_bboxes = []
with open(annotation_path, 'r') as f:
    for line in f:
        gt_id, xmin, ymin, xmax, ymax = line.split(' ')
        gt_ids.append(gt_id)
        gt_bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

for dir in glob.glob(test_pos_im_path+'/*.jpg'):
    img = cv.imread(dir)
    img = img[3:3+128,3:3+64]
    img = img_as_ubyte(img)
    fd = hog.compute(img)
    testdata.append(fd.flatten())
    testlabels.append(1)

for dir in glob.glob(test_pos_im_path+'/*.png'):
    img = Image.open(dir)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img = img[3:3+128,3:3+64]
    img = img_as_ubyte(img)
    fd = hog.compute(img)
    testdata.append(fd.flatten())
    testlabels.append(1)
    
for dir in glob.glob(test_neg_im_path+'/*.png'):
    img = cv.imread(dir)
    img = img_as_ubyte(img)
    h = hog.compute(img,winStride=(128,128))
    h = h.reshape(-1,3780)
    for i in range(0,h.shape[0]):
        fd = h[i]
        testdata.append(fd)
        testlabels.append(-1)

for dir in glob.glob(test_neg_im_path+'/*.jpg'):
    img = cv.imread(dir)
    img = img_as_ubyte(img)
    h = hog.compute(img,winStride=(128,128))
    h = h.reshape(-1,3780)
    for i in range(0,h.shape[0]):
        fd = h[i]
        testdata.append(fd)
        testlabels.append(-1)

testdata = np.array(testdata)
print(testdata.shape)
testlabels = np.array(testlabels)
confidences = model.predict_proba(testdata)
confidences = confidences[:,1]-confidences[:,0]
utils.report_accuracy(confidences, testlabels)
