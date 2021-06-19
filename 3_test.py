
import joblib
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import utils
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import cv2 as cv

# class LocalBinaryPatterns:
# 	def __init__(self, numPoints, radius):
# 		# store the number of points and radius
# 		self.numPoints = numPoints
# 		self.radius = radius
 
# 	def describe(self, image, eps=1e-7):
# 		# compute the Local Binary Pattern representation
# 		# of the image, and then use the LBP representation
# 		# to build the histogram of patterns
# 		lbp = local_binary_pattern(image, self.numPoints,
# 			self.radius, method="uniform")
# 		(hist, _) = np.histogram(lbp.ravel(),
# 			bins=np.arange(0, self.numPoints + 3),
# 			range=(0, self.numPoints + 2))
 
# 		# normalize the histogram
# 		hist = hist.astype("float")
# 		hist /= (hist.sum() + eps)
 
# 		# return the histogram of Local Binary Patterns
# 		return hist

def get_lbp_data(image, hist_size=256, lbp_radius=1, lbp_point=8):
    hist_tot = []
    for img in lbp_sliding_window(image):
        lbp = local_binary_pattern(img, lbp_point, lbp_radius, 'default')
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        eps=1e-7
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        hist_tot.append(hist)
    hist_tot = np.array(hist_tot)
    hist_tot = hist_tot.flatten()
    return hist_tot

def lbp_sliding_window(image):
    for y in range(0,image.shape[0]-15,16):
        for x in range(0, image.shape[1]-15, 16):
            yield image[y:y+16,x:x+16]

def sliding_window(image):
    for y in range(0, image.shape[0]-127, 128):
        for x in range(0, image.shape[1]-63, 64):
            yield image[y: y + 128, x:x + 64]


testdata = []
testlabels = []
test_pos_im_path = './INRIAPerson/INRIAPerson/70X134H96/Test/pos'
test_neg_im_path = './INRIAPerson/INRIAPerson/Test/neg'
annotation_path = './INRIAPerson/INRIAPerson/Test/annotations/edited/annotation.txt'
model = joblib.load('lbp_model.npy')
# desc = LocalBinaryPatterns(24, 8)

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
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = img_as_ubyte(img)
    # hist = desc.describe(img)
    hist = get_lbp_data(img)
    testdata.append(hist)
    testlabels.append(1)

for dir in glob.glob(test_pos_im_path+'/*.png'):
    img = Image.open(dir)
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
    img = img[3:3+128,3:3+64]
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = img_as_ubyte(img)
    # hist = desc.describe(img)
    hist = get_lbp_data(img)
    testdata.append(hist)
    testlabels.append(1)
    
for dir in glob.glob(test_neg_im_path+'/*.png'):
    img = cv.imread(dir)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = img_as_ubyte(img)
    for image in sliding_window(img):
        image = img_as_ubyte(image)
        # hist = desc.describe(image)
        hist = get_lbp_data(image)
        testdata.append(hist)
        testlabels.append(-1)

for dir in glob.glob(test_neg_im_path+'/*.jpg'):
    img = cv.imread(dir)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = img_as_ubyte(img)
    for image in sliding_window(img):
        image = img_as_ubyte(image)
        # hist = desc.describe(image)
        hist = get_lbp_data(image)
        testdata.append(hist)
        testlabels.append(-1)

# testdata = np.array(testdata)
# print(testdata.shape)
# testlabels = np.array(testlabels)
confidences = model.predict_proba(testdata)
confidences = confidences[:,1]-confidences[:,0]
utils.report_accuracy(confidences, testlabels)
