import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from skimage import img_as_ubyte
import numpy as np
import cv2
import glob
from skimage.feature import local_binary_pattern
from PIL import Image

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

pos_im_path = './INRIAPerson/INRIAPerson/96X160H96/Train/pos'
neg_im_path= './INRIAPerson/INRIAPerson/Train/neg'
annotation_path = './INRIAPerson/INRIAPerson/Train/annotations/edited/annotation.txt'

data= []
labels = []

# desc = LocalBinaryPatterns(24, 8)

for dir in glob.glob(pos_im_path+'/*.jpg'):
    img = cv2.imread(dir)
    img = img[16:16+128,13:13+64]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img_as_ubyte(img)
    # hist = desc.describe(img)
    hist = get_lbp_data(img)
    data.append(hist)
    labels.append(1)

for dir in glob.glob(pos_im_path+'/*.png'):
    img = Image.open(dir)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = img[16:16+128,16:16+64]
    img = img_as_ubyte(img)
    # hist = desc.describe(img)
    hist = get_lbp_data(img)
    data.append(hist)
    # print('asdf')
    # print(hist.shape)
    labels.append(1)
    
for dir in glob.glob(neg_im_path+'/*.png'):
    img = cv2.imread(dir)
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for image in sliding_window(img):
        image = img_as_ubyte(image)
        # hist = desc.describe(image)
        hist = get_lbp_data(image)
        data.append(hist)
        # print('asdf1')
        # print(hist.shape)
        labels.append(-1)


for dir in glob.glob(neg_im_path+'/*.jpg'):
    img = cv2.imread(dir)
    img = img_as_ubyte(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for image in sliding_window(img):
        image = img_as_ubyte(image)
        # hist = desc.describe(image)
        hist = get_lbp_data(image)
        data.append(hist)
        # print('asdf2')
        # print(hist.shape)
        labels.append(-1)

print(len(data))

le = LabelEncoder()
labels = le.fit_transform(labels)

print(" Training Linear SVM classifier...")
model = SVC(kernel='linear',probability=True)
model.fit(data, labels)

joblib.dump(model, 'lbp_model.npy')
