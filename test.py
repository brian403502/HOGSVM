from skimage.feature import hog
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score, precision_recall_curve, plot_precision_recall_curve
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

def sliding_window(image):
    for y in range(0, image.shape[0], 128):
        for x in range(0, image.shape[1], 64):
            yield image[y: y + 128, x:x + 64]


orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

testdata = []
testlabels = []
test_pos_im_path = './INRIAPerson/INRIAPerson/test_64x128_H96/pos'
test_neg_im_path = './INRIAPerson/INRIAPerson/Test/neg'
model = joblib.load('model_name.npy')
for dir in glob.glob(test_pos_im_path+'/*.jpg'):
    img= Image.open(dir)
    img = img.resize((64,128))
    gray= img.convert('L')
    gray= np.array(gray)
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    testdata.append(fd)
    testlabels.append(1)
for dir in glob.glob(test_pos_im_path+'/*.png'):
    img= Image.open(dir)
    img = img.resize((64,128))
    gray= img.convert('L')
    gray= np.array(gray)
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    testdata.append(fd)
    testlabels.append(1)

for dir in glob.glob(test_neg_im_path+'/*.jpg'):
    img= Image.open(dir)
    img = img.resize((64,128))
    gray= img.convert('L')
    gray = np.array(gray)
    for images in sliding_window(gray):
        if images.shape[0] != 128 or images.shape[1] != 64:
            continue
        fd = hog(images, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
        testdata.append(fd)
        testlabels.append(0)   
for dir in glob.glob(test_neg_im_path+'/*.png'):
    img= Image.open(dir)
    img = img.resize((64,128))
    gray= img.convert('L')
    gray = np.array(gray)
    for images in sliding_window(gray):
        if images.shape[0] != 128 or images.shape[1] != 64:
            continue
        fd = hog(images, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
        testdata.append(fd)
        testlabels.append(0)   
testdata = np.array(testdata)
le = LabelEncoder()
testlabels = le.fit_transform(testlabels)
print(" Evaluating classifier on test data ...")
predictions = model.predict(testdata)
print(classification_report(testlabels, predictions))

y_test_pred = model.decision_function(testdata)
test_fpr, test_tpr, te_thresholds = roc_curve(testlabels, y_test_pred)
average_precision = average_precision_score(testlabels, y_test_pred)

plt.grid()
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(roc_auc_score(testlabels, y_test_pred))) # , label=" AUC TEST ="+str(roc_auc_score(test_fpr, test_tpr))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

disp = plot_precision_recall_curve(model, testdata, testlabels)
disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()