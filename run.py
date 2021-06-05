from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob

import numpy as np


orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

model = joblib.load('model_name.npy')

scale = 0
detections = []
img= cv2.imread("person_250.png")
h,w,_ = np.shape(img)
print(w,h)
if w > 400:
    img = cv2.resize(img,(400,int(h*400/w)))
(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.5

for resized in pyramid_gaussian(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), downscale=1.5):
    for (x,y,window) in sliding_window(resized, stepSize=10, windowSize=(winW,winH)):
        if window.shape[0] != winH or window.shape[1] !=winW:
            continue
        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')
        fds = fds.reshape(1, -1)
        pred = model.predict(fds)
        
        if pred == 1:
            if model.decision_function(fds) > 0.6: 
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0]*(downscale**scale)),
                                      int(windowSize[1]*(downscale**scale))))
    scale+=1
    
clone = resized.copy()
# for (x_tl, y_tl, _, w, h) in detections:
#     cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.5)
        
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0,255,0), 2)
cv2.imshow("Raw Detections after NMS", img)
k = cv2.waitKey(0) & 0xFF 
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('asdf.png',img)
    cv2.destroyAllWindows()

