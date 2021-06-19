import joblib
from sklearn.preprocessing import LabelEncoder
from skimage.transform import pyramid_gaussian
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score, precision_recall_curve, plot_precision_recall_curve
from skimage import color
from imutils.object_detection import non_max_suppression
import utils
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import cv2 as cv
from skimage import img_as_ubyte
import pickle

def batch_iou(boxes, box):
  """Compute the Intersection-Over-Union of a batch of boxes with another
  box.
  Args:
    box1: 2D array of [cx, cy, width, height].
    box2: a single array of [cx, cy, width, height]
  Returns:
    ious: array of a float number in range [0, 1].
  """
  lr = np.maximum(
      np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
      np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
      0
  )
  tb = np.maximum(
      np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
      np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
      0
  )
  inter = lr*tb
  union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
  return inter/union

def nms(boxes, probs, threshold):
  """Non-Maximum supression.
  Args:
    boxes: array of [cx, cy, w, h] (center format)
    probs: array of probabilities
    threshold: two boxes are considered overlapping if their IOU is largher than
        this threshold
    form: 'center' or 'diagonal'
  Returns:
    keep: array of True or False.
  """
 
  order = probs.argsort()[::-1]
  keep = [True]*len(order)

  for i in range(len(order)-1):
    ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
    for j, ov in enumerate(ovps):
      if ov > threshold:
        keep[order[j+i+1]] = False
  return keep

def sliding_window(image):
  for y in range(0, image.shape[0]-127, 16):
      for x in range(0, image.shape[1]-63, 16):
          yield (x,y,image[y: y + 128, x:x + 64])

def apply_gaussian(img, image_id):
    
    img_bboxes = []
    img_confidences = []
    img_image_ids = []
    scale = 0
    downscale = 1.1
    windowSize = (64,128)
    for resized in pyramid_gaussian(img, downscale=1.1,multichannel=True):
        if resized.shape[0]<128 or resized.shape[1]<64:
            break
        resized = img_as_ubyte(resized)
        
        for (x,y,image) in sliding_window(resized):
            pred, confidence = model.get_confidence(image)
            if pred == 1:
                if confidence > 0.9: 
                    img_image_ids.append(image_id)
                    img_confidences.append(confidence)
                    img_bboxes.append((int(x * (downscale**scale)), int(y * (downscale**scale)),
                                        int(windowSize[0]*(downscale**scale)),
                                            int(windowSize[1]*(downscale**scale))))
        scale+=1
    
    img_bboxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in img_bboxes])
    nms_bboxes = np.array([[x+int(w/2), y+int(h/2), w, h] for (x, y, w, h) in img_bboxes])
    img_confidences = np.array(img_confidences)
    img_image_ids = np.array(img_image_ids)
    # print("detection confidence score: ", img_confidences)
    keep = nms(nms_bboxes, img_confidences, 0.4)
    
    return img_bboxes[keep], img_confidences[keep], img_image_ids[keep]


bboxes = []
confidences = []
image_ids = []
test_pos_im_path = './INRIAPerson/INRIAPerson/Test/pos'
test_neg_im_path = './INRIAPerson/INRIAPerson/Test/neg'
label_path = './INRIAPerson/INRIAPerson/Test/annotations/edited/annotation.txt'
with open("cascade.pkl","rb") as f:
    model = pickle.load(f)
for dir in glob.glob(test_pos_im_path+'/*.jpg'):
    img = cv.imread(dir)
    image_id = dir[len(test_pos_im_path)+1:]
    img_bboxes, img_confidences, img_image_ids = apply_gaussian(img, image_id)
    bboxes.extend(img_bboxes)
    confidences.extend(img_confidences)
    image_ids.extend(img_image_ids)
for dir in glob.glob(test_pos_im_path+'/*.png'):
    img = cv.imread(dir)
    image_id = dir[len(test_pos_im_path)+1:]
    img_bboxes, img_confidences, img_image_ids = apply_gaussian(img, image_id)
    bboxes.extend(img_bboxes)
    confidences.extend(img_confidences)
    image_ids.extend(img_image_ids)

bboxes = np.array(bboxes)
confidences = np.array(confidences)
image_ids = np.array(image_ids)
print(bboxes.shape, confidences.shape, image_ids.shape)
gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections = utils.evaluate_detections(bboxes, confidences, image_ids, label_path, True)
utils.visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp,
    test_pos_im_path, label_path, onlytp=False)