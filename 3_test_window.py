import joblib
from skimage.transform import pyramid_gaussian
import utils
import numpy as np
import glob
from skimage.feature import local_binary_pattern
import cv2 as cv
from skimage import img_as_ubyte

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
    for resized in pyramid_gaussian(img, downscale=1.1):
        if resized.shape[0]<128 or resized.shape[1]<64:
            break
        resized = img_as_ubyte(resized)
        
        for (x,y,image) in sliding_window(resized):
          hist = get_lbp_data(image)
          hist = hist.reshape(1,-1)
          pred = model.predict(hist)
          if pred == 1:
            confidence = model.predict_proba(hist)[0]
            if confidence[1]-confidence[0] > 0.95: 
              img_image_ids.append(image_id)
              img_confidences.append(confidence[1]-confidence[0])
              img_bboxes.append((int(x * (downscale**scale)), int(y * (downscale**scale)),
                                  int(windowSize[0]*(downscale**scale)),
                                      int(windowSize[1]*(downscale**scale))))
        scale+=1
    
    img_bboxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in img_bboxes])
    nms_bboxes = np.array([[x+int(w/2), y+int(h/2), w, h] for (x, y, w, h) in img_bboxes])
    img_confidences = np.array(img_confidences)
    img_image_ids = np.array(img_image_ids)
    # print("detection confidence score: ", img_confidences)
    keep = nms(nms_bboxes, img_confidences, 0.35)
    
    return img_bboxes[keep], img_confidences[keep], img_image_ids[keep]

bboxes = []
confidences = []
image_ids = []
test_pos_im_path = './INRIAPerson/INRIAPerson/Test/pos'
test_neg_im_path = './INRIAPerson/INRIAPerson/Test/neg'
label_path = './INRIAPerson/INRIAPerson/Test/annotations/edited/annotation.txt'
model = joblib.load('lbp_model.npy')
for dir in glob.glob(test_pos_im_path+'/*.jpg'):
    img = cv.imread(dir)
    image_id = dir[len(test_pos_im_path)+1:]
    print(image_id)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_bboxes, img_confidences, img_image_ids = apply_gaussian(img, image_id)
    bboxes.extend(img_bboxes)
    confidences.extend(img_confidences)
    image_ids.extend(img_image_ids)
for dir in glob.glob(test_pos_im_path+'/*.png'):
    img = cv.imread(dir)
    image_id = dir[len(test_pos_im_path)+1:]
    print(image_id)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
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