from itertools import compress
from sklearn.svm import SVC
from skimage import img_as_ubyte
import numpy as np
import cv2 as cv
import random

class WeakClassifier:
    def __init__(self, size, position):
            # single classifier instance (1 size, position HOG)
            self.size = size
            self.position = position
            self.model = SVC(kernel='linear',probability=True)

    def hog_compute(self, image):
        winSize = self.size
        blockSize = self.size
        blockStride = self.size
        cellSize = (int(self.size[0]/2),int(self.size[1]/2))
        nbins = 9
        hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
        # print(image.shape)
        img = image[self.position[1]:self.position[1]+self.size[1],self.position[0]:self.position[0]+self.size[0]]
        # print(img.shape)
        # print(self.size)
        # print(self.position)
        return hog.compute(img)

    def train(self, pos_im_list, neg_im_list, pos_im_weight, neg_im_weight):
        data = []
        label = []
        im_weight = []
        i = 0
        for image in pos_im_list:
            h = self.hog_compute(image)
            h = h.flatten()
            data.append(h)
            label.append(1)
            im_weight.append(pos_im_weight[i])
            i = i + 1
        i = 0
        for image in neg_im_list:
            h = self.hog_compute(image)
            h = h.flatten()
            data.append(h)
            label.append(0)
            im_weight.append(neg_im_weight[i])
            i = i + 1
        data = np.array(data)
        label = np.array(label)
        im_weight = np.array(im_weight)
        self.model.fit(data, label, sample_weight=im_weight)
    
    def classify(self, image):
        h = self.hog_compute(image)
        h = np.array(h)
        h = h.reshape(-1,36)
        return self.model.predict(h)[0]

    def classify_list(self, img_list):
        result = []
        for image in img_list:
            result.append(self.classify(image))
        result = np.array(result)
        return result


class StrongClassifier:
    def __init__(self, pos_im_list = [], neg_im_list = []):
        self.classifier_stack = []
        self.classifier_weight = []
        self.pos_im_list = pos_im_list
        self.neg_im_list = neg_im_list
        self.threshold = 0.5

    def random_select(self, pos_im_weight, neg_im_weight):
        weak_classifiers = []
        # randomly select 250 HOGs
        # since 250 is too big, select 60
        for _ in range(0,20):
            size_num = random.randrange(1,4)
            if size_num == 1:
                scale = 2 * random.randrange(6,33)
                size = (scale, scale)
            elif size_num ==2:
                scale = 2 * random.randrange(6,33)
                size = (scale, scale*2)
            else:
                scale = 2 * random.randrange(6,17)
                size = (2*scale, scale)
            x = random.randrange(0,64-size[0]+1)
            y = random.randrange(0,128-size[1]+1)
            classifier = WeakClassifier(size=size,position=(x,y))
            print(size,(x,y))
            classifier.train(self.pos_im_list, self.neg_im_list, pos_im_weight, neg_im_weight)
            weak_classifiers.append(classifier)
        print("random_selected")
        return weak_classifiers

    def adaBoost_train(self):
        # implementation of algorithm
        pos_len = len(self.pos_im_list)
        neg_len = len(self.neg_im_list)
        pos_im_weight = np.ones(pos_len)
        neg_im_weight = np.ones(neg_len)
        d_min = 0.9975
        f_max = 0.7
        f = 1
        pos_classifier_score = np.zeros(pos_len)
        neg_classifier_score = np.zeros(neg_len)
        while f > f_max:
            # adaBoost training with weight adjustment
            pos_im_weight = pos_im_weight/np.mean(pos_im_weight)
            neg_im_weight = neg_im_weight/np.mean(neg_im_weight)
            # print(pos_im_weight.shape, neg_im_weight.shape, np.sum(pos_im_weight), np.sum(neg_im_weight))
            weak_classifiers = self.random_select(pos_im_weight, neg_im_weight)
            min_eps = 1
            for classifier in weak_classifiers:
                pos_result = classifier.classify_list(self.pos_im_list)
                neg_result = classifier.classify_list(self.neg_im_list)
                eps = (np.sum(pos_im_weight*(1-pos_result)))/(2*pos_len) + (np.sum(neg_im_weight*neg_result))/(2*neg_len) # 1
                print(eps)
                if min_eps > eps:
                    min_eps = eps
                    beta = eps/(1-eps)
                    best_classifier = classifier
                    best_pos_result = pos_result
                    best_neg_result = neg_result
                    print("classifying loop",np.sum(best_pos_result),np.sum(best_neg_result),beta)
            weight = np.log(1/beta)
            print("weight, results",weight, np.sum(best_pos_result), np.sum(best_neg_result))
            pos_im_weight = pos_im_weight * (beta ** best_pos_result)
            neg_im_weight = neg_im_weight * (beta ** (1-best_neg_result))
            self.classifier_stack.append(best_classifier)
            self.classifier_weight.append(weight)
            pos_classifier_score = pos_classifier_score + weight*best_pos_result
            neg_classifier_score = neg_classifier_score + weight*best_neg_result
            # print("neg_result",neg_classifier_score[0:100,])
            # print("pos_result",pos_classifier_score[0:100,])
            sorted = np.sort(np.array(pos_classifier_score))
            index = int(np.floor((1-d_min)*pos_len))
            self.threshold = sorted[index] / np.sum(np.array(self.classifier_weight))
            print(index, self.threshold)
            d_score = np.sum(np.array(pos_classifier_score >= self.threshold * np.sum(np.array(self.classifier_weight)))) / pos_len
            fp_list = np.array(neg_classifier_score >= self.threshold * np.sum(np.array(self.classifier_weight)))
            f = np.sum(fp_list) / neg_len
            print("f, d_score",f,d_score)
        return d_score, f, fp_list

    def adaBoost_classify(self, image):
        score = 0
        # print("number of weak classifiers",len(self.classifier_stack))
        for i in range(0,len(self.classifier_stack)):
            score = score + self.classifier_stack[i].classify(image) * self.classifier_weight[i]
        return 1 if score >= self.threshold * np.sum(np.array(self.classifier_weight)) else -1

    def adaBoost_confidence(self, image):
        score = 0
        # print("number of weak classifiers",len(self.classifier_stack))
        for i in range(0,len(self.classifier_stack)):
            score = score + self.classifier_stack[i].classify(image) * self.classifier_weight[i]
        class_index = 1 if score >= self.threshold * np.sum(np.array(self.classifier_weight)) else -1
        return class_index, score/len(self.classifier_stack)


class EntireClassifier:
    def __init__(self):
        self.classifier_stack = []
    
    def train(self, pos_im_list, neg_im_list):
        f_target = 0.3
        f_tot = 1
        d_tot = 1
        pos_ims = pos_im_list
        neg_ims = neg_im_list
        while f_tot > f_target:
            classifier = StrongClassifier(pos_ims, neg_ims)
            d_score, f, fp_list = classifier.adaBoost_train()
            self.classifier_stack.append(classifier)
            f_tot = f_tot * f
            d_tot = d_tot * d_score
            neg_ims = list(compress(neg_ims,fp_list))
            print(f_tot, d_tot)
        return f_tot, d_tot

    def classify(self, image):
        for classifier in self.classifier_stack:
            if classifier.adaBoost_classify(image) == -1:
                return -1
        return 1

    def get_confidence(self, image):
        score = 0
        class_index = 1
        for classifier in self.classifier_stack:
            class_index, score_val = classifier.adaBoost_confidence(image)
            score = score + score_val
            if class_index == -1:
                return -1, 0
        score = score / len(self.classifier_stack)
        return class_index, score

    
    def classify_list(self, image_list):
        pred_list = []
        i=0
        for image in image_list:
            # print("image",i)
            i = i + 1
            pred_list.append(self.classify(image))
        return pred_list