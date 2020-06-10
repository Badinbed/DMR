# -*- coding: utf-8 -*-
__author__ = '凌霄一笑'

import cv2
import numpy as np

from numpy.linalg import norm
import sys
import os
import json
import math
from scipy import misc, ndimage

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 1500  # 表盘区域允许最大面积

def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def hough_change(img):
    
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    
    #霍夫变换
    lines = cv2.HoughLines(edges,1,np.pi/180,0)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
    if x1 == x2 or y1 == y2:
        rotate_img=img
    else:
        t = float(y2-y1)/(x2-x1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        rotate_img = ndimage.rotate(img, rotate_angle)
    return rotate_img        
        
#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    up_point = -1#上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i,x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks

#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        
    # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        
        samples.append(hist)
    return np.float32(samples)
    
class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        pass

    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        # 识别数字
        self.model = SVM(C=1, gamma=0.5)

        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("train\\chars"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = os.path.basename(root)
                for filename in files:
                    filepath = os.path.join(root,filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
            #chars_label.append(1)
                    chars_label.append(root_int)
            
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
        #chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train, chars_label)



    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
            
    def predict(self, car_pic):
        """
        :param meter_pic_file: 图像文件
        :return:已经处理好的图像文件 原图像文件
        """
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]

        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight*resize_rate)), interpolation=cv2.INTER_AREA)
        

        blur = 3
        img = cv2.GaussianBlur(img, (blur, blur), 0)
        oldimg = img
        img = cv2.cvtColor(oldimg, cv2.COLOR_BGR2GRAY)
        #img_edg=cv2.Canny(img, 100, 200)
 


        ret, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, img_thresh1 = cv2.threshold(img_thresh , 127, 255, cv2.THRESH_BINARY_INV)
        
        img_hough=hough_change(img_thresh1)
        ret, img_thresh2 = cv2.threshold(img_hough , 127, 255, cv2.THRESH_BINARY)

        Matrix = np.ones((5, 3), np.uint8)
    
        img_edge1 = cv2.morphologyEx(img_thresh2, cv2.MORPH_CLOSE, Matrix)  
        
        Matrix = np.ones((5, 20), np.uint8)
    
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        #img_edge1 = cv2.morphologyEx(img_edge2, cv2.MORPH_CLOSE, Matrix)
        
        contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]




        #print(len(contours))
        #oldimg = cv2.drawContours(oldimg, contours, -1, (0, 0, 255), 3)
        #cv2.imshow("edge4", oldimg)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            ration = w / h

            if 2.5 < ration < 4:
                meter_img=img_hough[y+2:y+h-2,x+2:x+w-2]
                img_thresh3 = img_thresh2[y+2:y+h-2,x+2:x+w-2]
        #print(len(car_contours))
   

        predict_result = []
      
    
        
        #img_gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
 
        ret, img_thresh4 = cv2.threshold(img_thresh3 , 127, 255, cv2.THRESH_BINARY_INV)
        #ret, img_thresh4 = cv2.threshold(img_thresh3 , 127, 255, cv2.THRESH_BINARY)
        #cv2.imshow("edge4", img_thresh4)
    
        x_histogram = np.sum( img_thresh4, axis=1)

        x_min = np.min(x_histogram)
        #print(x_min)
        x_average = np.sum(x_histogram) / x_histogram.shape[0]
        #print(x_average)
        x_threshold = (x_min + x_average) / 3
        wave_peaks = find_waves(x_threshold, x_histogram)
        if len(wave_peaks) == 0:
            print("peak less 0:")

    #认为水平方向，最大的波峰数字区域
        wave = max(wave_peaks, key=lambda x:x[1]-x[0])
        gray_img = img_thresh4[wave[0]:wave[1]]
        #cv2.imshow("edge4", gray_img)
        y_histogram = np.sum(gray_img, axis=0)
        y_min = np.min(y_histogram)
        y_average = np.sum(y_histogram)/y_histogram.shape[0]
        y_threshold = (y_min + y_average)/2#U和0要求阈值偏小，否则U和0会被分成两半

        wave_peaks = find_waves(y_threshold, y_histogram)

        #print(wave_peaks)
        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        max_wave_dis = wave[1] - wave[0]
        # 判断是否是左侧示数边缘
        if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
           wave_peaks.pop(0)
        cur_dis = 0
        for i, wave in enumerate(wave_peaks):
            if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                break
            else:
                cur_dis += wave[1] - wave[0]
        if i > 0:
            wave = (wave_peaks[0][0], wave_peaks[i][1])
            wave_peaks = wave_peaks[i + 1:]
            wave_peaks.insert(0, wave)
        part_cards = seperate_card(gray_img, wave_peaks)
        #print(len(part_cards))

        #cv2.imshow('part',part_cards[3])
        for i, part_card in enumerate(part_cards):
            #可能是固定车牌的铆钉
            if np.mean(part_card) < 255/5:
                print("a point")
                continue
            part_card_old = part_card
            w = int((part_card.shape[0] - part_card.shape[1])/2)
                    
            part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
            part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    
            #part_card = deskew(part_card)
            #kernel = np.ones((2,2),np.uint8)  
            #part_card = cv2.erode(part_card,kernel,iterations = 1)
             
            part_card = preprocess_hog([part_card])
              
            resp = self.model.predict(part_card)
            charactor = chr(resp[0])
            # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
            if charactor == "1" and i == len(part_cards) - 1:
                if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                    continue
            predict_result.append(charactor)

        
                
      

        return predict_result, img_thresh4 # 识别到的字符、定位的车牌图像

if __name__ == '__main__':
    c = CardPredictor()
    c.train_svm()
    r, roi = c.predict("meter5.jpg")
    cv2.imshow('meter',roi)
    print(r)


