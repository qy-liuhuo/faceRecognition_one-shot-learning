import numpy as np
import tensorflow as tf
from PIL import Image
from nets.FeatureExtractor import FeatureExtractor
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Lambda,
                                     MaxPooling2D)
from tensorflow.keras import *
import cv2
import os

class Recognizer:

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.data=[]
        self.extractData()
        self.recognizer= tf.keras.models.Sequential()
        self.recognizer.add(Dense(512, activation='relu', input_shape=(None, 2048)))
        self.recognizer.add(Dense(1, activation='sigmoid'))
        self.recognizer.build()
        self.recognizer.load_weights("./weights/weights2.h5")

    def extractData(self):
       path = "./faceGallery"  # 照片目录
       files = os.listdir(path)  # 得到文件夹下的所有文件名称
       for file in files:  # 遍历文件夹
           if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
               value=self.extractor.extract(Image.open(path + "/" + file))
               name=file.split(".")[0]
               self.data.append((name,value))
       print("The face image library is loaded.")

    def calculateSimilarity(self,data1,data2):
        X = tf.cast(tf.constant(abs(data1 - data2)), dtype=tf.float32)
        X = tf.expand_dims(X, axis=0)
        result=np.array(self.recognizer(X))[0][0][0]
        return result

    def predict(self,image):
        imageValue=self.extractor.extract(image)
        result=('other',0.8)
        for item in self.data:
            similarity=self.calculateSimilarity(imageValue,item[1])
            if similarity>result[1]:
                result=(item[0],similarity)

        return result

