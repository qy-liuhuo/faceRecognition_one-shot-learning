import os
import numpy as np
import tensorflow as tf
from nets.vgg import VGG16
from utils.utils import letterbox_image, preprocess_input, cvtColor, show_config
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Lambda,
                                     MaxPooling2D)
from tensorflow.keras.models import Model


# 图片特征提取器
class FeatureExtractor (object):
    _defaults = {
        #-----------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path
        #   model_path指向logs文件夹下的权值文件
        #-----------------------------------------------------#
        "model_path"        : './weights/weights1.h5',
        #-----------------------------------------------------#
        #   输入图片的大小。
        #-----------------------------------------------------#
        "input_shape"       : [64, 64],
        #--------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
        #   否则对图像进行CenterCrop
        #--------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    # 初始化模型
    def __init__(self, **kwargs):
        self.model = None
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()
        show_config(**self._defaults)

    #   载入模型
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        vgg_model = VGG16()
        input_image = Input(shape=[self.input_shape[0], self.input_shape[1], 3])
        encoded_image = vgg_model.call(input_image)
        self.model = Model(input_image, encoded_image)
        self.model.load_weights(self.model_path) # 加载权重
        #print('{} model loaded.'.format(model_path))


    #   提取图片特征
    def extract(self, image):

        #   对输入图像进行不失真的resize
        image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)

        #   归一化+添加上batch_size维度
        photo= np.expand_dims(preprocess_input(np.array(image, np.float32)), 0)

        #   获得特征值
        output = np.array(self.model(photo, training=False))

        return output
