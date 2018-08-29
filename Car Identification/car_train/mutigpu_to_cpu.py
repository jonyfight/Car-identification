# -*- coding:utf-8 -*-
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
from keras.models import load_model

img_width = 299
img_height = 299
batch_size = 1
num_class = 106

weights_path = "./log/0.93_freeze100_InceptionV3_best_vehicleModel.h5"

print('Loading model and weights from training process ...')

with tf.device("/cpu:0"):
    inception = InceptionV3(include_top=False, weights='imagenet',
                            input_tensor=None, input_shape=(img_width, img_height, 3), pooling='avg')
    output = inception.get_layer(index=-1).output  
    #output = Dense(1024, name='features')(output)
    output = Dense(num_class, activation='softmax', name='predictions')(output)
    model1 = Model(outputs=output, inputs=inception.input)
    for layer in inception.layers[:100]:
        layer.trainable = False
    # 本地运行时使用,注意变量名
    #model1 = load_model(weights_path)

# 变换多GPU模型为cpu模型,本地可运行; 关键位使用muti_gpu_model()的参数model
model = multi_gpu_model(model1, gpus=2)
model.load_weights(weights_path, by_name=True)
model1.save("./log/one_gpu_model.h5")

print('Begin to predict for testing data ...')




