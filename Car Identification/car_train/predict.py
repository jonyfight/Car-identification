# -*- coding:utf-8 -*-
import cv2
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
nbr_test_samples = 1000

def center_crop(x, center_crop_size):

    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw: centerw + halfw,
              centerh - halfh: centerh + halfh, :]

    return cropped

# 读取图片,并得到固定大小
def scale_byRatio(img_path, ratio=1.0, return_width=299, crop_method=center_crop):
    # Given an image path, return a scaled array

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    shorter = min(w, h)

    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(img_cropped, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb


with open('./class_file.txt') as f:
    classes = f.readlines()

carnames = [w.strip() for w in classes]


weights_path = "./log/InceptionV3_best_vehicleModel.h5"

#test_data_dir = "....txt"



print('Loading model and weights from training process ...')

with tf.device("/cpu:0"):
    inception = InceptionV3(include_top=False, weights = None,
                            input_tensor=None, input_shape=(img_width, img_height, 3), pooling='avg')
    output = inception.get_layer(index=-1).output  # shape=(None, 1, 1, 2048)
    output = Dense(1024, name='features')(output)
    output = Dense(num_class, activation='softmax', name='predictions')(output)
    model1 = Model(outputs=output, inputs=inception.input)

model = multi_gpu_model(model1, gpus=2)
model.load_weights(weights_path, by_name=True)
model1.save("./log/one_gpu_model.h5")
print('Begin to predict for testing data ...')

if batch_size > 1:
    pass
    
else:
    image_path = '1.png'
    img = scale_byRatio(image_path, return_width=img_width)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    index = np.argmax(predictions)
    print(predictions[0,index])    
    print(carnames[index])




