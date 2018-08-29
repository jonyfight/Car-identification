# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense,Dropout
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model
from tqdm import tqdm
import traceback
import time


img_width = 299
img_height = 299
batch_size = 1
num_class = 106

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


weights_path = "./log/best_model.h5"
test_path = './test_file.txt'

print('Loading model and weights from training process ...')
print(weights_path)

with tf.device("/cpu:0"):
    base = InceptionResNetV2(include_top=False, weights='imagenet',
                       input_tensor=None, input_shape=(img_width, img_height, 3), pooling='avg')
    output = base.get_layer(index=-1).output

    output = Dense(num_class, activation='softmax', name='predictions')(output)
    model1 = Model(outputs=output, inputs=base.input)

    for layer in base.layers[:600]:
        layer.trainable = False

model = multi_gpu_model(model1, gpus=2)
model.load_weights(weights_path, by_name=True)

print('save model cpu ........')
model1.save('./one_cpu.h5')
print('Begin to predict for testing data ...')

test_data_lines = open(test_path).readlines()
test_data_lines = [w.strip()for w in test_data_lines if os.path.exists(w.strip().split(' ')[0])]


list_file = open('./new.txt','w')
time_file = open('./time.txt','w')



for annline in tqdm(test_data_lines):
    try:
        line = annline.strip().split(' ')
        label = str(line[-1])
        if len(line) == 2:
            img_path = line[0]
        else:
            img_path = ''.join(line[:-1])

        start = time.time()

        #img = image.load_img(img_path, target_size=(img_width, img_height))
        img = scale_byRatio(img_path,return_width=img_width)
        x = np.array(img)

        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x)
        index = np.argmax(predictions)

        list_file.writelines(label + ';' +carnames[index] + ';' + \
                             str(predictions[0, index]) + ';' + img_path + '\n')
        end = time.time()
        t = end - start
        time_file.writelines(str(t) + 'ms' + '\n')

    except:
        print(img_path)
        traceback.print_exc()
        continue

list_file.close()
time_file.close()



