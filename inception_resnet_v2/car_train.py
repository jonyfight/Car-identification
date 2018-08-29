# -*- coding: utf-8 -*-

from math import ceil
import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense,Dropout
from keras.models import Model
from keras.models import load_model
from keras.optimizers import *
from utils import CustomModelCheckpoint, SequenceData
from keras.utils.training_utils import multi_gpu_model
from keras import regularizers
import os
#from sparse import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
np.random.seed(1024)

FINE_TUNE = True
USE_PROCESSING =False
NEW_OPTIMIZER = True
LEARNING_RATE = 0.001
NBR_EPOCHS = 100
BATCH_SIZE = 32
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'acc'
NBR_MODELS = 106
USE_CLASS_WEIGHTS = False
RANDOM_SCALE = True
CLASS_WARE = False
nbr_gpus = 2

train_path = './train_file.txt'
val_path = './val_file.txt'


if __name__ == "__main__":
    K.clear_session()

    if FINE_TUNE:
        print('Finetune and Loading InceptionResNetV2 Weights ...')

        weights_path = "./log/1_InceptionResNetV2_best_vehicleModel.h5"
        with tf.device("/cpu:0"):
            print('Loading InceptionResNetV2 Weights ...')
            inception = InceptionResNetV2(include_top=False, weights=None,
                                          input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='avg')

            output = inception.get_layer(index=-1).output  # shape=(None, 1, 1, 2048)
            output = Dropout(0.5)(output)
            output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
            model = Model(outputs=output, inputs=inception.input)
            for layer in inception.layers[:600]:
                layer.trainable = False

        model = multi_gpu_model(model, gpus=2)
        model.load_weights(weights_path, by_name=True)

    else:
        with tf.device("/cpu:0"):
            print('Loading InceptionResNetV2 Weights ...')
            inception = InceptionResNetV2(include_top=False, weights="imagenet",
                   input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='avg')

            output = inception.get_layer(index=-1).output
            output = Dropout(0.5)(output)
            output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
            model = Model(outputs=output, inputs=inception.input)
            for layer in inception.layers[:500]:
                layer.trainable = False

        model = multi_gpu_model(model, gpus=nbr_gpus)

    print('Training model begins...')
    # 多GPU编译环境

    BATCH_SIZE *= nbr_gpus

    # 优化器的选择
    optimizer = SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)
    #optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #optimizer = Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    #model.compile(loss=amsoftmax_loss, optimizer=optimizer, metrics=["accuracy"])

    #inception.summary()
    print(len(inception.layers))


    # autosave best Model
    best_model_file = "./log/best_model.h5"
    # Define several callbacks
    #loging = TensorBoard(log_dir='./log')

    # 存储多GPU模型只能保存权重,不能保存结构;可以在训练完成后,重新保存,详见mutigpu_to_cpu.py
    best_model = CustomModelCheckpoint(model, best_model_file)

    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=10, verbose=1, min_lr=0.000001)

    early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=30, verbose=1)

    # 准备数据
    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w.strip() for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train / BATCH_SIZE))

    val_data_lines = open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val / BATCH_SIZE))

    # 开始训练
    model.fit_generator(SequenceData(train_data_lines, nbr_classes=NBR_MODELS,
                        batch_size=BATCH_SIZE, img_width=IMG_WIDTH,
                        img_height=IMG_HEIGHT, random_scale=RANDOM_SCALE,
                        shuffle=True, augment=True),
                        steps_per_epoch=steps_per_epoch, epochs=NBR_EPOCHS, verbose=1,
                        validation_data=SequenceData(val_data_lines,
                        nbr_classes=NBR_MODELS, batch_size=BATCH_SIZE,
                        img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
                        shuffle=False, augment=False),
                        validation_steps=validation_steps,
                        callbacks=[best_model, early_stop, reduce_lr],
                        max_queue_size=128, workers=8, use_multiprocessing=True, class_weight='auto')
