# -*- coding: utf-8 -*-

from math import ceil
import numpy as np
from inceptionv3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense,Dropout
from keras.models import Model
from keras.models import load_model
from keras.optimizers import *
from utils import SGDRScheduler, CustomModelCheckpoint, SequenceData
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import TensorBoard

import os
#from sparse import *
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

np.random.seed(1024)

FINE_TUNE = False
LEARNING_RATE = 0.1
NBR_EPOCHS = 100
BATCH_SIZE = 16
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'val_acc'
NBR_MODELS = 106
USE_CLASS_WEIGHTS = False
RANDOM_SCALE = True
nbr_gpus = 2

train_path = './train_file.txt'
val_path = './val_file.txt'

if __name__ == "__main__":

    K.clear_session()

    if FINE_TUNE:
        print('Finetune and Loading InceptionV3 Weights ...')
        weights_path = "./log/0.93_freeze100_InceptionV3_best_vehicleModel.h5"
        with tf.device("/cpu:0"):
            inception = InceptionV3(include_top=False, weights="imagenet",
                                    input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='max')

            output = inception.get_layer(index=-1).output  # shape=(None, 1, 1, 2048)
            # output = Dropout(0.3)(output)
            # output = Dense(1024, name='features')(output)
            output = Dropout(0.5)(output)
            # output = AMSoftmax(NBR_MODELS, 30, 0.35)(output)
            output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
            model = Model(outputs=output, inputs=inception.input)
            for layer in inception.layers[:100]:
                layer.trainable = False

        model = multi_gpu_model(model, gpus=2)
        model.load_weights(weights_path, by_name=True)

    else:
        with tf.device("/cpu:0"):
            print('Loading InceptionV3 Weights ...')
            inception = InceptionV3(include_top=False, weights= None,
                   input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='max')

            output = inception.get_layer(index=-1).output     # shape=(None, 1, 1, 2048)
            #output = Dropout(0.3)(output)
            #output = Dense(1024, name='features')(output)
            output = Dropout(0.5)(output)
            #output = AMSoftmax(NBR_MODELS, 30, 0.35)(output)
            output = Dense(NBR_MODELS, activation='softmax', name='predictions')(output)
            model = Model(outputs=output, inputs=inception.input)

        model = multi_gpu_model(model, gpus=nbr_gpus)
    print('Training model begins...')
    # 多GPU编译环境

    BATCH_SIZE *= nbr_gpus

    # 优化器的选择
    #optimizer = SGD(lr=LEARNING_RATE, momentum=0.9, decay=0.0, nesterov=True)
    optimizer = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    #model.compile(loss=amsoftmax_loss, optimizer=optimizer, metrics=["accuracy"])

    # autosave best Model
    best_model_file = "./log/InceptionV3_best_vehicleModel.h5"
    # Define several callbacks
    #loging = TensorBoard(log_dir='./log')


    # 存储多GPU模型只能保存权重,不能保存结构;可以在训练完成后,重新保存,详见mutigpu_to_cpu.py
    best_model = CustomModelCheckpoint(model, best_model_file, monitor_index=monitor_index)
    reduce_lr = ReduceLROnPlateau(monitor=monitor_index, factor=0.5, patience=5, verbose=1, min_lr=0.0001)
    early_stop = EarlyStopping(monitor=monitor_index, patience=50, verbose=1)

    # 准备数据
    train_data_lines = open(train_path, 'r', encoding='utf-8').readlines()
    # Check if image path exists.
    train_data_lines = [w.strip() for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train / BATCH_SIZE))

    val_data_lines = open(val_path, 'r', encoding='utf-8').readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val / BATCH_SIZE))

    #train_data_lines = train_data_lines + val_data_lines
    #nbr_train = len(train_data_lines)
    #print('# Train Images: {}.'.format(nbr_train))
    #steps_per_epoch = int(ceil(nbr_train / BATCH_SIZE))

    #sgdr = SGDRScheduler(epochsize=nbr_train, batchsize=BATCH_SIZE)


    # 开始训练
    model.fit_generator(SequenceData(train_data_lines, nbr_classes=NBR_MODELS,
                        batch_size=BATCH_SIZE, img_width=IMG_WIDTH,
                        img_height=IMG_HEIGHT, random_scale=RANDOM_SCALE,
                        shuffle=True, augment=True),
                        steps_per_epoch=steps_per_epoch, epochs=NBR_EPOCHS, verbose=1,
                        validation_data=SequenceData(val_data_lines, nbr_classes=NBR_MODELS,
                        batch_size=BATCH_SIZE, img_width=IMG_WIDTH,
                        img_height=IMG_HEIGHT, random_scale=RANDOM_SCALE,
                        shuffle=True, augment=False),
                        validation_steps=validation_steps,
                        callbacks=[best_model, early_stop, TensorBoard(log_dir='./tensorboard_log')],
                        max_queue_size=128, workers=8, use_multiprocessing=True, class_weight='auto')