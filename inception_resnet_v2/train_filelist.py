# -*- coding: utf-8 -*-
#  提取当前目录下所有文件及路径
import os
#import cv2

if __name__ == "__main__":
    file_train = '/data0/data/train_split'
    f = open("./train_file.txt", "w")

    with open('./class_file.txt') as f2:
        classes = f2.readlines()

    files = [w.strip() for w in classes]

    num_img = 0
    for subfile in files:
        num_img = len(os.listdir(os.path.join(file_train, subfile)))
        print("{} images of {}".format(num_img, subfile))
        for file in os.listdir(os.path.join(file_train, subfile)):
            if file.endswith(('.jpg','.png','.jpeg','.JPG','.PNG','.JPEG')):
                f.writelines(os.path.join(file_train, subfile, file) + " " + subfile +'\n')



    f.close()

