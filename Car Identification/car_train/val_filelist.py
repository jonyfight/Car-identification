# -*- coding: utf-8 -*-
#  提取当前目录下所有文件及路径
import os
#import cv2

if __name__ == "__main__":

    file_train = '/data0/data/val_split'
    f = open("./val_file.txt", "w")
    files = os.listdir(file_train)

    for subfile in files:
        num_img = len(os.listdir(os.path.join(file_train, subfile)))
        print("{} images of {}".format(num_img, subfile))
        for file in os.listdir(os.path.join(file_train, subfile)):
            if file.endswith(('.jpg','.png','.jpeg','.JPG','.PNG','.JPEG')):
                f.writelines(os.path.join(file_train, subfile, file) + " " + subfile +'\n')



    f.close()

