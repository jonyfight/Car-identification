# -*- coding: utf-8 -*-
#  提取当前目录下所有文件及路径
import os
from random import sample
import shutil
import random
from tqdm import tqdm

if __name__ == "__main__":
    dir1 = "/data0/testSet3"
    dir2 = "/data0/data/picture"

    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    file = list(set(files1).difference(set(files2)))

    for subfile in tqdm(file):
        datalist = []
        num = 2
        for file in os.listdir(os.path.join(dir1, subfile)):
                datalist.append(os.path.join(subfile, file))

        num = min(2, len(datalist))
        slice = sample(datalist, num)
        for i in range(num):
            source = os.path.join(dir1, slice[i])
            target = '/data0/data/picture/0'
            try:
                shutil.move(source, target)
            except:
                continue