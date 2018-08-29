# -*- coding: utf-8 -*-

import os


if __name__ == "__main__":

    file_train = '/data0/data/picture'
    f = open("./class_file.txt", "w")
    files = os.listdir(file_train)
    print(len(files))
    num_img = 0
    for subfile in files:
         f.writelines(subfile + '\n')

    f.close()

