# -*- coding:utf-8 -*-
import os

if __name__ == "__main__":

    file_test = '/data0/data/test/22363'
    f = open("./tmp.txt", "w", encoding='gbk')
    files = os.listdir(file_test)
    num_img = 0
    for file in files:
            file2 = file.encode('gbk','ignore')
            file2 = file2.decode('gbk')
            os.rename(file, file2)
            if file.endswith(('.jpg','.png','.jpeg','.JPG','.PNG','.JPEG')):
                   f.writelines(os.path.join(file_test, file2) + " " + '\n')
    f.close()