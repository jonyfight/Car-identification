import os
import multiprocessing as mp
import numpy as np
import shutil
from random import sample
from tqdm import tqdm
np.random.seed(2018)

file_test = '/data0/data/test'
file_total = '/data0/testSet3'

with open('./class_file.txt') as f:
    classes = f.readlines()

carnames = [w.strip() for w in classes]


for car in tqdm(carnames):
    if car not in os.listdir(file_test):
        os.mkdir(os.path.join(file_test, car))

    total_images = os.listdir(os.path.join(file_total, car))
    num = min(1000, len(total_images))
    test_images = sample(total_images, num)

    for img in test_images:
        source = os.path.join(file_total, car, img)
        target = os.path.join(file_test, car, img)
        shutil.copy(source, target)



print('Finish test images!')

