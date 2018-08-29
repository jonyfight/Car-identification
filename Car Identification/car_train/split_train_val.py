import os
import numpy as np
import shutil
from tqdm import tqdm

np.random.seed(2018)

file_train = '/data0/data/train_split'
file_val = '/data0/data/val_split'

file_total = '/data0/data/picture'

with open('./class_file.txt') as f:
    classes = f.readlines()

CarNames = [w.strip() for w in classes]

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8

for car in tqdm(CarNames):

    if car not in os.listdir(file_train):
        os.mkdir(os.path.join(file_train, car))

    total_images = os.listdir(os.path.join(file_total, car))

    nbr_train = int(len(total_images) * split_proportion)

    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]

    val_images = total_images[nbr_train:]

    for img in train_images:
        source = os.path.join(file_total, car, img)
        target = os.path.join(file_train, car, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    if car not in os.listdir(file_val):
        os.mkdir(os.path.join(file_val, car))

    for img in val_images:
        source = os.path.join(file_total, car, img)
        target = os.path.join(file_val, car, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))


