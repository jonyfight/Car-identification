import multiprocessing as mp
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

carnames = [w.strip() for w in classes]

# Training proportion
split_proportion = 0.8


def move_picture(car):
        if car not in os.listdir(file_train):
            os.mkdir(os.path.join(file_train, car))

        total_images = os.listdir(os.path.join(file_total, car))

        nbr_train = int(len(total_images) * split_proportion)

        np.random.shuffle(total_images)

        train_images = total_images[:nbr_train]

        val_images = total_images[nbr_train:]

        for img in tqdm(train_images):
            source = os.path.join(file_total, car, img)
            target = os.path.join(file_train, car, img)
            shutil.copy(source, target)

        if car not in os.listdir(file_val):
            os.mkdir(os.path.join(file_val, car))

        for img in val_images:
            source = os.path.join(file_total, car, img)
            target = os.path.join(file_val, car, img)
            shutil.copy(source, target)
        print('%s is finished !' % car)


def main():
    core = 8
    pool = mp.Pool(core)
    for car in carnames:
        pool.apply_async(move_picture, (car,))
    pool.close()
    pool.join()
    print('Finish splitting train and val images!')


if __name__ == '__main__':
    main()