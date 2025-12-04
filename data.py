import os
import random

import cv2
import natsort
import numpy as np
import torch
from torch.utils.data import Dataset


class MyTrainDataSet(Dataset):
    def __init__(self, dataset, patch_size=256):
        super(MyTrainDataSet, self).__init__()

        self.hazyPath = os.path.join(dataset, 'train/hazy')
        self.clearPath = os.path.join(dataset, 'train/clear')

        self.inputImages = natsort.natsorted(os.listdir(self.hazyPath), alg=natsort.ns.PATH)
        self.targetImages = natsort.natsorted(os.listdir(self.clearPath), alg=natsort.ns.PATH)

        self.patch_size = patch_size

    def __len__(self):
        return len(self.inputImages)

    def augData(self, haze, clear, patch_size):


        height, width, _ = haze.shape
        if height < patch_size or width < patch_size:
            padding_height = max(0, patch_size - height)
            padding_width = max(0, patch_size - width)

            padding = ((0, padding_height), (0, padding_width), (0, 0))

            haze = np.pad(haze, padding, mode='constant', constant_values=0)
            clear = np.pad(clear, padding, mode='constant', constant_values=0)


        height, width, _ = haze.shape
        x1, y1 = random.randint(0, width - patch_size), random.randint(0, height - patch_size)
        haze = haze[y1:y1 + patch_size, x1:x1 + patch_size, :]
        clear = clear[y1:y1 + patch_size, x1:x1 + patch_size, :]


        rand_hor = random.randint(0, 1)
        if rand_hor:
            haze = np.fliplr(haze)
            clear = np.fliplr(clear)


        rand_rot = random.randint(0, 3)
        haze = np.rot90(haze, k=rand_rot)
        clear = np.rot90(clear, k=rand_rot)

        haze = np.ascontiguousarray(haze)
        clear = np.ascontiguousarray(clear)

        return haze, clear

    def __getitem__(self, index):

        input_path = os.path.join(self.hazyPath, self.inputImages[index])
        target_path = os.path.join(self.clearPath, self.targetImages[index])

        load_input = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        load_target = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)

        input_image, target_image = self.augData(load_input, load_target, self.patch_size)

        input_image = torch.tensor(input_image).float().permute(2, 0, 1) / 255
        target_image = torch.tensor(target_image).float().permute(2, 0, 1) / 255

        return input_image, target_image


class MyValueDataSet(Dataset):
    def __init__(self, dataset):
        super(MyValueDataSet, self).__init__()

        self.inputPath = os.path.join(dataset, 'test/hazy')
        self.targetPath = os.path.join(dataset, 'test/clear')

        self.inputImages = natsort.natsorted(os.listdir(self.inputPath), alg=natsort.ns.PATH)
        self.targetImages = natsort.natsorted(os.listdir(self.targetPath), alg=natsort.ns.PATH)

    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):

        input_path = os.path.join(self.inputPath, self.inputImages[index])
        target_path = os.path.join(self.targetPath, self.targetImages[index])

        img_name = self.inputImages[index]

        load_input = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        load_target = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)


        height, width, _ = load_input.shape
        new_height = (height // 16) * 16
        new_width = (width // 16) * 16

        if new_height != height or new_width != width:
            load_input = load_input[:new_height, :new_width, :]
            load_target = load_target[:new_height, :new_width, :]

        input_image = torch.tensor(load_input).float().permute(2, 0, 1) / 255
        target_image = torch.tensor(load_target).float().permute(2, 0, 1) / 255

        return input_image, target_image, img_name


if __name__ == "__main__":
    dataset_train = MyTrainDataSet("../datasets/ITS", patch_size=256)
    dataset_test = MyValueDataSet("../datasets/ITS")

    for input, target, img_name in dataset_test:
        print(img_name)