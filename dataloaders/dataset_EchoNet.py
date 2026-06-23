import os

import cv2
import numpy as np
from torch.utils.data import Dataset
class BaseDataSets_TN3K(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        data_list = os.listdir(os.path.join(base_dir, split, "images"))

        self.data_list = data_list
        self.data_len = len(data_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        image = cv2.imread(os.path.join(self._base_dir, self.split, "images", data_name), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(self._base_dir, self.split, "labels", data_name), cv2.IMREAD_GRAYSCALE)
        label = (label > 0).astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = data_name
        return sample

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        data_list = os.listdir(os.path.join(base_dir, split, "images"))

        self.data_list = data_list
        self.data_len = len(data_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        image = cv2.imread(os.path.join(self._base_dir, self.split, "images", data_name), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(self._base_dir, self.split, "labels", data_name), cv2.IMREAD_GRAYSCALE)
        label = (label > 0).astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = data_name
        return sample
