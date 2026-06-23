import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomChoice, RandomEqualize, ColorJitter

from dataloaders.dataset import random_rot_flip, random_rotate, color_jitter


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
        image = cv2.imread(os.path.join(self._base_dir, self.split, "images", data_name))
        label = cv2.imread(os.path.join(self._base_dir, self.split, "labels", data_name), cv2.IMREAD_GRAYSCALE)

        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = data_name
        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.trans = RandomChoice([
            RandomEqualize(0.5),
            ColorJitter(brightness=0.5),
            ColorJitter(contrast=0.5),
            ColorJitter(saturation=0.5),
            ColorJitter(hue=0.5)
        ])

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = self.trans(image)
        image = image / 255
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        image = image / 255
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).permute(2, 0, 1)
        image_weak = image_weak / 255
        label = torch.from_numpy(label.astype(np.uint8))
        image_strong = image_strong / 255

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample
