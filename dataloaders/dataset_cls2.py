import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from torchvision.transforms import RandomChoice, RandomEqualize, ColorJitter

import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image

def safe_list_images(folder, valid_ext=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
    """返回 folder 下所有非隐藏、扩展名合法的图片绝对路径"""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if not f.startswith('.') and f.lower().endswith(valid_ext)
    ])
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        # data_list = os.listdir(os.path.join(base_dir, split, "images"))
        img_dir = os.path.join(self._base_dir, self.split, "images")
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')  #
        self.data_list = sorted([
        os.path.basename(f) for f in glob(os.path.join(img_dir, "*"))
        if not f.startswith('.') and f.lower().endswith(valid_ext)
        ])
        # self.data_list = data_list
        self.data_len = len(self.data_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_name = self.data_list[idx]
        
        image = cv2.imread(os.path.join(self._base_dir, self.split, "images", data_name), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(self._base_dir, self.split, "labels", data_name), cv2.IMREAD_GRAYSCALE)
        # 在 __getitem__ 里，cv2.imread 之后、transform 之前加：
        # gray2class = np.zeros(256, dtype=np.uint8)
        # gray2class[0]   = 0  # 背景
        # gray2class[85]  = 1  # 腔
        # gray2class[170] = 2  # 心肌
        # gray2class[255] = 3  # 心房
        label =label/255          # 0-3
        # gray2class = np.zeros(256, dtype=np.uint8)
        # gray2class[0]   = 0  # 背景
        # gray2class[85]  = 1  # 腔
        # gray2class[170] = 2  # 心肌
        # gray2class[255] = 3  # 心房
        # # 其余灰度全部当背景
        # label = gray2class[label]    

        if image is None or label is None:
            img_path=os.path.join(self._base_dir, self.split, "images", data_name)
            lab_path=os.path.join(self._base_dir, self.split, "labels", data_name)
        

            raise FileNotFoundError(f'cv2 cannot read image: {img_path} or {lab_path}')
        # print(label.min(), label.max(), np.unique(label))
     # ---------- 新增 ----------
        # 0=背景, 1=腔, 2=心肌, 3=心房, 其余全部当背景
        # mapping = {0:0, 1:1, 2:2, 3:3}
    # 用 np.vectorize 快速映射，未在表里的默认 0
        # label = np.vectorize(lambda v: mapping.get(v, 0), otypes=[np.uint8])(label)
        # print(np.unique(label))   # 打印每张 label 里的像素值

    # --------------------------
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = data_name
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


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
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            # print(image.shape ,label.shape)
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image).unsqueeze(0)
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

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = image / 255
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = image_weak / 255
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        image_strong = image_strong / 255

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
