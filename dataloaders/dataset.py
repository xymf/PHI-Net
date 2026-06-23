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
        max_pixel_value = np.max(label)
        # print("最大像素值:", max_pixel_value)
        # label = (label > 0).astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = data_name
        return sample

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


import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class BaseDataSets_BUSI(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None):
        self._base_dir = base_dir
        self.split = split
        self.transform = transform

        # 1. 确保路径正确拼接
        images_dir = os.path.join(base_dir, split, "images")
        self.data_list = os.listdir(images_dir)

        # [重要修改] 2. 必须排序！
        # 否则每次运行顺序可能不同，导致 labeled/unlabeled 划分混乱，无法复现实验
        self.data_list.sort()

        self.data_len = len(self.data_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_name = self.data_list[idx]

        # 读取图片和标签
        image_path = os.path.join(self._base_dir, self.split, "images", data_name)
        label_path = os.path.join(self._base_dir, self.split, "labels", data_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        max_pixel_value = np.max(label)
        print("最大像素值:", max_pixel_value)
        # [重要修改] 3. 标签处理 (适配二分类)
        # BUSI 的标签通常是 0 和 255。我们需要将其转换为 0 和 1。
        # 如果不处理，CrossEntropy 会报错 (因为找不到 class 255)
        if label.max() > 1:
            label = (label > 128).astype(np.uint8)

        sample = {'image': image, 'label': label}

        if self.split == "train":
            # 训练集：依靠 transform (RandomGenerator) 进行数据增强、归一化和转 Tensor
            if self.transform:
                sample = self.transform(sample)
        else:
            # [重要修改] 4. 验证集/测试集处理
            # 这里的 sample 还是 numpy，必须手动转为 Tensor 并归一化，
            # 否则 DataLoader 取出来的是 numpy，送入网络会报错。

            # 归一化图片 [0, 255] -> [0.0, 1.0]
            image = image.astype(np.float32) / 255.0
            # 增加通道维度 [H, W] -> [1, H, W]
            image = torch.from_numpy(image).unsqueeze(0).float()
            # 标签转 Tensor [H, W] -> [H, W]
            label = torch.from_numpy(label.astype(np.uint8)).long()

            sample['image'] = image
            sample['label'] = label

        sample['case'] = data_name
        return sample


class RandomGenerator_BUSI(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # ... 这里是旋转、翻转、缩放等增强代码 ...
        # ... 假设增强后的 image 还是 numpy array ...

        # [关键] 必须包含以下归一化和转 Tensor 的步骤
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).float()
        # 训练时通常除以 255.0 使其在 [0, 1] 范围
        image /= 255.0

        label = torch.from_numpy(label.astype(np.uint8)).long()

        sample = {'image': image, 'label': label}
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
