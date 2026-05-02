import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import transforms

from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage.interpolation import zoom

from utils import *

class RandomCrop(transforms.RandomResizedCrop):
    def __call__(self, images):
        i, j, h, w = self.get_params(images[0], self.scale, self.ratio)
        for imageCount in range(len(images)):
            images[imageCount] = transforms.functional.resized_crop(images[imageCount], i, j, h, w, self.size, self.interpolation)
        return images
        
class SkinDataset(Dataset):
    def __init__(self, images, masks,
                 transform=True, typeData = "train"):
        self.transform = transform if typeData == "train" else False  # augment data bool
        self.typeData = typeData
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)

    def rotate(self, image, mask, degrees=(-15,15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask
    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask
    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask
    def random_resized_crop(self, image, mask, p=0.1):
        if torch.rand(1) < p:
            image, mask = RandomCrop((192, 256), scale=(0.8, 0.95))([image, mask])
        return image, mask

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        return image, mask

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = Image.fromarray(self.masks[idx])
    ####################### augmentation data ##############################
        if self.transform:
            image, mask = self.augment(image, mask)
        image = transforms.ToTensor()(image)
        mask = np.asarray(mask, np.int64)
        mask = torch.from_numpy(mask[np.newaxis])
        return image, mask


class DSBDataset(Dataset):
    """Data path saved in .npz file
        x_train, x_test: (256, 256, 3), [0, 255]
        y_train, y_test: (256, 256), {0, 255} """

    def __init__(self, data_path, type = None , transform=False):
      super().__init__()

      data_np = np.load(data_path)
      self.images = data_np[f"x_{type}"] #ISIC, DSB
      self.masks  = data_np[f"y_{type}"]
      self.transform = transform

    def __getitem__(self, idx):
      image = Image.fromarray(self.images[idx])
      mask = Image.fromarray(self.masks[idx])
      target_size = (256, 192)
      image = image.resize(target_size, Image.Resampling.BICUBIC)
      mask = mask.resize(target_size, Image.Resampling.BICUBIC)

      if self.transform:
          image, mask = augment(image, mask)

      image = transforms.ToTensor()(np.array(image))
      mask = np.expand_dims(mask, axis = -1)
      mask = transforms.ToTensor()(np.array(mask))

      return image, mask

    def __len__(self):
      return len(self.images)


class BUSIDataset(Dataset):
    def __init__(self, images, masks, transform=True, typeData="train"):
        self.transform = transform if typeData == "train" else False
        self.typeData = typeData
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def rotate(self, image, mask, degrees=(-15, 15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask

    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def vertical_flip(self, image, mask, p=0.2):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def random_resized_crop(self, image, mask, p=0.3):
        if torch.rand(1) < p:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, h, w, (192, 256))
            mask = TF.resized_crop(mask, i, j, h, w, (192, 256), interpolation=Image.NEAREST)
        return image, mask

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        image, mask = elastic_deformation(image, mask)

        # brightness/contrast jitter
        color_aug = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        image = color_aug(image)
        image = add_speckle_noise(image, p=0.3)
        return image, mask

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            mask = mask.astype(np.uint8)

        if len(mask.shape) == 3:
            mask = mask.squeeze()

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.transform:
            image, mask = self.augment(image, mask)

        image = transforms.ToTensor()(image)
        mask = np.asarray(mask, np.uint8)
        mask = (mask > 0).astype(np.uint8)  # binarize
        mask = torch.from_numpy(mask[np.newaxis])

        return image, mask