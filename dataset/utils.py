import numpy as np
import torch
from PIL import Image
import random
import cv2

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF

def rotate(img, msk, degrees=(-15,15), p=0.5):
    if torch.rand(1) < p:
        degree = np.random.uniform(*degrees)
        img = img.rotate(degree, Image.NEAREST)
        msk = msk.rotate(degree, Image.NEAREST)
    return img, msk

def horizontal_flip(img, msk, p=0.5):
    if torch.rand(1) < p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
    return img, msk

def vertical_flip(img, msk, p=0.5):
    if torch.rand(1) < p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            msk = msk.transpose(Image.FLIP_TOP_BOTTOM)
    return img, msk

def augment(img, msk):
    img, msk = horizontal_flip(img, msk)
    img, msk = vertical_flip(img, msk)
    img, msk = rotate(img, msk)
    return img, msk

# --------- Elastic deformation (dùng OpenCV) ----------
def elastic_deformation(image, mask, alpha=50, sigma=6, p=0.3):
    if torch.rand(1) > p:
        return image, mask
    random_state = np.random.RandomState(None)
    shape = image.size[::-1]  # PIL: (W,H), numpy: (H,W)
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    img_np = np.array(image)
    mask_np = np.array(mask)

    distorted_img = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    distorted_mask = cv2.remap(mask_np, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    return Image.fromarray(distorted_img), Image.fromarray(distorted_mask)


# --------- Speckle noise ----------
def add_speckle_noise(image, p=0.3, sigma=0.1):
    if torch.rand(1) > p:
        return image
    img_np = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img_np.shape)
    noisy_img = img_np + img_np * noise
    noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)
