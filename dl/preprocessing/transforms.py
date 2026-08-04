"""PyTorch Image Transformation Pipelines for CT Kidney Images (DL).

Defines training transformations (Resize, RandomHorizontalFlip, RandomRotation,
ColorJitter, Normalization) and validation/test transformations (Resize, Normalization).
"""

from typing import Tuple
from torchvision import transforms

# ImageNet normalization standard values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns PyTorch data augmentation and normalization pipeline for training."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_test_transforms(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Returns PyTorch evaluation transformation pipeline (Resize & Normalize only)."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
