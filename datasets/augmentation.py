"""Module for defining augmentation strategies."""

from torchvision import transforms

augmentation_strategies = {
    "none": None,
    "default": transforms.RandomHorizontalFlip(),
    }