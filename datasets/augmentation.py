"""Module for defining augmentation strategies."""

from torchvision import transforms

augmentation_strategies = {
    "none": None,
    "default": transforms.RandomHorizontalFlip(),
    "false": transforms.RandomHorizontalFlip()
    }

'''
from torchvision import transforms

# 定义一个不做任何改变的 lambda 函数，作为“无增强”策略
no_augmentation = lambda x: x

augmentation_strategies = {
    "none": no_augmentation,  # 使用无操作的 lambda 函数代替 None
    "default": transforms.RandomHorizontalFlip(),
}


# 在使用字典的代码中添加逻辑以处理 False
def apply_augmentation(augment):
    if not augment:
        augment_key = "none"
    else:
        augment_key = augment  # 或其他逻辑以确定正确的键

    # 获取转换函数
    transform = augmentation_strategies[augment_key]
    return transform
'''