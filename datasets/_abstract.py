import os
from collections import Counter
from typing import Union, Iterable
from abc import abstractmethod, ABC
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision

from datasets.util import RoundrobinSampler
from datasets.augmentation import augmentation_strategies

IMAGENET_NORMALIZE_VALUES = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def center_crop_to_max_dim(img_res):
    return transforms.Compose([
        transforms.Lambda(lambda x: transforms.CenterCrop(max(x.shape[-2], x.shape[-1]))(x)),
        transforms.Resize(img_res)
    ])


class FileListDataset(Dataset):

    def __init__(self, data_ref_list, transform=None):
        self.data = data_ref_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.data[idx]
        image = torchvision.io.read_image(img_path) / 255
        if image.shape[0] > 3:  # remove alpha channel
            image = image[:3, :, :]
        if self.transform:
            image = self.transform(image)

        return image, label


class DataModule(ABC):

    def __init__(self, image_resolution=None):

        if image_resolution is not None:
            if isinstance(image_resolution, int):
                self.img_res = (image_resolution, image_resolution)
            else:
                self.img_res = image_resolution

            self.squared_image = (self.img_res[0] == self.img_res[1])
        else:
            self.img_res = None
            self.squared_image = None

        self.setup()

    def __getitem__(self, key):
        if key == "train":
            return self._train_dataloader()
        elif key == "val":
            return self._val_dataloader()
        elif key == "test":
            return self._test_dataloader()

    @property
    @abstractmethod
    def n_classes(self) -> Union[Iterable[int], int]:
        pass

    @property
    @abstractmethod
    def n_samples(self):
        pass

    @property
    @abstractmethod
    def samples_per_class(self):
        pass

    @property
    @abstractmethod
    def class_names(self):
        pass

    @abstractmethod
    def setup(self) -> None:
        """Run once at initialization."""
        pass

    @abstractmethod
    def _train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def _val_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def _test_dataloader(self) -> DataLoader:
        pass

    @staticmethod
    def _get_label_weights(labels):
        counter = Counter(labels)
        n_samples = len(labels)
        n_classes = len(counter)
        weights = {k: n_samples / (n_classes * v) for k, v in counter.items()}
        return [weights[l] for l in labels]

    @staticmethod
    def _get_index_lists_by_labels(labels):
        return [[i for i, x in enumerate(labels) if x == l] for l in set(labels)]


class FileListDataModule(DataModule):
    train_frac = 0.64
    val_frac = 0.16
    test_frac = 0.2

    file_ending = None  # to implement in child class
    normalize_values = None  # to implement in child class

    def __init__(self, batch_size, image_resolution=224, y_labels=None, data_dir=None, pre_transform=None,
                 normalize=False, remove_background=None, sampler="random", split_shuffle_seed=42, sampler_seed=42,
                 mode="train_val_test", augment="none", use_random_resize_crop=False, custom_dataset_class=None):
        self.batch_size = batch_size
        self.y_labels = y_labels
        self.data_dir = data_dir
        self.normalize = normalize
        self.split_shuffle_seed = split_shuffle_seed
        self.sampler = sampler
        assert sampler in ["random", "weighted", "roundrobin"]
        self.sampler_seed = sampler_seed
        assert mode in ["train_val_test", "train_test", "train"]
        self.mode = mode

        self.pre_transform = pre_transform
        if use_random_resize_crop:
            self.resize_transform = transforms.RandomResizedCrop(image_resolution, scale=(0.5, 1.0), ratio=(1.0, 1.0))
        else:
            self.resize_transform = center_crop_to_max_dim(image_resolution)
        self.augment_transform = augmentation_strategies[augment]

        if remove_background:
            raise NotImplementedError("Background Removal not implemented for this version")
        else:
            self.background_transform = None

        if normalize:
            self.normalize_transform = transforms.Normalize(*(self.normalize_values or IMAGENET_NORMALIZE_VALUES))
        else:
            self.normalize_transform = None

        self.dataset_class = custom_dataset_class or FileListDataset

        self.data_ref_train = None  # filled in data_ref_setup method
        self.data_ref_val = None  # filled in data_ref_setup method
        self.data_ref_test = None  # filled in data_ref_setup method

        super().__init__(image_resolution=image_resolution)

    @staticmethod
    def split_list(list_, frac):
        split_int = int(frac * len(list_))
        a = list_[:split_int]
        b = list_[split_int:]
        return a, b

    @abstractmethod
    def build_file_ref_list(self):
        """Setup specific for this class. returns a list of tuples containing file paths to images and class labels"""
        pass

    def setup(self) -> None:
        file_ref_list = self.build_file_ref_list()
        self._distribute_file_refs_to_splits(file_ref_list)

    def _distribute_file_refs_to_splits(self, file_refs):

        if type(file_refs) is tuple:  #predefined splits
            if len(file_refs) == 1:
                file_refs = file_refs[0]
            elif len(file_refs) == 2:
                assert self.mode in ["train_val_test", "train_test"], f"Got 2 file ref lists but mode is {self.mode}"
                file_refs, self.data_ref_test = file_refs
                if self.mode == "train_test":
                    pass
            elif len(file_refs) == 3:
                assert self.mode == "train_val_test", f"Got 3 file ref lists but mode is {self.mode}"
                self.data_ref_train, self.data_ref_val, self.data_ref_test = file_refs
                return
            else:
                raise AssertionError(f"Cannot handle file ref tuple len == {len(file_refs)}")

        random.seed(self.split_shuffle_seed)
        random.shuffle(file_refs)
        assert sum([self.train_frac, self.val_frac, self.test_frac]) == 1.0
        if self.mode == "train":
            self.data_ref_train = file_refs
        else:
            if self.data_ref_test is not None:
                print("Found test split already assigned explicitly. Skip splitting test data.")
                self.data_ref_train = file_refs
            else:
                print("Splitting train and test data.")
                self.data_ref_test, self.data_ref_train = self.split_list(file_refs, self.test_frac)
            if self.mode == "train_val_test":
                print("Splitting train and val data.")
                self.data_ref_val, self.data_ref_train = self.split_list(self.data_ref_train,
                                                                         self.val_frac / (self.train_frac + self.val_frac))

    def _dataloader(self, transform, file_ref_list):

        dataset = self.dataset_class(file_ref_list, transform=transform)

        if self.sampler == "weighted":
            weights = self._get_label_weights([x[1] for x in file_ref_list])
            torch.manual_seed(self.sampler_seed)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=len(weights),
                                                                     replacement=False)
        elif self.sampler == "roundrobin":
            index_buckets = self._get_index_lists_by_labels([x[1] for x in file_ref_list])
            torch.manual_seed(self.sampler_seed)
            sampler = RoundrobinSampler(index_buckets)

        else:
            sampler = None

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler)

    @staticmethod
    def _compose_transforms(ts):
        return transforms.Compose([t for t in ts if t is not None])

    def _train_dataloader(self):
        transform = self._compose_transforms([self.pre_transform,
                                              self.resize_transform,
                                              self.background_transform,
                                              self.augment_transform,
                                              self.normalize_transform])
        return self._dataloader(transform, self.data_ref_train)

    def _val_dataloader(self):
        if self.mode != "train_val_test":
            raise AssertionError("Validation split not available for this mode. Use mode 'train_val_test'")

        transform = self._compose_transforms([self.pre_transform,
                                              self.resize_transform,
                                              self.background_transform,
                                              self.normalize_transform])

        return self._dataloader(transform, self.data_ref_val)

    def _test_dataloader(self):
        if self.mode == "train":
            raise AssertionError("Test split not available for this mode. Use mode 'train_val_test' or 'train_test'")

        transform = self._compose_transforms([self.pre_transform,
                                              self.resize_transform,
                                              self.background_transform,
                                              self.normalize_transform])

        return self._dataloader(transform, self.data_ref_test)


class SimpleFolderDataModule(FileListDataModule):

    file_endings = [".png", ".jpeg", ".jpg"]

    def __init__(self, batch_size, data_dir, normalize_values=None, **kwargs):
        self.data_dir = data_dir
        self.normalize_values = normalize_values
        super().__init__(batch_size, data_dir=data_dir,  **kwargs)

    @property
    def n_classes(self):
        return len(self.class_names)

    @property
    def n_samples(self):
        return sum(self.samples_per_class)

    def _samples_per_class(self, dir_):
        return [len(f) for f in self.file_lists(dir_)]

    @property
    def samples_per_class(self):
        return self._samples_per_class(self.data_dir)

    @property
    def class_names(self):
        """Return folders in data_dir as classes."""
        return sorted([c for c in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, c))])

    def class_dirs(self, dir_):
        """Return folders in data_dir as classes."""
        return [os.path.join(os.path.join(dir_, c)) for c in self.class_names]

    def file_lists(self, dir_):
        return [[os.path.join(c_dir, file) for file in os.listdir(c_dir) if any([file.endswith(end) for end in self.file_endings])]
                for c_dir in self.class_dirs(dir_)]

    def _build_file_ref_list_single_dir(self, dir_):
        out = list()
        for i, file_list in enumerate(self.file_lists(dir_)):
            for file_ in file_list:
                out.append((file_, i))
        return out

    def build_file_ref_list(self):
        return self._build_file_ref_list_single_dir(self.data_dir)


class SimpleFolderDataModuleWithPredefinedSplit(SimpleFolderDataModule):

    file_endings = [".png", ".jpeg", ".jpg"]

    def __init__(self, batch_size, data_dir, train_folder="train", val_folder="val", test_folder="test", **kwargs):
        self.data_dir = data_dir
        self.train_dir = os.path.join(self.data_dir, train_folder) if train_folder else None
        self.val_dir = os.path.join(self.data_dir, val_folder) if val_folder else None
        self.test_dir = os.path.join(self.data_dir, test_folder) if test_folder else None

        super().__init__(batch_size, data_dir=data_dir,  **kwargs)

    @property
    def class_names(self):
        """Return folders in data_dir as classes."""
        return sorted([c for c in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, c))])

    @property
    def samples_per_class(self):
        spcs = list()
        if self.train_dir:
            spcs += [self._samples_per_class(self.train_dir)]
        if self.val_dir:
            spcs += [self._samples_per_class(self.val_dir)]
        if self.test_dir:
            spcs += [self._samples_per_class(self.test_dir)]
        return np.sum(spcs, axis=0).tolist()

    def build_file_ref_list(self):
        outs = list()
        if self.train_dir:
            outs += [self._build_file_ref_list_single_dir(self.train_dir)]
        if self.val_dir:
            outs += [self._build_file_ref_list_single_dir(self.val_dir)]
        if self.test_dir:
            outs += [self._build_file_ref_list_single_dir(self.test_dir)]
        assert len(outs) in (2, 3)
        return tuple(outs)
