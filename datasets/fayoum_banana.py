import os

import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip

from datasets._abstract import FileListDataModule, FileListDataset

CLASS_NAMES = {
    "ripeness": ["Green", "Yellowish_Green", "Midripen", "Overripen"],
    "orientation": ["Left", "Right"]
}

N_SAMPLES = np.array([104, 48, 88, 33])


class FayoumNormalizedOrientationDataset(FileListDataset):

    def __init__(self, data_ref_list, transform=None):
        super(FayoumNormalizedOrientationDataset, self).__init__(data_ref_list, transform=transform)

    def __getitem__(self, idx):
        image, label = super(FayoumNormalizedOrientationDataset, self).__getitem__(idx)
        if label == 0:
            image = RandomHorizontalFlip(p=1.0)(image)
            label = 1
        return image, label


class FayoumBananaDataModule(FileListDataModule):
    orient_txt = "/datasets/Fayoum_Banana/fayoum_orientation.txt"
    file_ending = ".jpg"
    normalize_values = ((0.6237, 0.6076, 0.5367), (0.1759, 0.1925, 0.3011))

    def __init__(self, batch_size, image_resolution=(536, 960), y_labels="ripeness",
                 data_dir="./datasets/data/Fayoum_Banana",
                 normalize=True, norm_orientation=False, augment="default", **kwargs):

        if y_labels is None:
            y_labels = "ripeness"
        if y_labels not in CLASS_NAMES.keys():
            raise NotImplementedError(f"y labels {y_labels} not implemented")

        if y_labels == "orientation":
            augment = "none"
        if norm_orientation:
            augment = "none"
            assert y_labels == "orientation", "set y_labels='orientation' when norm_orientation is True"
            custom_ds = FayoumNormalizedOrientationDataset
        else:
            custom_ds = None

        # dataset has a weird distirbution of 540 and 536 height. we want to remove that bias.
        pre_transform = lambda x: transforms.CenterCrop((536, 960))(x) if x.shape[-2] == 540 else x
        '''
        if not augment:  # 当augment为False时
            augment = "none"  # 转换为"none"
        '''
        super().__init__(batch_size, image_resolution=image_resolution, y_labels=y_labels, data_dir=data_dir,
                         normalize=normalize, augment=augment, custom_dataset_class=custom_ds,
                         pre_transform=pre_transform, **kwargs)

    @property
    def n_classes(self):
        return len(CLASS_NAMES[self.y_labels])

    @property
    def n_samples(self):
        return N_SAMPLES.sum()

    @property
    def samples_per_class(self):
        return N_SAMPLES

    @property
    def class_weights(self):
        raise NotImplementedError("method is deprecated")

    @property
    def class_names(self):
        return [f"{i}_{c}" for i, c in enumerate(CLASS_NAMES[self.y_labels])]

    @classmethod
    def get_orientation(cls, filepath):
        df = pd.read_csv(cls.orient_txt).set_index("filepath")
        label = df.loc[filepath, "orientation"]
        assert label in ["l", "r"]
        if label == "l":
            return "Left"
        return "Right"

    def build_file_ref_list(self):
        paths = []
        classes = []
        for folder in os.listdir(self.data_dir):
            if not os.path.isdir(os.path.join(self.data_dir, folder)):
                continue
            for file in os.listdir(os.path.join(self.data_dir, folder)):
                if not file.endswith(self.file_ending):
                    continue
                filepath = os.path.join(self.data_dir, folder, file)
                paths.append(filepath)
                if self.y_labels == "ripeness":
                    classes.append(folder)
                elif self.y_labels == "orientation":
                    classes.append(self.get_orientation(filepath))
        ys = [CLASS_NAMES[self.y_labels].index(c) for c in classes]
        data_all = list(zip(paths, ys))
        return data_all

