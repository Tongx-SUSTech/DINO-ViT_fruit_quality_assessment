import os
from datetime import datetime

import numpy as np

from ._abstract import FileListDataModule

CLASS_NAMES = {
    "cultivar": ["Fuji", "Golden", "York", "Red"],
    "condition": ["Healthy", "Injured"],
}

N_SAMPLES = np.array([[451, 836], [418, 931], [601, 996], [588, 1037]])


class CascIfwDataModule(FileListDataModule):

    file_ending = ".jpg"
    image_folder = "Raw image"
    txt_folder = "Ground Truth Text File"
    normalize_values = ((0.5296, 0.5476, 0.3714), (0.2398, 0.2247, 0.2105))

    def __init__(self, batch_size, image_resolution=120, y_labels="condition", data_dir="./data/CASC_IFW",
                 normalize=True, filter_cultivar=False, filter_condition=False, **kwargs):

        if y_labels is None:
            y_labels = "condition"
        assert y_labels in ["cultivar", "condition", "day", "damage_size", "num_damages"]
        self.filter_cultivar = filter_cultivar
        self.filter_condition = filter_condition

        self.img_dir = os.path.join(data_dir, self.image_folder)
        self.txt_dir = os.path.join(data_dir, self.txt_folder)
        self.preprocess_directory(self.img_dir)
        self.preprocess_directory(self.txt_dir)

        super().__init__(batch_size, image_resolution=image_resolution, y_labels=y_labels, data_dir=self.img_dir,
                         normalize=normalize, **kwargs)

    @staticmethod
    def preprocess_directory(dir_):
        def reformat_str(str_):
            out = str_
            if str_.startswith("2009"):
                new_date = datetime.strptime(str_[:8], "%Y%m%d").strftime("%m%d%y")
                out = new_date + out[8:].lower()
            out = out.replace("_healthy", "_Healthy").replace("_damaged", "_Injured")\
                .replace("_injured", "_Injured").replace("_fuji", "_Fuji").replace(
                "_golden", "_Golden").replace("_york", "_York").replace("_red", "_Red")
            return out

        for folder in os.listdir(dir_):
            if folder.startswith("."):
                continue
            new_folder = reformat_str(folder)
            os.rename(os.path.join(dir_, folder), os.path.join(dir_, new_folder))
            for file in os.listdir(os.path.join(dir_, new_folder)):
                if file.startswith("."):
                    continue
                new_file = reformat_str(file)
                os.rename(os.path.join(dir_, new_folder, file), os.path.join(dir_, new_folder, new_file))

    @property
    def n_classes(self):
        if self.y_labels not in CLASS_NAMES:
            raise NotImplementedError(f"n_classes not implemented for y_labels=={self.y_labels}")
        return len(CLASS_NAMES[self.y_labels])

    @property
    def n_samples(self):
        if self.y_labels in ["condition", "cultivar"]:
            return N_SAMPLES.sum()
        raise NotImplementedError(f"n_samples not implemented for y_labels=={self.y_labels}")

    @property
    def samples_per_class(self):
        if self.y_labels == "condition":
            return N_SAMPLES.sum(axis=0)
        elif self.y_labels == "cultivar":
            return N_SAMPLES.sum(axis=1)
        raise NotImplementedError(f"method not implemented for y_labels=={self.y_labels}")

    @property
    def class_names(self):
        if self.y_labels == "cultivar":
            return CLASS_NAMES["cultivar"]
        elif self.y_labels == "condition":
            return CLASS_NAMES["condition"]
        raise Exception(f"class_names method not available for y_labels=={self.y_labels}")

    @staticmethod
    def read_damage_txt(path):
        if not os.path.isfile(path):
            return []
        with open(path, 'r') as file:
            defects = list()
            raw_str = file.read()
            for defect in raw_str.split(","):
                if len(defect) == 0:
                    continue
                defect_d = dict()
                for var in defect.split(" "):
                    name, value = var.split("=")
                    defect_d[name] = int(value)
                defects.append(defect_d)
        return defects


    def img_path_to_txt_path(self, path):
        assert path.startswith(self.img_dir)
        new_path = os.path.join(self.txt_dir, path[len(self.img_dir)+1:])
        new_path = new_path.replace(".jpg", ".txt")
        return new_path[:-7] + 'gt_' + new_path[-7:]

    def build_file_ref_list(self):
        paths = []
        classes = []
        earliest_date = None
        for folder in os.listdir(self.data_dir):
            if not os.path.isdir(os.path.join(self.data_dir, folder)):
                continue
            for file in os.listdir(os.path.join(self.data_dir, folder)):
                if not file.endswith(self.file_ending):
                    continue
                filepath = os.path.join(self.data_dir, folder, file)
                date, cultivar, condition, _ = file.split("_", 3)
                if self.filter_cultivar and cultivar != self.filter_cultivar:
                    continue
                if self.filter_condition and condition != self.filter_condition:
                    continue
                paths.append(filepath)
                if self.y_labels == "condition":
                    classes.append(condition)
                elif self.y_labels == "cultivar":
                    classes.append(cultivar)
                elif self.y_labels == "day":
                    date = datetime.strptime(date, "%m%d%y")
                    if earliest_date is None or date < earliest_date:
                        earliest_date = date
                    classes.append(date)
        if self.y_labels == "day":
            ys = [(date - earliest_date).days for date in classes]
        elif self.y_labels in ["damage_size", "num_damages"]:
            txt_paths = [self.img_path_to_txt_path(p) for p in paths]
            defects = [self.read_damage_txt(p) for p in txt_paths]
            if self.y_labels == "num_damages":
                ys = [len(d) for d in defects]
            else:
                ys = list()
                for img_defects in defects:
                    ys.append(max([d["r"] for d in img_defects]) if len(img_defects) > 0 else 0)
        else:
            ys = [CLASS_NAMES[self.y_labels].index(c) for c in classes]
        data_all = list(zip(paths, ys))
        return data_all
