import os
from typing import List

import numpy as np
import torch

#from datasets import CascIfwDataModule, FayoumBananaDataModule
from datasets import FayoumBananaDataModule
#from datasets import AppleDataModule


device = 'cuda' if torch.cuda.is_available() else 'cpu'

AVAILABLE_DATASETS = {
    # "cascifw": CascIfwDataModule,
    "fayoum": FayoumBananaDataModule
    # "apple": AppleDataModule
}


def get_seeded_data_loader(dataset, subset, seed, batch_size=100, resize=True, mode="train_val_test",
                           y_labels=None, **kwargs):
    assert dataset in AVAILABLE_DATASETS.keys()
    assert subset in ["train", "val", "test"]
    img_res = 224 if resize else None
    if dataset == "fayoum" and subset == "train":
        sampler = "roundrobin"
    # elif dataset == "apple" and subset == "train":
        # sampler = "roundrobin"
    # elif dataset == "cascifw" and subset == "train":
        # sampler = "weighted"
    # elif dataset == "fayoum" and subset == "train":
        # sampler = "roundrobin"
    else:
        sampler = "random"
    return AVAILABLE_DATASETS[dataset](batch_size=batch_size, image_resolution=img_res, normalize=True,
                                       sampler=sampler, sampler_seed=seed, mode=mode, y_labels=y_labels,
                                       **kwargs)[subset]


def get_embeddings(model, dataloader, n_samples, batch_size=100):
    n_iter = n_samples // batch_size
    embeddings, ys = [], []
    i = 0
    for x, y in dataloader:
        embeddings += [model(x.to(device)).cpu().detach().numpy()]
        ys += [y]
        i += 1
        if i == n_iter:
            break
    embed_cat = np.concatenate(embeddings, axis=0)
    return embed_cat.reshape((embed_cat.shape[0], -1)), np.concatenate(ys)


class Logger:

    def __init__(self, experiment_name):
        os.makedirs("../results/", exist_ok=True)
        self.filepath = "results/" + experiment_name + ".txt"

    def log(self, params: List):
        log_str = ";".join([str(p) for p in params])
        with open(self.filepath, "a") as f:
            f.write(log_str)
            f.write("\n")

