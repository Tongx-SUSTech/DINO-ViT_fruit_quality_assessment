import warnings

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import umap.umap_ as umap
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from dino_experiments.util import get_embeddings
# from datasets import FayoumBananaDataModule, CascIfwDataModule
from datasets import FayoumBananaDataModule

data_modules = {
    "fayoum": FayoumBananaDataModule
    # "cascifw": CascIfwDataModule
}

warnings.filterwarnings("ignore", category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 5
GLOBAL_SEED = 42
RESIZE = True

ENCODERS = {
    "resnet50": nn.Sequential(*list(models.resnet50(pretrained=True, progress=False).children())[:-1]),
    "vgg": nn.Sequential(*list(models.vgg11_bn(pretrained=True, progress=False).children())[:-1]),
    "dino_vitb8": torch.hub.load('facebookresearch/dino:main', "dino_vitb8", verbose=False),
    "dino_vits8": torch.hub.load('facebookresearch/dino:main', "dino_vits8", verbose=False)
}


def generate_and_save_embeddings(dataset, encoder_name="dino_vitb8", subfolder="", norm_orient=False):

    kws = {"norm_orientation": True, "y_labels": "orientation"} if norm_orient else {}

    data_module = data_modules[dataset](BATCH_SIZE, image_resolution=224, augment=False, mode="train", **kws)
    data_loader = data_module["train"]
    file_ref_list = data_loader.dataset.data
    encoder = ENCODERS[encoder_name].to(device).to(device)
    X, y = get_embeddings(encoder, data_loader, -1, batch_size=BATCH_SIZE)

    # extra split for semi supervised umap
    random_excluded_indexes = np.random.choice(len(y), size=len(y)//2, replace=False)
    y_masked = y.copy()
    y_masked[random_excluded_indexes] = -1
    label_included = [0 if idx in random_excluded_indexes else 1 for idx in range(len(y))]
    np.save(f"results/dim_reduction/{subfolder}umap_semi_included.npy", label_included)

    if norm_orient:
        dataset = dataset + "_oriented"

    np.save(f"results/dim_reduction/{subfolder}labels_{dataset}.npy", y)
    np.save(f"results/dim_reduction/{subfolder}filenames_{dataset}.npy", np.array(file_ref_list)[:, 0])

    pca = PCA(n_components=3, random_state=GLOBAL_SEED)
    X_pca = pca.fit_transform(X)
    np.save(f"results/dim_reduction/{subfolder}{dataset}_{encoder_name}_pca_embed.npy", X_pca)
    np.save(f"results/dim_reduction/{subfolder}{dataset}_{encoder_name}_pca_explained_variance.npy", pca.explained_variance_ratio_)

    X_tsne = TSNE(perplexity=30, n_components=2, init='random', random_state=GLOBAL_SEED).fit_transform(X)
    np.save(f"results/dim_reduction/{subfolder}{dataset}_{encoder_name}_tsne_embed.npy", X_tsne)

    X_umap_u = umap.UMAP(n_neighbors=50, random_state=GLOBAL_SEED).fit_transform(X)
    X_umap_s = umap.UMAP(n_neighbors=50, random_state=GLOBAL_SEED).fit_transform(X, y=y_masked)
    np.save(f"results/dim_reduction/{subfolder}{dataset}_{encoder_name}_umap_unsupervised_embed.npy", X_umap_u)
    np.save(f"results/dim_reduction/{subfolder}{dataset}_{encoder_name}_umap_semi-supervised_embed.npy", X_umap_s)


def main(dataset, **kwargs):
        for encoder in tqdm(ENCODERS, leave=False):
            generate_and_save_embeddings(dataset, encoder, **kwargs)


if __name__ == '__main__':
    # main("cascifw", subfolder="cascifw/")
    main("fayoum", subfolder="fayoum/")
    main("fayoum", subfolder="fayoum_oriented/", norm_orient=True)

