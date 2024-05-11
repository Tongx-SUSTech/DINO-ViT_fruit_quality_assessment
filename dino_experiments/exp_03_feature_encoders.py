import warnings
from torchvision import models
import torch.nn as nn
import torch
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from dino_experiments.util import get_seeded_data_loader, get_embeddings, Logger, AVAILABLE_DATASETS
from dino_experiments.exp_01_best_clf import XGBOOST_PARAMS

warnings.filterwarnings("ignore", category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_RANDOM_RUNS = 5
BATCH_SIZE = 10
TEST_SAMPLE_SIZE = -1
GLOBAL_SEED = 42
RESIZE = True


BEST_CLF = {  # based on exp_01 results
    "cascifw": SVC(**{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}, random_state=GLOBAL_SEED),
    "fayoum": XGBClassifier(**XGBOOST_PARAMS["fayoum"], **{"n_estimators": 100}, nthread=1,
                                           use_label_encoder=False, verbosity=0, random_state=GLOBAL_SEED)
}


def main(dataset: str):
    assert dataset in AVAILABLE_DATASETS
    feature_encoder = {
        "resnet18": nn.Sequential(*list(models.resnet18(pretrained=True, progress=False).children())[:-1]),
        "resnet50": nn.Sequential(*list(models.resnet50(pretrained=True, progress=False).children())[:-1]),
        "resnet101": nn.Sequential(*list(models.resnet101(pretrained=True, progress=False).children())[:-1]),
        "resnet152": nn.Sequential(*list(models.resnet152(pretrained=True, progress=False).children())[:-1]),
        "vgg": nn.Sequential(*list(models.vgg11_bn(pretrained=True, progress=False).children())[:-1]),
    }

    for dino in ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50']:
        feature_encoder[dino] = torch.hub.load('facebookresearch/dino:main', dino, verbose=False).to(device)

    logger = Logger(f"exp_03_{dataset}")
    logger.log(["encoder", "model", "train_samples", "val_samples", "test_samples", "seed", "test_acc", "test_prec",
                "test_recall", "test_f1"])

    for encoder_name, encoder in tqdm(feature_encoder.items()):
        encoder.to(device)
        encoder.eval()

        for train_sample_size in tqdm([-1], leave=False):
            for seed in tqdm(range(N_RANDOM_RUNS), leave=False):
                train_loader = get_seeded_data_loader(dataset, "train", seed, batch_size=BATCH_SIZE, resize=RESIZE)
                test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, batch_size=BATCH_SIZE, resize=RESIZE)
                X_train, y_train = get_embeddings(encoder, train_loader, train_sample_size, batch_size=BATCH_SIZE)
                X_test, y_test = get_embeddings(encoder, test_loader, TEST_SAMPLE_SIZE)
                clf = BEST_CLF[dataset]
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = clf.score(X_test, y_test)
                prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds)

                logger.log([encoder_name, BEST_CLF[dataset].__class__.__name__, str(len(y_train)), str(0),
                            str(len(y_test)), str(seed), str(acc), str(prec), str(rec), str(f1)])

        del encoder
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main("fayoum")
    # main("cascifw")
