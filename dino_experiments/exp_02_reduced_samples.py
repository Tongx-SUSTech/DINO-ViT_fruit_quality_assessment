import warnings

import torch
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from dino_experiments.util import get_embeddings, get_seeded_data_loader, Logger, AVAILABLE_DATASETS
from .exp_01_best_clf import XGBOOST_PARAMS


warnings.filterwarnings("ignore", category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 10
N_RANDOM_RUNS = 10
TEST_SAMPLE_SIZE = -1
N_JOBS = -1
GLOBAL_SEED = 42
RESIZE = True

TRAIN_SET_SIZES = {
    "cascifw": [100, 200, 500, 1000, 2000],
    "fayoum": [4, 8, 20, 40, 120, -1]
}

BEST_CLF = {  # based on exp_01
    "cascifw": ('dino_vitb8', SVC(**{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}, random_state=GLOBAL_SEED)),
    "fayoum": ('dino_vits8', XGBClassifier(**XGBOOST_PARAMS["fayoum"], **{"n_estimators": 100}, nthread=1,
                                           use_label_encoder=False, verbosity=0, random_state=GLOBAL_SEED))
}


def main(dataset: str):
    assert dataset in AVAILABLE_DATASETS
    logger = Logger(f"exp_02_{dataset}")
    logger.log(["transformer", "model", "train_samples", "val_samples", "test_samples", "seed", "test_acc", "test_prec",
                "test_recall", "test_f1"])

    transformer = torch.hub.load('facebookresearch/dino:main', BEST_CLF[dataset][0], verbose=False)
    transformer.to(device)
    transformer.eval()

    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=BATCH_SIZE)
    X_test, y_test = get_embeddings(transformer, test_loader, TEST_SAMPLE_SIZE, batch_size=BATCH_SIZE)

    for train_sample_size in tqdm(TRAIN_SET_SIZES[dataset]):
        batch_size = min(BATCH_SIZE, train_sample_size) if train_sample_size != -1 else BATCH_SIZE
        for seed in tqdm(range(N_RANDOM_RUNS), leave=False):
            train_loader = get_seeded_data_loader(dataset, "train", seed, batch_size=batch_size, resize=RESIZE)
            X_train, y_train = get_embeddings(transformer, train_loader, train_sample_size, batch_size=batch_size)

            clf = BEST_CLF[dataset][1]
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = clf.score(X_test, y_test)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds)

            logger.log([BEST_CLF[dataset][0], BEST_CLF[dataset][1].__class__.__name__, str(len(y_train)),
                        str(0), str(len(y_test)), str(seed), str(acc), str(prec), str(rec), str(f1)])


if __name__ == '__main__':
     main("fayoum")
    #main("cascifw")
    # main("apple")
