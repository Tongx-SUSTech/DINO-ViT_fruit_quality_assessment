import warnings

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from dino_experiments.util import get_embeddings, get_seeded_data_loader, Logger, AVAILABLE_DATASETS


warnings.filterwarnings("ignore", category=UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
N_RANDOM_RUNS = 5
TEST_SAMPLE_SIZE = VAL_SAMPLE_SIZE = -1
N_JOBS = -1
GLOBAL_SEED = 42
RESIZE = True

XGBOOST_PARAMS = {
    # "cascifw": {"objective": 'binary:logistic'},
    "fayoum": {"objective": "multi:softmax", "num_class": 4},
    # "apple": {"objective": "multi:softmax", "num_class": 2}
}


def main(dataset: str):
    assert dataset in AVAILABLE_DATASETS
    test_loader = get_seeded_data_loader(dataset, "test", GLOBAL_SEED, resize=RESIZE, batch_size=BATCH_SIZE)
    transformers = {dino: torch.hub.load('facebookresearch/dino:main', dino, verbose=False) for dino in
                    ['dino_vits8', 'dino_vitb8']}


    setup = {
        LogisticRegression(random_state=GLOBAL_SEED, max_iter=10000): {"C": [1, 10, 100, 1000]},
        KNeighborsClassifier(): {"n_neighbors": [3, 5, 10],
                                 "weights": ["uniform", "distance"],
                                 },
        SVC(random_state=GLOBAL_SEED): [
                {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
                ],
        RandomForestClassifier(random_state=GLOBAL_SEED): {"n_estimators": [10, 100, 500, 1000]},
        MLPClassifier(random_state=GLOBAL_SEED, max_iter=10000): {
            'hidden_layer_sizes': [(10, 30, 10), (20,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
            },
        XGBClassifier(**XGBOOST_PARAMS[dataset], nthread=1, use_label_encoder=False, verbosity=0,
                      random_state=GLOBAL_SEED): {"n_estimators": range(100, 1100, 100)},
    }

    model_names = [model.__class__.__name__ for model in setup.keys()]

    logger = Logger(f"exp_01_{dataset}")
    logger.log(["transformer", "model", "train_samples", "val_samples", "test_samples", "seed", "test_acc", "test_prec",
                "test_recall", "test_f1", "best_params"])

    for transformer_name, transformer in tqdm(transformers.items()):
        transformer.to(device)
        transformer.eval()
        X_test, y_test = get_embeddings(transformer, test_loader, TEST_SAMPLE_SIZE, batch_size=BATCH_SIZE)
        for train_sample_size in tqdm([-1], leave=False):
            for seed in tqdm(range(N_RANDOM_RUNS), leave=False):
                train_loader = get_seeded_data_loader(dataset, "train", seed, batch_size=BATCH_SIZE, resize=RESIZE)
                val_loader = get_seeded_data_loader(dataset, "val", seed, batch_size=BATCH_SIZE, resize=RESIZE)
                X_train, y_train = get_embeddings(transformer, train_loader, train_sample_size, batch_size=BATCH_SIZE)
                X_val, y_val = get_embeddings(transformer, val_loader, VAL_SAMPLE_SIZE, batch_size=BATCH_SIZE)

                ps = PredefinedSplit([-1] * len(X_train) + [0] * len(X_val))
                X = np.concatenate([X_train, X_val], axis=0)
                y = np.concatenate([y_train, y_val], axis=0)

                for model_name, (model, params) in tqdm(list(zip(model_names, setup.items())), leave=False):
                    gs = GridSearchCV(model, params, cv=ps, n_jobs=N_JOBS, refit=True)
                    gs.fit(X, y)
                    clf = gs.best_estimator_
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)
                    acc = clf.score(X_test, y_test)
                    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds)

                    logger.log([transformer_name, model_name, str(len(y_train)), str(len(y_val)), str(len(y_test)),
                                str(seed), str(acc), str(prec), str(rec), str(f1), str(gs.best_params_)])


if __name__ == '__main__':
    main("fayoum")
    # main("cascifw")
    # main("apple")
