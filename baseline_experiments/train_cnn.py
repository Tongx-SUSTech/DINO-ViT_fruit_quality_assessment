import os
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torchmetrics.classification as torch_metrics
from tqdm import tqdm


# from datasets import CascIfwDataModule, FayoumBananaDataModule
from datasets import FayoumBananaDataModule
# from datasets import AppleDataModule
from dino_experiments.util import get_seeded_data_loader

from baseline_experiments.model import initialize_model, unfreeze_all_params

AVAILABLE_DATASETS = {
    # "casc_ifw": CascIfwDataModule,
    "fayoum": FayoumBananaDataModule
    # "apple": AppleDataModule
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, data_module, optimizer, scheduler, logger, ckpt_dir, num_epochs=25, class_weighting=True,
          early_stopping_metric="accuracy", early_stopping_patience=30, n_train_batches=-1):
    if class_weighting:
        class_weights = torch.FloatTensor(data_module.class_weights).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
    metrics = {
        # "train": {"accuracy": torch_metrics.Accuracy().to(device)},
        # "val": {"accuracy": torch_metrics.Accuracy().to(device)}
        "train": {"accuracy": torch_metrics.Accuracy(num_classes=data_module.n_classes, average='macro',
                                                     task='multiclass').to(device)},
        "val": {"accuracy": torch_metrics.Accuracy(num_classes=data_module.n_classes, average='macro',
                                                   task='multiclass').to(device)}

    }

    assert early_stopping_metric == "loss" or early_stopping_metric in metrics["val"].keys()

    best_val_loss = np.inf
    best_early_stop_metric = 0.0
    early_stopping_count = 0
    for epoch in tqdm(range(num_epochs)):

        for phase in ['train', 'val']:
            dataloader = data_module[phase]

            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloader):

                if phase == "train" and i == n_train_batches:
                    break

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                for metric in metrics[phase].values():
                    metric(preds, labels)  # Update metrics

            epoch_metrics = dict()
            epoch_loss = running_loss / len(dataloader)
            logger.log({f"{phase}_loss": epoch_loss})
            for metric_name, metric in metrics[phase].items():
                m = metric.compute()
                epoch_metrics[metric_name] = m
                logger.log({f"{phase}_{metric_name}": m})
                metric.reset()

            if phase == 'val':
                #  Evaluate early stopping criteria
                best_epoch = False
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    if early_stopping_metric == "loss":
                        best_epoch = True
                if early_stopping_metric != "loss":
                    epoch_early_stop_metric = epoch_metrics[early_stopping_metric]
                    if epoch_early_stop_metric > best_early_stop_metric:
                        best_early_stop_metric = epoch_early_stop_metric
                        best_epoch = True
                if best_epoch:
                    early_stopping_count = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    early_stopping_count += 1
                    if early_stopping_count == early_stopping_patience:
                        print(f"Early stopping at epoch {epoch} based on criterion {early_stopping_metric}.\n"
                              f"Loading state from epoch {epoch - early_stopping_patience}.\n"
                              f"Best validation {early_stopping_metric}: "
                              f"{best_val_loss if early_stopping_metric == 'loss' else best_early_stop_metric}")
                        model.load_state_dict(torch.load(ckpt_path))
                        return model

                #  Schedule Learning rate
                logger.log({"lr": optimizer.param_groups[0]['lr']})
                scheduler.step(epoch_loss if early_stopping_metric == "loss" else epoch_early_stop_metric)

    return model


@torch.no_grad()
def test(model, data_module, logger):
    n_classes = data_module.n_classes
    metrics = {"accuracy": torch_metrics.Accuracy(num_classes=data_module.n_classes, average='macro',
                                                 task='multiclass').to(device),
               "precision": torch_metrics.Precision(average=None, num_classes=n_classes, task='multiclass').to(device),
               "recall": torch_metrics.Recall(average=None, num_classes=n_classes, task='multiclass').to(device),
               "f1": torch_metrics.F1Score(average=None, num_classes=n_classes, task='multiclass').to(device)
               }

    model.eval()

    for dataset in ["train", "val", "test"]:
        all_labels = list()
        all_probas = list()
        for inputs, labels in data_module[dataset]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for metric in metrics.values():
                metric(preds, labels)  # Update metrics

            all_labels.append(labels)
            all_probas.append(outputs)

        for metric_name, metric in metrics.items():
            value = metric.compute()
            if not value.shape == torch.Size([]):  # 0-dim tensor (contains 1 value)
                value = str(value)  # convert tensor with more than 1 value to string to make it loggable
                value = value[value.find("[") + 1:value.rfind("]")]
            logger.run.summary[f"{dataset}_{metric_name}"] = value
            metric.reset()

    # log roc curve, pr curve, and confusion matrix
    y_true = torch.cat(all_labels).to("cpu")
    y_probas = torch.cat(all_probas).to("cpu")
    class_names = data_module.class_names

    wandb.log({"test_roc": wandb.plot.roc_curve(y_probas=y_probas, y_true=y_true, labels=class_names)})
    wandb.log({"test_pr": wandb.plot.pr_curve(y_probas=y_probas, y_true=y_true, labels=class_names)})
    wandb.log({"test_cm": wandb.plot.confusion_matrix(probs=y_probas.detach().numpy(),
                                                      y_true=y_true.detach().numpy(),
                                                      class_names=class_names,
                                                      title="Confusion matrix"
                                                      )})


def main(args):
    assert args.mode in ["all", "clf", "finetune"]

    torch.manual_seed(args.seed)
    if args.n_train_samples > 0:
        batch_size = min(args.n_train_samples, args.batch_size)
    else:
        batch_size = args.batch_size

    # Initialize Data Module
    # dm = AVAILABLE_DATASETS[args.dataset](args.batch_size, y_labels=args.y_labels, normalize=True)
    class DummyDataModule(dict):
        if args.dataset == "apple":
            n_classes = 2
            class_names = ["Fresh", "Rotten"]
        elif args.dataset == "casc_ifw":
            n_classes = 2
            class_names = ["Healthy", "Damaged"]
        else:
            n_classes = 4
            class_names = ["Green", "Yellowish_Green", "Midripen", "Overripen"]

        def __init__(self, use_sampler, img_res):
            self.use_sampler = use_sampler
            self.img_res = img_res

        def __getitem__(self, item):
            assert item in ["train", "val", "test"]
            return get_seeded_data_loader(args.dataset, item, args.seed,
                                          batch_size=batch_size)

    # Initialize model
    train_conv = args.mode == "all"
    torch.manual_seed(args.seed)
    model, input_size = initialize_model(args.model, DummyDataModule.n_classes, train_conv=train_conv,
                                         use_pretrained=not args.no_pretrain)

    dm = DummyDataModule(args.n_train_samples != -1, input_size)

    model = model.to(device)

    if device == "cuda":
        # device_ids = [0]  # 如果你只有一个 GPU
        # model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = torch.nn.DataParallel(model, args.gpu_ids)

    # Initialize optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min' if args.es_metric == "loss" else "max",
                                                     patience=args.scheduler_patience,
                                                     verbose=True
                                                     )

    # Initialize wandb logger
    wandb.init(project=args.project,
               group=args.group,
               name=f"{args.name} ({datetime.datetime.now().strftime('%Y.%m.%d. %H:%M')})",
               mode="offline" if args.offline is True else "online",
               config=args)
    wandb.watch(model, log_freq=100)  # log gradients

    ckpt_dir = "checkpoints/" + args.name + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    os.makedirs(ckpt_dir, exist_ok=True)
    # regular training
    print("start regular training")
    model = train(model, dm, optimizer, scheduler, logger=wandb, ckpt_dir=ckpt_dir, num_epochs=args.n_epochs,
                  class_weighting=args.class_weight, early_stopping_metric=args.es_metric,
                  early_stopping_patience=args.es_patience, n_train_batches=args.n_train_samples // batch_size)

    # finetuning
    if args.mode == "finetune":
        print("start finetuning")
        finetune_lr = args.lr / 10
        optimizer = optim.SGD(model.parameters(), lr=finetune_lr, momentum=args.momentum)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min' if args.es_metric == "loss" else "max",
                                                         patience=args.scheduler_patience,
                                                         verbose=True
                                                         )
        unfreeze_all_params(model)
        model = train(model, dm, optimizer, scheduler, logger=wandb, ckpt_dir=ckpt_dir, num_epochs=args.n_epochs,
                      class_weighting=args.class_weight, early_stopping_metric=args.es_metric,
                      early_stopping_patience=args.es_patience, n_train_batches=args.n_train_samples // batch_size)

    print("Testing...")
    test(model, dm, logger=wandb)

    # Close wandb run
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # wandb args
    parser.add_argument('--project', default="test", type=str, help='Run Name')
    parser.add_argument('--group', default="test", type=str, help='Run Name')
    parser.add_argument('--name', default='test', type=str, help='Run Name')
    parser.add_argument('--offline', default=False, action="store_true", help='wand offline mode')
    # dataset args
    parser.add_argument('--dataset', default="fayoum", choices=AVAILABLE_DATASETS.keys())
    parser.add_argument('--y_labels', default=None, type=str, help="Dataset specific label definition, if applicable.")
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size per GPU')
    # model args
    parser.add_argument('--model', default="resnet50", type=str, help='Name of pretrained model')
    parser.add_argument('--mode', default="all", choices=["all", "clf", "finetune"],
                        help='all: full model'
                             'clf: classifier only'
                             'finetune: first classifier then all with reduced learning rate')
    parser.add_argument('--no_pretrain', default=False, action='store_true', help='No init w/ pretrained weights')
    # optimizer / scheduler args
    parser.add_argument('--class_weight', default=False, action='store_true',
                        help='Disable class weighting in loss function')
    parser.add_argument('--lr', default=0.0001, type=float, help='Starting learning rate of optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='Starting momentum of optimizer')
    parser.add_argument('--es_metric', default="loss", type=str, help='observed metric for early stopping')
    parser.add_argument('--es_patience', default=30, type=int, help='patience for early stopping (n epochs)')
    parser.add_argument('--scheduler_patience', default=15, type=int, help='patience for lr adaption (n epochs)')
    # other args
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--n_epochs', default=10000, type=int, help='Max number of epochs')
    parser.add_argument('--n_train_samples', default=-1, type=int, help='Number of training samples (DINO)')

    main(parser.parse_args())
