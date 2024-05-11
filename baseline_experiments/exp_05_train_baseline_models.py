from baseline_experiments.train_cnn import main as train_script


class ArgNamespace(object):

    def __init__(self, adict):
        self.__dict__.update(adict)


'''
def casc_ifw_binary():
    default_args = {
        "project": "dino_baseline_cascifw_reduced_samples",
        "group": None,  # set
        "name": "",  # set
        "offline": False,
        "no_wandb": True,

        "dataset": "casc_ifw",
        "y_labels": "condition",
        "batch_size": 100,

        "model": None,  # set
        "mode": None,  # set all and clf
        "no_pretrain": False,

        "class_weight": False,
        "lr": 0.0001,
        "momentum": 0.9,
        "es_metric": "loss",
        "es_patience": 30,
        "scheduler_patience": 15,

        "seed": None,  # set
        "gpu_ids": [0, 1],
        "n_epochs": 5000,
        "n_train_samples": None  # set
    }

    args = ArgNamespace(default_args)

    seeds = list(range(5))
    modes = ["all", "clf"]

    # models = ["resnet18", "resnet50", "resnet101", "resnet152", "alexnet", "densenet", "squeezenet", "alexnet", "vgg"]
    # n_trains = [-1]

    models = ["squeezenet", "vgg"]
    n_trains = [100, 200, 500, 1000, 2000]

    for seed in seeds:
        for mode in modes:
            for n_train in n_trains:
                for model in models:
                    args.mode = mode
                    args.seed = seed
                    args.n_train_samples = n_train
                    args.model = model
                    _n = f"{model}_n={'all'if n_train == -1 else n_train}_{mode}"
                    args.group = _n
                    args.name = _n
                    try:
                        train_script(args)
                    except:
                        print(f"Training model {model} failed.")
                        exit(1)
'''


def fayoum():
    default_args = {
        "project": "dino_baseline_fayoum_reduced_samples",
        "group": None,  # set
        "name": "",  # set
        "offline": False,
        "no_wandb": True,

        "dataset": "fayoum",
        "y_labels": "ripeness",
        "batch_size": 200,

        "model": None,  # set
        "mode": None,  # set all and clf
        "no_pretrain": False,

        "class_weight": False,
        "lr": 0.0001,
        "momentum": 0.9,
        "es_metric": "loss",
        "es_patience": 30,
        "scheduler_patience": 15,

        "seed": None,  # set
        "gpu_ids": [0],
        "n_epochs": 5000,
        "n_train_samples": None  # set
    }

    args = ArgNamespace(default_args)

    seeds = list(range(10))
    modes = ["all", "clf"]

    models = ["resnet50", "alexnet"]
    # models = ["vgg"]
    n_trains = [4, 8, 20, 40, 120, -1]

    for seed in seeds:
        for mode in modes:
            for n_train in n_trains:
                for model in models:
                    args.mode = mode
                    args.seed = seed
                    args.n_train_samples = n_train
                    args.model = model
                    _n = f"{model}_n={'all'if n_train == -1 else n_train}_{mode}"
                    args.group = _n
                    args.name = _n
                    try:
                        train_script(args)
                    except Exception as e:
                        print(f"Training model {model} failed due to: {e}")
                        '''                  
                    except:
                        print(f"Training model {model} failed.")
                        exit(1)
                        '''


'''
def apple():
    default_args = {
        "project": "dino_baseline_apple_reduced_samples",
        "group": None,  # set
        "name": "",  # set
        "offline": False,
        "no_wandb": True,

        "dataset": "apple",
        "y_labels": "quality",
        "batch_size": 100,

        "model": None,  # set
        "mode": None,  # set all and clf
        "no_pretrain": False,

        "class_weight": False,
        "lr": 0.0001,
        "momentum": 0.9,
        "es_metric": "loss",
        "es_patience": 30,
        "scheduler_patience": 15,

        "seed": None,  # set
        "gpu_ids": [0],
        "n_epochs": 5000,
        "n_train_samples": None  # set
    }

    args = ArgNamespace(default_args)

    seeds = list(range(10))
    modes = ["all", "clf"]

    models = ["resnet50", "alexnet"]
    n_trains = [4, 8, 20, 40, 120, -1]

    for seed in seeds:
        for mode in modes:
            for n_train in n_trains:
                for model in models:
                    args.mode = mode
                    args.seed = seed
                    args.n_train_samples = n_train
                    args.model = model
                    _n = f"{model}_n={'all' if n_train == -1 else n_train}_{mode}"
                    args.group = _n
                    args.name = _n
                    try:
                        train_script(args)
                    except Exception as e:
                        print(f"Training model {model} failed due to: {e}")
                                    
                    # except:
                        # print(f"Training model {model} failed.")
                        # exit(1)
                    
                    
'''

if __name__ == '__main__':
    fayoum()
    # casc_ifw_binary()
    #apple()
