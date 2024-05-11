from dino_experiments.exp_01_best_clf import main as experiment_1
from dino_experiments.exp_02_reduced_samples import main as experiment_2
from dino_experiments.exp_03_feature_encoders import main as experiment_3
from dino_experiments.exp_04_dim_reduction import main as experiment_4
from baseline_experiments.exp_05_train_baseline_models import fayoum as experiment_05_fayoum
#from baseline_experiments.exp_05_train_baseline_models import apple as experiment_05_apple
#                                                             casc_ifw_binary as experiment_05_cascifw

RUN_BASELINE = False
# RUN_BASELINE = True

if __name__ == '__main__':

    # Run DINO-based experiments

    # Experiment 1:
    # - combine pretrained DINO ViT with different shallow classifiers
    # - run GridSearch on each to find the best combination
    # - Paper reference: Table 1(c), 1(d)
    experiment_1("fayoum")
    #experiment_1("cascifw")
    #experiment_1("apple")

    # Experiment 2:
    # - take best combinations from experiment 1 and refit with reduced training data sizes
    # - Paper reference: Fig. 3
    experiment_2("fayoum")
    #experiment_2("cascifw")
    #experiment_2("apple")

    # Experiment 3:
    # - take best shallow classifier from experiment 1 and combine it with different feature encoders
    # - Paper reference: Table 2
    experiment_3("fayoum")
    #experiment_3("cascifw")
    #experiment_3("apple")

    # Experiment 4:
    # - Create low-dimensional embedding representations using PCA.
    # - Paper reference: Fig. 4 and Fig. 5
    experiment_4("fayoum", subfolder="fayoum/")
    experiment_4("fayoum", subfolder="fayoum_oriented/", norm_orient=True)
    #experiment_4("cascifw", subfolder="cascifw/")
    #experiment_4("apple", subfolder="apple/")

    if RUN_BASELINE:
        # run baseline experiments (CNN training)
        # Please note: these experiments currently require Weights&Biases for logging

        # Paper reference: Table 1(a), 1(b), and Fig. 3
        experiment_05_fayoum()
        # experiment_05_cascifw()
        # experiment_05_apple()
