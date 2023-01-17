# Overview
Code for the paper *Facilitated machine learning for image-based fruit quality assessment* published in the [Journal of Food Engineering](https://www.sciencedirect.com/science/article/pii/S0260877422004551?via%3Dihub).

A preprint version was published earlier on [arXiv](https://arxiv.org/abs/2207.04523).

# Appendix

For additional illustrations see the appendix file: [appendix.pdf](appendix.pdf)

# Source Code

## Python setup

The code was tested with python version 3.8 and 3.10. Make sure to install all packages in [requirements.txt](requirements.txt) and to have CUDA-compatible GPU available to be able to run all experiments.

## Datasets

The data sets used in this research are owned by the respective authors and are therefore not shared in this repository.
If you like to use them, please reach out to the authors.

In order to reproduce these experiments, place the files in the `datasets/data/` folder in accordance with the depicted folder structure.

## Run experiments

If you want to run all experiments at once, please refer to the [run_all_experiments.py](run_all_experiments.py) file.
These scripts save interim results in the `results/` folder.

Basline experiments are logged using Weights&Biases. To run these, you need an account there.

Please note that this might take several hours and your machine should be set up with a CUDA-compatible GPU.

## Plots and tables

Tables and figures are generated in the notebook `tables_and_figures.ipynb`.
It relies on precomputed data that is saved in the `results` folder by the `run_all_experiments.py` script.