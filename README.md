# Adaptive Divergence for Rapid Adversarial Optimization

This repository contains experiments for Adaptive Divergence for Rapid Adversarial Optimization study.

## Installation

This repository uses the following libraries:
- catboost               0.17.4
- matplotlib             3.0.2
- numpy                  1.17.2
- pythia-mill            (https://gitlab.com/mborisyak/pythia-mill)
- scikit-learn           0.20.1
- scikit-optimize        0.5.2
- scipy                  1.3.0
- torch                  1.1.0
- tqdm                   4.32.2


Among non-default packages [PythiaMill](https://gitlab.com/mborisyak/pythia-mill) library requires manual installation.
Please, follow the instructions in the corresponding repositories.

Other packages are available from the default pip repository and required versions are specified in `setup.py`.

## Experiments

Jupyter notebooks with the experiments described in the paper can be found in `notebooks/` directory:
- `AD-<task name>-<method name>.ipynb` --- notebooks for profiling adaptive divergences on the synthetic tasks;
- `BO-XOR-GBDT.ipynb` --- the experiment with Bayesian Optimization over GBDT-based adaptive divergences on one of the synthetic tasks;
- `BO-Pythia1-Cat.ipynb` --- tuning Pythia hyper-parameters with Bayesian Optimization and CatBoost-based adaptive divergences;
- `AVO.ipynb` --- experiments with Adversarial Variational Optimization;
- `Pythia-Tune-AVO.ipynb` --- an example implementation of AVO (source: [MLHEP 2019](https://github.com/yandexdataschool/mlhep2019/blob/master/notebooks/day-8/Pythia-Tune-AVO.ipynb)).

*Note: inside the package adaptive divergences might be referred as 'pseudo-Jensen-Snannon divergences' or 'pJSD'.*