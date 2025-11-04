# Representing Subgrid-Scale Cloud Effects in a Radiation Parameterization using Machine Learning: MLe-radiation v1.0

[![DOI](https://zenodo.org/badge/1070938076.svg)](https://doi.org/10.5281/zenodo.17280639)


This repository contains the code for the ML-enhanced radiation scheme which is based on [RTE+RRTMGP](https://github.com/earth-system-radiation/rte-rrtmgp) used in the [ICON Model](https://gitlab.dkrz.de/icon/icon-model). The corresponding paper is submitted to the Journal Geoscientific Model Development. 

The corresponding paper is available on arxiv as preprint
>  Hafner, K., Shamekh, S., Bertoli, G., Lauer, A., Pincus, R., Savre, J., & Eyring, V., 2025, Representing Subgrid-Scale Cloud Effects in a Radiation Parameterization using Machine Learning: MLe-radiation v1.0 [https://doi.org/10.48550/arXiv.2510.05963](https://doi.org/10.48550/arXiv.2510.05963)



If you want to use this repository, you can start by executing
```
conda env create -f environment_ml.yml
conda activate hafner1_ml_rad
```
for training and evaluation the ML-based part.

If you want to use [pyrte-rrtmgp](https://github.com/earth-system-radiation/pyRTE-RRTMGP), activate the following environment
```
conda env create -f environment_pyrte.yml
conda activate hafner1_pyrte_rrtmgp
```

# Repository content
- [evaluation](evaluation) contains some functions for prediction and evaluation
- [models](models) contains the NN architecture including preprocessing layer
- [nn_config](nn_config) contains the configuration of all NNs
- [plotter](plotter) contains plotting functions
- [preprocessing](preprocessing) contains the normalization file and data loader
- [utils](utils) contains some helper functions
- [train_jsc_cloudy.py](train_coarse_levante.py) contains the training script
- [eval_jsc_cloudy.py](eval_coarse_levante.py) contains the evaluation script
- [pyrte_on_coarse_grained_data.py](pyrte_on_coarse_grained_data.py) script to tun pyrte+rrtmgp on coarse grained data for reference
- [config.py](config.py) contains some general routines that are used for the training and evaluation script such as reading config files, loading data, creating an instance of the NN
- [data_distribution.ipynb](data_distribution.ipynb) used to calculate and plot the distributions of variables in the simulations (Figure 2, 3, A1, B1)
- [combined_eval_plots.ipynb](combined_eval_plots.ipynb) used to calculate statisitcs and plot the results (Figure 4, 5, C1)

# Previous Work
The code is partialy based on previous work on an [ML-based radiaiton emulator](https://github.com/EyringMLClimateGroup/hafner24jgrml_MLradiationemulation_offline) which hase been published:

> Hafner, K., Iglesias-Suarez, F., Shamekh, S., Gentine, P., Giorgetta, M. A., Pincus, R., & Eyring, V. (2025). Interpretable machine learning-based radiation emulation for ICON. *Journal of Geophysical Research: Machine Learning and Computation*, 2, e2024JH000501. [https://doi.org/10.1029/2024JH000501](https://doi.org/10.1029/2024JH000501)
