# Code and data for "Upscaling and downscaling Monte Carlo ensembles with generative models" (Scheiter et al., GJI, 2022)

This repository contains code and data supporting the work published by Scheiter et al. (2022). It contains implementations of different GAN varieties (GAN, WGAN, and DCGAN) and a their application to a synthetic and a geophysical example. If you make use of any of the code or models in this repository, please cite this publication:

Scheiter, M., Valentine, A., Sambridge, M., 2022. [Upscaling and downscaling Monte Carlo ensembles with generative models](https://doi.org/10.1093/gji/ggac100), Geophys. J. Int., 230(2):916--931.

The repository also contains tomography data from Mousavi et al. (2021). If you use these in their original or GAN-reproduced version, please cite their paper:

Mousavi, S., Tkalcic, H., Hawkins, R., & Sambridge, M., 2021. [Lowermost mantle shear-velocity structure from hierarchical trans-dimensional Bayesian tomography](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JB021557), J. Geophys. Res. Solid Earth, 126, e2020JB021557.

## GAN codes
The codes used in this study to train the GANs can be found in the folder `synthetic_example/`.
- `generative_model.py` has an implementation of all GAN variants used in the study (standard GAN, Wasserstein GAN, and Deep Convolutional GAN).
- `run_gans.py` provides an example on how to train a GAN based on the settings of the synthetic example.

## Synthetic example
The directory `synthetic_example/` contains everything needed to reproduce the synthetic example in section 3 of the paper. The full data processing pipeline can be reproduced:
- `run_gans.py` will use the original data provided in `datasets/` and train GANs on them as specified in the paper (see section 3 and Appendix B1). The trained GANs are stored in `trained_gans/`. (Script runs ~15 hours)
- `process_data.py` will prepare the data to make plots, i.e. draw fake data from the GANs, calculate integrals, etc. The processed data can be found in `processed_data/`. (Script runs ~1 hour)
- `plot_figures.py` will reproduce Figures 1-4 of the paper and store them in the directory `figures/`.

## Geophysical example
The directory `geophysical_example/` contains the trained GANs from section 4 and everything needed to reproduce the figures of the geophysical example based on the study of Mousavi et al. (2021).
- `plot_figures.py` can be used to reproduce Figures 5-9 of the paper and store them in the directory `figures/`. All data necessary for these plots can be found in `plotting_data/`.
- `trained_gans/` contains the trained GANs of the Australia patch from section 4.2 and the 16 patches from section 4.4.

## Recommended package versions
All codes in this repository have been tested with the following package versions:
- Python 3.7.10
- Numpy 1.20.2
- Matplotlib 3.4.1
- PyTorch 1.8.0
- Cartopy 0.18.0
- tqdm 4.32.2
