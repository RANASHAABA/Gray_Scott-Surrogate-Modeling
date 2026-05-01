
# Spline-Based Neural Surrogates for the 2D Gray–Scott Reaction–Diffusion System

This repository contains the implementation of all surrogate models developed
for the paper:

> **Spline-Based Neural Surrogates for the 2D Gray–Scott Reaction–Diffusion System**
> Rana Shaaban, Zaynab Al Haj

---

## Overview

We investigate data-driven surrogate modeling of the 2D Gray–Scott
reaction–diffusion system using a range of architectures, from classical
linear methods to hybrid neural operator models. All experiments are
conducted on the maze regime of the
[The Well](https://github.com/PolymathicAI/the_well) benchmark dataset.

---
## Models

| Model | Description |
|---|---|
| **DMD** | Dynamic Mode Decomposition in a POD-reduced latent space. Linear baseline. |
| **PCA + KAN** | KAN surrogate operating in a 200-dimensional PCA latent space. |
| **AE + KAN** | Convolutional autoencoder encoder with KAN latent predictor. Intermediate iteration. |
| **FNO** | Fourier Neural Operator operating directly on the full spatial fields. |
| **FNO-KAN (pointwise)** | FNO with per-layer pointwise KAN reaction blocks, motivated by operator splitting of the Gray–Scott equations. |
| **FNO-KAN (dual-domain)** | FNO backbone with a dual-domain KAN head combining a spectral KAN path and a spatial depthwise convolution path. |



## Dataset

All models are trained and evaluated on the **maze regime**
(`f = 0.029`, `k = 0.057`) of the Gray–Scott dataset from
[The Well](https://github.com/PolymathicAI/the_well).

We use 9 trajectories in total: **5 train / 2 validation / 2 test**.
Data is downloaded from HuggingFace:

```python
from the_well.data import WellDataset

ds = WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="gray_scott_reaction_diffusion",
    well_split_name="train",
    include_filters=["maze"],
)
```

---

## Requirements

All notebooks are designed to run on **Google Colab** with a T4 GPU.

```bash
pip install efficient_kan h5py torch torchvision matplotlib numpy tqdm the_well
```

The KAN implementation uses the
[EfficientKAN](https://github.com/Blealtan/efficient-kan) library.

---

## Results

| Model | Params | VRMSE |
|---|---|---|
| DMD | — | — |
| PCA + KAN | — | — |
| FNO | 1.18M | **0.0106** |
| FNO-KAN (pointwise) | 1.35M | 0.0108 |
| FNO-KAN (dual-domain) | 2.06M | 0.0135 |

Full results and analysis are reported in the paper.
