# DA6401 Assignment 2: Multi-Stage Visual Perception Pipeline (PyTorch)

## Overview

This project implements a **VGG11-based** visual perception stack on **Oxford-IIIT Pet**: **37-way breed classification**, **head bounding-box regression** \([x_c, y_c, w, h]\) in **pixel coordinates** at 224×224, **U-Net–style trimap segmentation**, and a **unified multi-task** model that loads **`classifier.pth`**, **`localizer.pth`**, and **`unet.pth`**. Training uses **ImageNet-style normalization**; the autograder-facing API matches the course skeleton (`VGG11`, `CustomDropout`, `IoULoss`, `MultiTaskPerceptionModel`).

## Links

- **Weights & Biases Report (public):** https://api.wandb.ai/links/ma24m011-iit-madras/eu156tt2  
- **GitHub Repository:** https://github.com/Ind-Sharma/DL_ass2

## Usage

Run commands from the **`da6401_assignment_2`** directory (so imports resolve):

```bash
cd da6401_assignment_2
```

### Training (single-task)

Classification (saves **best** checkpoint by **validation macro-F1**):

```bash
python train.py --task cls --data_root path/to/oxford-iiit-pet --epochs 30 --batch_size 16 --lr 1e-4 --save classifier.pth
```

Localization (optional: warm-start encoder from a trained classifier):

```bash
python train.py --task loc --data_root path/to/oxford-iiit-pet --epochs 30 --batch_size 16 --lr 1e-4 --save localizer.pth --encoder_ckpt classifier.pth
```

Segmentation:

```bash
python train.py --task seg --data_root path/to/oxford-iiit-pet --epochs 30 --batch_size 8 --lr 1e-4 --save unet.pth --encoder_ckpt classifier.pth
```
