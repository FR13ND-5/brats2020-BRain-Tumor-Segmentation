# BraTS2020 Brain Tumor Segmentation: Architecture Comparison

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-AMP%20enabled-76B900?logo=nvidia&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-BraTS2020-blue)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-purple)

This repository trains four segmentation architectures on BraTS2020 and
compares them: U-Net, Attention U-Net, UNet++, and a TransUNet-style hybrid.
All four are written from scratch in PyTorch and trained one after another
under the same conditions, sharing a data split, a loss, an optimizer, a set
of metrics, and a single training loop. A difference in the results can
therefore only come from the architecture.

## About the task

This is semantic segmentation of brain tumors on 2D axial MRI slices. For
every pixel of a 240×240 slice, the model predicts which tumor tissue (if
any) it belongs to.

The input has 4 channels, one per MRI modality: T1, T1ce (contrast
enhanced), T2, and FLAIR. Each modality makes different tissue properties
visible, so all four go into the network together. The output is 3 binary
masks, one per tumor sub-region: NCR/NET (necrotic and non-enhancing tumor
core, BraTS label 1), ED (peritumoral edema, label 2), and ET (enhancing
tumor, label 4). Those three channels are disjoint one-hot labels, checked
against the data itself, and not the nested WT/TC/ET regions that some BraTS
papers use.

## The four architectures

| Model | Params | Key mechanism | What changes |
| --- | --- | --- | --- |
| U-Net | 31.0M | Plain skip connections | Encoder features go to the decoder untouched |
| Attention U-Net | 31.4M | Attention gates on skips | The decoder decides how much of each skip to let through |
| UNet++ | 9.2M | Nested dense skip pathways | Skips pass through intermediate conv nodes first |
| TransUNet (style) | 64.9M | Transformer at the bottleneck | Every position can attend to every other position |

The longer version:

1. U-Net (Ronneberger et al., 2015) is the baseline encoder-decoder. Skip
   connections restore the spatial detail that pooling throws away, but they
   copy everything, relevant or not.
2. Attention U-Net (Oktay et al., 2018) keeps that backbone and adds an
   additive attention gate on each skip, driven by the decoder signal. The
   gate suppresses encoder activations it scores as irrelevant before the
   concatenation happens.
3. UNet++ (Zhou et al., 2018) replaces each plain skip with a chain of
   nested conv nodes, so encoder features are refined in stages instead of
   jumping straight across the semantic gap to the decoder. It is the
   smallest model here because its filter counts start at 32 rather than 64.
4. The TransUNet-style model (after Chen et al., 2021) runs a CNN encoder
   for local features, flattens the 15×15 bottleneck map into 225 tokens,
   pushes them through a 4-layer Transformer encoder, and then restores
   resolution with a U-Net decoder and its skips. Global context costs it
   double the parameters of U-Net.

Every model file carries a docstring describing its mechanism, which is
where the pros and cons discussion in the report starts.

## Repository structure

```text
.
├── train.py               # entry point: train one model
├── inference.py           # entry point: evaluate a checkpoint + save predictions
├── compare.py             # entry point: aggregate all runs into a report table
├── test_smoke.py          # fast sanity check of models, data, split, metrics
├── requirements.txt
├── src/
│   ├── data.py            # BraTSSliceDataset, patient-wise split, augmentation, dataloaders
│   ├── losses.py          # DiceLoss, DiceBCELoss (0.5 Dice + 0.5 BCE)
│   ├── metrics.py         # Dice / IoU / sensitivity / specificity, per class,
│   │                      #   accumulated over the whole epoch (honest on empty slices)
│   ├── engine.py          # the one shared train/val loop: AMP, early stopping,
│   │                      #   checkpointing, JSON logging, curve plots
│   └── models/
│       ├── __init__.py    # model registry: adding a model = one file + one dict entry
│       ├── unet.py        # also holds the DoubleConv block shared by all models
│       ├── attention_unet.py
│       ├── unetpp.py
│       └── transunet.py
├── dataset/               # not in git; see dataset/README.md for download & layout
└── outputs/               # one folder per trained model; logs and images are
                           #   tracked in git, checkpoints (*.pth) are not
```

## Setup

```bash
# 1. environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # torch, h5py, numpy, matplotlib, tqdm

# 2. dataset: download from Kaggle and extract (details in dataset/README.md)
#    https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

# 3. sanity check before any long training
python3 test_smoke.py
```

Training really wants a CUDA GPU. The code switches on mixed precision (AMP)
when CUDA is available and falls back to CPU otherwise, just slowly.

## Dataset

Source: [BraTS2020 training data on Kaggle (awsaf49)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data).
The BraTS2020 challenge volumes come pre-sliced into 57,195 HDF5 files from
369 patient volumes, about 155 axial slices each, roughly 15 GB extracted.

Each `volume_{V}_slice_{S}.h5` contains:

| Key | Shape | Dtype | Content |
| --- | --- | --- | --- |
| `image` | (240, 240, 4) | float64 | T1, T1ce, T2, FLAIR, already z-score normalized |
| `mask` | (240, 240, 3) | uint8 | binary one-hot: NCR/NET, ED, ET |

The code expects the data at
`dataset/brats2020/BraTS2020_training_data/content/data/` and takes a
`--data-dir` override. Download and extraction instructions, plus the exact
directory layout, are in [dataset/README.md](dataset/README.md).

The train/val split works on patient volumes, never on individual slices.
Adjacent slices of one patient are nearly identical, so splitting by slice
would leak training data straight into validation. With the fixed seed 42
the split comes out at 295 train / 74 val volumes (45,725 / 11,470 slices),
and it stays identical for every model.

## Usage

### 1. Train (one model per run)

```bash
python3 train.py --model unet
python3 train.py --model attention_unet
python3 train.py --model unetpp
python3 train.py --model transunet
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | (required) | `unet` \| `attention_unet` \| `unetpp` \| `transunet` |
| `--epochs` | 30 | max epochs (early stopping usually ends sooner) |
| `--batch-size` | 32 | |
| `--lr` | 1e-4 | Adam learning rate |
| `--patience` | 5 | early-stop after N epochs without val-Dice improvement |
| `--data-dir` | `dataset/.../data` | HDF5 folder |
| `--output-dir` | `outputs` | results root |
| `--num-workers` | 8 | dataloader workers |
| `--seed` | 42 | keep identical across models for a fair comparison |
| `--limit` | off | cap files per split; quick sanity run: `--limit 200 --epochs 2` |

The training setup is identical for all four models: Dice+BCE loss, Adam,
AMP, flip and rot90 augmentation on the training set only, and the best
checkpoint chosen by mean validation Dice.

### 2. Evaluate and visualize

```bash
python3 inference.py --model unet          # uses outputs/unet/unet_best.pth
```

This recomputes every metric on the unseen validation patients, both pooled
over all pixels and per patient (mean and standard deviation of Dice across
the 74 validation volumes, which is how BraTS results normally get
reported). It also times inference in ms/slice and saves side-by-side
`FLAIR | ground truth | prediction` images for tumor-containing slices.

### 3. Compare all trained models

```bash
python3 compare.py
```

This writes a markdown table covering pooled and per-patient Dice, per-class
Dice, IoU, sensitivity, specificity, params, speed, epochs, and s/epoch,
together with val-Dice curves for all four models on one axis. The tables in
the report come from here.

## Outputs

```text
outputs/
└── <model>/
    ├── <model>_best.pth    # best-val-Dice checkpoint (weights + metrics)
    ├── train_log.json      # run config + full per-epoch metrics, written every epoch
    ├── curves.png          # loss + Dice/IoU curves for this run
    ├── test_metrics.json   # from inference.py: pooled + per-patient metrics, params, ms/slice
    └── predictions.png     # from inference.py: qualitative results
outputs/comparison.md       # from compare.py: cross-model table
outputs/comparison.png      # from compare.py: val Dice curves, all models
```

## Results

Scores on the 74 held-out validation patients, best checkpoint per model,
threshold 0.5. Produced by `python3 compare.py`; the full numbers sit in
`outputs/comparison.md` and each model's `test_metrics.json`.

| Model | Params (M) | Val Dice | Patient Dice (mean±std) | Dice NCR/NET | Dice ED | Dice ET | IoU | ms/slice | Epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 31.0 | 0.7942 | 0.7016±0.1756 | 0.7366 | 0.8014 | 0.8446 | 0.6609 | 1.71 | 17 |
| Attention U-Net | 31.4 | 0.7967 | 0.7029±0.1796 | 0.7398 | 0.8078 | 0.8423 | 0.6641 | 1.88 | 17 |
| UNet++ | **9.2** | **0.7976** | **0.7050±0.1732** | 0.7380 | **0.8081** | **0.8466** | **0.6656** | 2.96 | 20 |
| TransUNet (style) | 64.9 | 0.7799 | 0.6776±0.1802 | 0.7277 | 0.7799 | 0.8322 | 0.6412 | 2.67 | 19 |

UNet++ finishes on top, and it gets there with 9.2M parameters against
U-Net's 31.0M. The margin is 0.003 patient Dice while the per-patient
standard deviation is about 0.17, so the three CNNs are better described as
tied than ranked. What the parameter count buys is efficiency rather than
accuracy, and UNet++ pays for its nested nodes at inference time, where it
is the slowest of the four.

The attention gates were close to free and did close to nothing: 0.001
patient Dice over plain U-Net for an extra 0.4M parameters. With only three
tumor classes on fairly well-centered anatomy, there may not be much
irrelevant skip content left for a gate to suppress.

The TransUNet-style model is the clearest signal in the table. It has twice
the parameters of U-Net and scores about 0.025 Dice lower. Trained from
scratch on 46k slices it has no pretraining to lean on, which lines up with
the original TransUNet paper leaning on ImageNet-pretrained weights.

Per-class difficulty comes out in the same order for all four models: ET is
easiest, ED sits in the middle, and NCR/NET is hardest. Every run used early
stopping with patience 5 and settled on a best checkpoint somewhere between
epoch 12 and 15. One caveat on speed: the wall-clock epoch times in the logs
are not comparable across models, because the GPU was shared with an
unrelated training job at the time. The ms/slice column was measured in one
sequential session and is the fair comparison.

## Pretrained weights

The training logs, metrics, curves, and prediction images are all committed
under `outputs/`. The checkpoints are too large for the repository, so they
live in the
[v1.0 release](https://github.com/FR13ND-5/brats2020-BRain-Tumor-Segmentation/releases/tag/v1.0)
instead.

To reproduce the reported scores without retraining, drop a checkpoint into
the matching output folder and run inference:

```bash
mkdir -p outputs/unetpp
curl -L -o outputs/unetpp/unetpp_best.pth \
  https://github.com/FR13ND-5/brats2020-BRain-Tumor-Segmentation/releases/download/v1.0/unetpp_best.pth
python3 inference.py --model unetpp
```

| File | Model | Size |
| --- | --- | --- |
| `unet_best.pth` | U-Net | 119 MB |
| `attention_unet_best.pth` | Attention U-Net | 120 MB |
| `unetpp_best.pth` | UNet++ | 36 MB |
| `transunet_best.pth` | TransUNet (style) | 248 MB |

Each file is a `torch.save` dict holding `model`, `epoch`, `state_dict`, and
`val_metrics`. Load it with `torch.load(path, weights_only=True)` and hand
`state_dict` to the matching `get_model(name)`.

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, MIDL 2018
- Zhou et al., *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*, DLMIA 2018
- Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, 2021
- Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)*, IEEE TMI 2015
