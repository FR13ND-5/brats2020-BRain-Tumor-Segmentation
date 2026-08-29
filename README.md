# BraTS2020 Brain Tumor Segmentation: Architecture Comparison

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-AMP%20enabled-76B900?logo=nvidia&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-BraTS2020-blue)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-purple)

Four segmentation architectures (U-Net, Attention U-Net, UNet++, and a
TransUNet-style hybrid) implemented from scratch in PyTorch and trained one
after another under identical conditions: same data split, same loss, same
optimizer, same metrics, same training loop. Any difference in results
therefore comes from the architecture itself. The point of the project is to
compare the four on accuracy, cost, and mechanism.

## About the task

This is semantic segmentation of brain tumors on 2D axial MRI slices. For
every pixel of a 240×240 slice, the model predicts which tumor tissue (if
any) it belongs to.

The input has 4 channels, one per MRI modality: T1, T1ce (contrast
enhanced), T2, and FLAIR. Each modality makes different tissue properties
visible, so all four are fed to the network together. The output is 3 binary
masks, one per tumor sub-region: NCR/NET (necrotic and non-enhancing tumor
core, BraTS label 1), ED (peritumoral edema, label 2), and ET (enhancing
tumor, label 4). These channels are disjoint one-hot labels. We checked this
directly against the data; they are not the nested WT/TC/ET regions used in
some BraTS papers.

## The four architectures

| Model | Params | Key mechanism | In one line |
| --- | --- | --- | --- |
| U-Net | 31.0M | Plain skip connections | Copies encoder features straight to the decoder at each scale |
| Attention U-Net | 31.4M | Attention gates on skips | The decoder gates the skips, suppressing regions it considers irrelevant |
| UNet++ | 9.2M | Nested dense skip pathways | Encoder features are refined through intermediate nodes before the decoder sees them |
| TransUNet (style) | 64.9M | Transformer at the bottleneck | Self-attention gives every position a global receptive field |

In short:

1. U-Net (Ronneberger et al., 2015) is the baseline encoder-decoder. Skip
   connections restore the spatial detail lost to pooling, but they copy
   everything, relevant or not.
2. Attention U-Net (Oktay et al., 2018) keeps the same backbone but passes
   each skip through an additive attention gate driven by the decoder
   signal. Irrelevant encoder activations are suppressed before
   concatenation.
3. UNet++ (Zhou et al., 2018) replaces each plain skip with a chain of
   nested conv nodes that gradually close the semantic gap between shallow
   encoder features and deep decoder features. It is the smallest model here
   because its filter counts start at 32 instead of 64.
4. The TransUNet-style model (after Chen et al., 2021) uses a CNN encoder
   for local features, flattens the 15×15 bottleneck map into 225 tokens,
   runs them through a 4-layer Transformer encoder, and restores resolution
   with a U-Net decoder plus skips. It buys global context at the cost of
   double the parameters.

Each model's source file has a docstring restating its mechanism, which is
where the pros/cons analysis in the report starts from.

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

Training wants a CUDA GPU. Mixed precision (AMP) is used when CUDA is
available; everything falls back to CPU otherwise, just slowly.

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

The train/val split is done by patient volume, never by slice. Adjacent
slices of one patient are nearly identical, so a slice-level split would
leak training data into validation. With the fixed seed 42 the split is 295
train / 74 val volumes (45,725 / 11,470 slices), and it is identical for
every model.

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

Training details, identical for all models: Dice+BCE loss, Adam, AMP,
flip/rot90 augmentation on the training set only, best checkpoint selected
by mean validation Dice.

### 2. Evaluate + visualize

```bash
python3 inference.py --model unet          # uses outputs/unet/best.pth
```

Recomputes all metrics on the unseen validation patients, both pooled over
all pixels and per patient (mean and standard deviation of Dice across the
74 validation volumes, which is how BraTS results are usually reported).
It also measures inference speed in ms/slice and saves side-by-side
`FLAIR | ground truth | prediction` images for tumor-containing slices.

### 3. Compare all trained models

```bash
python3 compare.py
```

Prints and saves a markdown table (Dice pooled and per patient, per-class
Dice, IoU, sensitivity, specificity, params, speed, epochs, s/epoch) plus
overlaid val-Dice curves for all models. The report tables come from here.

## Outputs

```text
outputs/
└── <model>/
    ├── best.pth            # best-val-Dice checkpoint (weights + metrics)
    ├── train_log.json      # run config + full per-epoch metrics, written every epoch
    ├── curves.png          # loss + Dice/IoU curves for this run
    ├── test_metrics.json   # from inference.py: pooled + per-patient metrics, params, ms/slice
    └── predictions.png     # from inference.py: qualitative results
outputs/comparison.md       # from compare.py: cross-model table
outputs/comparison.png      # from compare.py: val Dice curves, all models
```

## Results

Scores on the 74 held-out validation patients, best checkpoint per model
(threshold 0.5). Generated by `python3 compare.py`; full numbers are in
`outputs/comparison.md` and per-model `test_metrics.json`.

| Model | Params (M) | Val Dice | Patient Dice (mean±std) | Dice NCR/NET | Dice ED | Dice ET | IoU | ms/slice | Epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-Net | 31.0 | 0.7942 | 0.7016±0.1756 | 0.7366 | 0.8014 | 0.8446 | 0.6609 | 1.71 | 17 |
| Attention U-Net | 31.4 | 0.7967 | 0.7029±0.1796 | 0.7398 | 0.8078 | 0.8423 | 0.6641 | 1.88 | 17 |
| UNet++ | **9.2** | **0.7976** | **0.7050±0.1732** | 0.7380 | **0.8081** | **0.8466** | **0.6656** | 2.96 | 20 |
| TransUNet (style) | 64.9 | 0.7799 | 0.6776±0.1802 | 0.7277 | 0.7799 | 0.8322 | 0.6412 | 2.67 | 19 |

What the numbers say:

- UNet++ gets the best score of the four with 3.4x fewer parameters than
  U-Net, though it is also the slowest at inference because of its nested
  intermediate nodes.
- The three CNNs land within 0.004 patient Dice of each other while the
  per-patient standard deviation is about 0.17. At this data scale the three
  should be read as tied; which patients are hard matters far more than
  which of these architectures you pick.
- The TransUNet-style model trails by about 0.025 Dice despite having twice
  the parameters. Trained from scratch on 46k slices, the transformer
  bottleneck has no pretraining to lean on, which matches the original
  TransUNet paper's reliance on ImageNet-pretrained weights.
- The per-class ordering is identical for every model: ET is easiest, then
  ED, and NCR/NET is hardest.
- All models trained with early stopping (patience 5) and picked their best
  checkpoint at epoch 12 to 15. Wall-clock epoch times in the logs are not
  comparable across models because the GPU was shared with another training
  job; ms/slice above was measured in a single sequential session and is the
  fairer speed comparison.

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, MIDL 2018
- Zhou et al., *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*, DLMIA 2018
- Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, 2021
- Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)*, IEEE TMI 2015
