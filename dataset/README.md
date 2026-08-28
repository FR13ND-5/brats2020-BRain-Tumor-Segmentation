# Dataset: BraTS2020 (pre-sliced HDF5)

Download from Kaggle:
[www.kaggle.com/datasets/awsaf49/brats2020-training-data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

About 7 GB zipped and 15 GB extracted: 57,195 HDF5 slice files from 369
patient volumes.

## Download & extract

Via browser: download the zip from the Kaggle page above, place it in this
folder. Or via the Kaggle CLI:

```bash
pip install kaggle          # needs ~/.kaggle/kaggle.json API token
kaggle datasets download -d awsaf49/brats2020-training-data -p dataset/
```

Then extract into `dataset/brats2020/`:

```bash
unzip dataset/brats2020-training-data.zip -d dataset/brats2020/
```

## Required layout

The code expects exactly this structure (checked by `test_smoke.py`):

```text
dataset/
├── README.md                        <- this file
├── brats2020-training-data.zip      <- optional, can be deleted after extraction
└── brats2020/
    ├── BraTS20 Training Metadata.csv
    └── BraTS2020_training_data/
        └── content/
            └── data/
                ├── volume_1_slice_0.h5
                ├── volume_1_slice_1.h5
                ├── ...                      (57,195 files: volume_{V}_slice_{S}.h5)
                ├── volume_369_slice_154.h5
                ├── meta_data.csv            (slice_path, target, volume, slice)
                ├── name_mapping.csv         (BraTS IDs <-> original datasets)
                └── survival_info.csv        (patient age / survival days)
```

If you extract somewhere else, pass `--data-dir /path/to/.../content/data`
to `train.py` and `inference.py`.

## File format

Each `volume_{V}_slice_{S}.h5` holds one 240×240 axial slice:

- `image`: shape (240, 240, 4), float64. The four MRI modalities as
  channels, in order T1, T1ce, T2, FLAIR, already z-score normalized per
  volume (values roughly −0.6 … 8; no further normalization needed).
- `mask`: shape (240, 240, 3), uint8, values {0, 1}. One-hot channels for
  the three tumor classes: NCR/NET (label 1), ED (label 2), ET (label 4).
  The channels are mutually disjoint. Many edge slices have an all-zero mask
  (no tumor); they are kept, so models also learn tumor-free anatomy.

Volumes are numbered 1 to 369; each contributes about 155 slices. The train/val
split in `src/data.py` groups slices by volume number before splitting, so
no patient ever appears in both sets.
