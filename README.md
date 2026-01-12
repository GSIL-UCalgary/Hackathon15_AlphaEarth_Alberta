# Hackathon15_AlphaEarth_Alberta

This repository contains a multi-sensor semantic segmentation pipeline for evaluating AlphaEarth dataset potential relative to Sentinel-2 data using Alberta 2020 land cover classification.

## 📋 Overview

The project implements a comprehensive training pipeline for multi-sensor semantic segmentation with:
- Support for Landsat-8, Sentinel-2, and AlphaEarth sensors
- Multiple model architectures (BasicUNet, SwinUNet, SepViTUNet, etc.)
- Comprehensive metrics and experiment tracking with Weights & Biases
- Automatic mixed precision training and advanced data augmentation

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Clone the repository
git clone https://github.com/GSIL-UCalgary/Hackathon15_AlphaEarth_Alberta.git
cd Hackathon15_AlphaEarth_Alberta

# Create and activate conda environment (recommended)
conda create -n alphaearth python=3.8
conda activate alphaearth

# Install dependencies
pip install -r requirements.txt

---
```

### 2. Data Processing

1- Download LULC map from
  [Open Canada](https://open.canada.ca/data/en/dataset/ee1580ab-a23d-4f86-a09b-79763677eb47/resource/f1ba2faa-ff10-4526-815a-c57b99eef1bb)

2- Save the downloaded LULC map in `GroundTruth_Landsat_Canada` folder

1- Run preprocessing scripts
```
# Clip Landsat8 ground truth
python preprocessing/clip_Landsat8_30m_GT.py

# Extract patches for all datasets
python preprocessing/extract_patches_alldatasets.py
```


---
<table>
  <tr>
    <td align="center" valign="top">
      <img src="asset/Sentinel2.png" width="250" height="250" style="object-fit:contain;"/>
      <b>Sentinel-2</b>
    </td>
    <td align="center" valign="top">
      <img src="asset/Landsat8.png" width="250" height="250" style="object-fit:contain;"/>
      <b>Landsat-8</b>
    </td>
    <td align="center" valign="top">
      <img src="asset/AlphaEarth.png" width="250" height="250" style="object-fit:contain;"/>
      <b>AlphaEarth</b>
    </td>
    <td align="center" valign="top">
      <img src="asset/GT_Alberta_2020.png" width="250" height="250" style="object-fit:contain;"/>
      <b>Ground Truth</b>
    </td>
    <td align="center" valign="top">
      <img src="asset/colro index.png" width="250"/><br/>
      <b>Color Index</b>
    </td>
  </tr>
</table>


---
5- run extract_patches_alldatasets.py to get train, validation, and test patches of Landsat-8, Sentinel-2, and AlphaEarth datasets. It creates train_val_test_patches folder with the following structure:
```
train_val_test_patches/
├── LC_remapped.tif
├── LC_filtered.tif
├── label_mapping_metadata.json
├── multisensor_dataset_config.json
├── abundance_maps/
│   ├── class_0_abundance.tif
│   ├── class_1_abundance.tif
│   ├── ...
│   └── class_N_abundance.tif
├── patches/
│   ├── train/
│   │   ├── landsat8/
│   │   │   └── img/
│   │   │       ├── class_<class_id>_patch_<id>.tif
│   │   │       └── ...
│   │   ├── sentinel2/
│   │   │   └── img/
│   │   │       ├── class_<class_id>_patch_<id>.tif
│   │   │       └── ...
│   │   ├── alphaearth/
│   │   │   └── img/
│   │   │       ├── class_<class_id>_patch_<id>.tif
│   │   │       └── ...
│   │   └── labels/
│   │       ├── filtered/
│   │       │   ├── class_<class_id>_patch_<id>.tif
│   │       │   └── ...
│   │       └── unfiltered/
│   │           ├── class_<class_id>_patch_<id>.tif
│   │           └── ...
│   ├── val/
│   │   └── (same structure as `train`)
│   └── test/
│       └── (same structure as `train`)

```
