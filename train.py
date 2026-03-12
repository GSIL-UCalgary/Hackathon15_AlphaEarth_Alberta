"""
Multi-sensor semantic segmentation training pipeline
Supports multiple models with comprehensive metrics and Weights & Biases logging
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
from datetime import datetime
import json
from pathlib import Path
import copy
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
import argparse
import yaml
import platform
import rasterio
import time 
import pdb
# Import your models
from models import (
<<<<<<< HEAD
    MIMUNet,
    FocalUNet,
    SepViTUNet,
    SwinUNetWrapper, 
    CATUNet, 
    TwinsUNet, 
    BasicUNet, 
    HRNetWrapper,
    MambaHSISegWrapper,
    SSRNForSegmentation,
    ConvNeXtForSegmentation, 
    Global_superxiel_model,
    SimpleViTSegmentation,
    ImageHyperConnectionTransformer,
    ImageHyperConnectionTransformer_spec_spa,
    ImageHyperConnectionTransformer_mhc,
    MambaHSI, # ClusterMamba_aboundance
    ParallelGraphMHCSegNet
=======
    MIMUNet, FocalUNet, SepViTUNet, SwinUNetWrapper, 
    CATUNet, TwinsUNet, BasicUNet, HRNetWrapper,MambaHSISegWrapper, ImageHyperConnectionTransformerWrapper, 
    SSRNForSegmentation, ConvNeXtForSegmentation, Global_superxiel_model, ViTForSegmentation, AttentionDeepLabV3Plus
>>>>>>> 64040deaa131247df5240dbcfd55693b79946b81
)

start= time.time()
# Enable memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Optional: Reduce TensorFlow memory usage if you have it installed
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== WANDB API KEY SETUP ====================
def setup_wandb_api_key(api_key_path=None, force_login=False):
    """Setup WandB API key from file or environment, or login interactively"""
    
    # Check if already logged in (better check)
    try:
        if not force_login and wandb.api.api_key:
            print("✓ WandB already logged in")
            return True
    except:
        pass
    
    # Default path if not specified
    if api_key_path is None:
        # Try common locations
        possible_paths = [
            'wandb_api_key.txt',
            '.wandb_api_key.txt',
            'config/wandb_api_key.txt',
            '../wandb_api_key.txt',
            os.path.expanduser('~/.wandb_api_key.txt'),
            os.path.expanduser('~/.config/wandb/api_key'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                api_key_path = path
                break
    
    # Read API key from file if path exists
    if api_key_path and os.path.exists(api_key_path):
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            # Set environment variable
            os.environ['WANDB_API_KEY'] = api_key
            print(f" Loaded WandB API key from: {api_key_path}")
            return True
        except Exception as e:
            print(f" Could not read WandB API key from {api_key_path}: {e}")
    
    # Check if API key is already in environment
    if 'WANDB_API_KEY' in os.environ:
        print(" Using WandB API key from environment variable")
        return True
    
    # Try to get API key from wandb config file
    wandb_config_path = os.path.expanduser('~/.netrc')
    if os.path.exists(wandb_config_path):
        try:
            import netrc
            secrets = netrc.netrc(wandb_config_path)
            api_key = secrets.authenticators('api.wandb.ai')[2]
            if api_key:
                os.environ['WANDB_API_KEY'] = api_key
                print(" Loaded WandB API key from ~/.netrc")
                return True
        except:
            pass
    
    # If no API key found, try to login interactively
    print("No WandB API key found. Attempting interactive login...")
    try:
        wandb.login()
        print(" WandB login successful!")
        return True
    except Exception as e:
        print(f" WandB login failed: {e}")
        print("You can continue without WandB or set up API key manually.")
        return False

# Setup WandB API key at the start
setup_wandb_api_key()

class MultisensorDataset(Dataset):
    """Dataset for multi-sensor semantic segmentation"""
    
    def __init__(self, root_dir, sensor_name, split='train', label_type='filtered', transform=None):
        """
        Args:
            root_dir: Root directory with patches (e.g., "./train_val_test_patches_128x128_purity_threshold_0.3_v2")
            sensor_name: Name of sensor ('landsat8', 'sentinel2', 'alphaearth')
            split: 'train', 'val', or 'test'
            label_type: 'filtered' or 'unfiltered'
            transform: Optional transforms
        """
        self.root_dir = Path(root_dir)
        self.sensor_name = sensor_name
        self.split = split
        self.label_type = label_type
        self.transform = transform
        
        # CORRECT PATHS based on your structure:
        # Images: root_dir/patches/split/label_type/sensor_name/img/
        # Labels: root_dir/patches/split/label_type/labels/label_type/
        
        self.img_dir = self.root_dir /  split / label_type / sensor_name / 'img'
        self.label_dir = self.root_dir /  split / label_type / 'labels' / label_type
        
        print(f"Looking for images in: {self.img_dir}")
        print(f"Looking for labels in: {self.label_dir}")
        
        # Get all patch files
        self.img_files = sorted(list(self.img_dir.glob('*.tif')))
        self.label_files = sorted(list(self.label_dir.glob('*.tif')))
        
        print(f"Found {len(self.img_files)} image files")
        print(f"Found {len(self.label_files)} label files")
        
        # Verify matching files
        assert len(self.img_files) == len(self.label_files), \
            f"Number of images ({len(self.img_files)}) and labels ({len(self.label_files)}) don't match!"
        
        print(f"Loaded {len(self.img_files)} patches for {sensor_name} ({split})")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image patch
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        
        # Read GeoTIFF files using rasterio
        
        with rasterio.open(img_path) as src:
            image = src.read()  # Shape: (C, H, W)
        # This assumes your UInt8 data is scaled from original values to 0-255
        image = image.astype(np.uint8)

        with rasterio.open(label_path) as src:
            label = src.read(1)  # Shape: (H, W)
        
        # Convert to tensors
        image = torch.from_numpy(image).float() # Already [0, 1]
        label = torch.from_numpy(label).long()
        
        # Apply transforms if any
        if self.transform:
            # For transforms that work on single tensors, apply only to image
            # Or create custom transforms that handle both image and label
            image = self.transform(image)
            # Note: Some transforms like random flips should be applied to both
            # You need custom transform classes for that
        
        return image, label

class SegmentationMetrics:
    """Calculate comprehensive segmentation metrics"""
    
    def __init__(self, num_classes, ignore_index=-99, class_names=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        
    def update(self, pred, target):
        """Update confusion matrix"""
        pred = pred.flatten()
        target = target.flatten()
        
        # Remove ignored pixels
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        cm = confusion_matrix(
            target.cpu().numpy(), 
            pred.cpu().numpy(), 
            labels=range(self.num_classes)
        )
        self.confusion_matrix += cm
    
    def compute(self):
        """Compute all metrics"""
        metrics = {}
        
        # Per-class metrics
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        
        # Avoid division by zero
        epsilon = 1e-10
        
        # Per-class precision
        precision_per_class = tp / (tp + fp + epsilon)
        
        # Per-class recall
        recall_per_class = tp / (tp + fn + epsilon)
        
        # Per-class F1-score
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)
        
        # IoU (Jaccard Index) per class
        iou_per_class = tp / (tp + fp + fn + epsilon)
        
        # Overall metrics
        metrics['overall_accuracy'] = np.sum(tp) / np.sum(self.confusion_matrix + epsilon)
        metrics['mean_precision'] = np.mean(precision_per_class)
        metrics['mean_recall'] = np.mean(recall_per_class)
        metrics['mean_f1'] = np.mean(f1_per_class)
        metrics['mean_iou'] = np.mean(iou_per_class)
        
        # Frequency weighted IoU
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + epsilon)
        metrics['freq_weighted_iou'] = np.sum(freq * iou_per_class)
        
        # Per-class metrics for detailed analysis
        for i in range(self.num_classes):
            metrics[f'{self.class_names[i]}_iou'] = iou_per_class[i]
            metrics[f'{self.class_names[i]}_f1'] = f1_per_class[i]
            metrics[f'{self.class_names[i]}_precision'] = precision_per_class[i]
            metrics[f'{self.class_names[i]}_recall'] = recall_per_class[i]
            metrics[f'{self.class_names[i]}_support'] = int(tp[i] + fn[i])
        
        # Kappa coefficient
        total = np.sum(self.confusion_matrix)
        if total > 0:
            pe = np.sum(np.sum(self.confusion_matrix, axis=0) * np.sum(self.confusion_matrix, axis=1)) / (total ** 2)
            metrics['kappa'] = (metrics['overall_accuracy'] - pe) / (1 - pe + epsilon)
        else:
            metrics['kappa'] = 0.0
        
        return metrics
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


def create_model(model_name, sensor_name, dataset_config, training_config):
    """Create model based on name and sensor configuration"""
    print(f"The {model_name} model is being trained using {sensor_name} dataset")
    print(f"training_config ['hidden_dim']: {training_config ['hidden_dim']}")
    # --------------------------------------------------
    # Input channels per sensor
    # --------------------------------------------------
    bands_config = {
        'landsat8': 6,
        'sentinel2': 10,
        'alphaearth': 64
    }

    input_channels = bands_config[sensor_name]
    num_classes = 13  # fixed from your setup

    # --------------------------------------------------
    # Physical mHC Mamba
    # --------------------------------------------------
    if model_name == 'physical_mhc_mamba':
        # Define all hardcoded values here
        image_size = 224
        patch_size = 4
        depth = 4
        num_heads = 4
        mlp_ratio = 4.0
        dropout = 0.1
        n_layers = 4
        n_heads = 4 
        rate = 4 
        mask_ratio= 0.0
        dynamic = True
        embed_dim = training_config.get('hidden_dim', 384)
        print(f"physical_mhc_mamba config - n_layers: {n_layers}, embed_dim: {embed_dim}")
        # Store them in hyperparameters dict
        model_hyperparameters = {
            'physical_mhc_mamba_img_size': image_size,
            'physical_mhc_mamba_patch_size': patch_size,
            'physical_mhc_mamba_depth': depth,
            'physical_mhc_mamba_num_heads': num_heads,
            'physical_mhc_mamba_mlp_ratio': mlp_ratio,
            'physical_mhc_mamba_n_layers': n_layers,
            'physical_mhc_mamba_n_heads': n_heads,
            'physical_mhc_mamba_rate': rate,
            'physical_mhc_mamba_mask_ratio': mask_ratio,
            'physical_mhc_mamba_dynamic': dynamic,
            'physical_mhc_mamba_dropout': dropout,
            'physical_mhc_mamba_embed_dim': embed_dim
        }

        model = ImageHyperConnectionTransformer_mhc(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=input_channels,
        num_classes=num_classes,
        dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        rate=rate,
        dropout=dropout,
        mask_ratio=mask_ratio,
        dynamic=dynamic,
        sensor_name=sensor_name
                 )
        return model, model_hyperparameters
    # --------------------------------------------------
    # Parallel Graph-mHC SegNet
    # --------------------------------------------------
    elif model_name == 'ParallelGraphMHCSegNet':
        # Store them in hyperparameters dict
        image_size = 224
        in_channels=input_channels
        stem_dim=training_config.get('hidden_dim', 64)
        num_classes=num_classes
        mhc_block_dims=[64, 128, 256]
        mamba_blocks_per_stage=[2, 4, 6]
        patch_gcn_dims=[128, stem_dim]
        spatial_gcn_dims=[128, stem_dim]
        sigma=1.0
        spatial_stride=8
        knn_k=10
        fusion_strategy='cat'
        fusion_out_dim=256
        seg_head_dims=[128, 64]
        model_hyperparameters = {
            'ParallelGraphMHCSegNet_img_size': image_size,
            'ParallelGraphMHCSegNet_in_channels': in_channels,
            'ParallelGraphMHCSegNet_stem_dim': stem_dim,
            'ParallelGraphMHCSegNet_num_classes': num_classes,
            'ParallelGraphMHCSegNet_mhc_block_dims': mhc_block_dims,
            'ParallelGraphMHCSegNet_mamba_blocks_per_stage': mamba_blocks_per_stage,
            'ParallelGraphMHCSegNet_patch_gcn_dims': patch_gcn_dims,
            'ParallelGraphMHCSegNet_spatial_gcn_dims': spatial_gcn_dims,
            'ParallelGraphMHCSegNet_sigma': sigma,
            'ParallelGraphMHCSegNet_spatial_stride': spatial_stride,
            'ParallelGraphMHCSegNet_knn_k': knn_k,
            'ParallelGraphMHCSegNet_fusion_strategy': fusion_strategy,
            'ParallelGraphMHCSegNet_fusion_out_dim': fusion_out_dim,
            'ParallelGraphMHCSegNet_seg_head_dims': seg_head_dims
        }

        model = ParallelGraphMHCSegNet(
            in_channels=in_channels,
            stem_dim=stem_dim,
            num_classes=num_classes,
            mhc_block_dims=mhc_block_dims,
            mamba_blocks_per_stage=mamba_blocks_per_stage,
            patch_gcn_dims=patch_gcn_dims,
            spatial_gcn_dims=spatial_gcn_dims,
            sigma=sigma,
            spatial_stride=spatial_stride,
            knn_k=knn_k,
            fusion_strategy=fusion_strategy,
            fusion_out_dim=fusion_out_dim,
            seg_head_dims=seg_head_dims,
        )

        return model, model_hyperparameters
    # --------------------------------------------------
    # Cluster_MambaHSI
    # --------------------------------------------------

    elif model_name == 'Cluster_MambaHSI':
        print(f"input_channels: {input_channels}")
        return MambaHSI(in_channels=input_channels,
                        hidden_dim=64,
                        num_classes=num_classes, 
                        use_residual=True,
                        mamba_type='both',
                        token_num=4,
                        group_num=4,
                        use_att=True,
                        num_clusters=20*3,
                        sparsity_ratio=1.0,
                        attention_heads=4,
                        selection_mode='cluster')
 
    # --------------------------------------------------
    # HRNet
    # --------------------------------------------------
    elif model_name == 'HRNet':
        hrnet_config = {
            'in_channels': input_channels,
            'num_classes': num_classes
        }
        return HRNetWrapper(hrnet_config)

    # --------------------------------------------------
    # mHC_spec_spa_mamba
    # --------------------------------------------------
    elif model_name == 'mHC_spec_spa_mamba':
        return ImageHyperConnectionTransformer_spec_spa(
        image_size=224,
        patch_size=2,
        in_channels=input_channels,
        num_classes=num_classes,
        dim=32,
        n_layers=2,
        n_heads=2,
        rate=4,
        dropout=0.1,
        drop_path=0.0,
        mask_ratio=0.0,
        dynamic=True
    )

    # --------------------------------------------------
    # SSRN
    # --------------------------------------------------
    elif model_name == 'ssrn':
        return SSRNForSegmentation(
            in_channels=input_channels,
            num_classes=num_classes,
            msize=18,
            inter_size=64,
            downsample=2,
            apply_downsampling = True

        )
    # --------------------------------------------------
    # ConvNeXt
    # --------------------------------------------------
    elif model_name == 'convnext':
        convnext_config = {
            'in_chans': input_channels,
            'depths': [3, 3],
            'dims': [128, 128],
            'num_classes': num_classes,
            'patch_size': 1
        }
        return ConvNeXtForSegmentation(**convnext_config)
    # --------------------------------------------------
    # OBIA Mamba  
    # --------------------------------------------------
    elif model_name == 'OBIA_Mamba':
        return Global_superxiel_model(num_classes=num_classes, 
                                      num_superpixel=500, 
                                      dim=64, 
                                      d_conv=6, 
                                      in_channel=input_channels,
        )
    # --------------------------------------------------
    # ViT
    # --------------------------------------------------
<<<<<<< HEAD
    if model_name == 'ViT':
        # Define all hardcoded values here
        vit_img_size = 224
        vit_patch_size = 7
        vit_depth = 3
        vit_num_heads = 4
        vit_mlp_ratio = 4.0
        vit_dropout = 0.1
        vit_embed_dim = training_config.get('hidden_dim', 384)

        # Store them in hyperparameters dict
        model_hyperparameters = {
            'vit_img_size': vit_img_size,
            'vit_patch_size': vit_patch_size,
            'vit_depth': vit_depth,
            'vit_num_heads': vit_num_heads,
            'vit_mlp_ratio': vit_mlp_ratio,
            'vit_dropout': vit_dropout,
            'vit_embed_dim': vit_embed_dim
        }


=======
    elif model_name == 'ViT':
>>>>>>> 64040deaa131247df5240dbcfd55693b79946b81
        vit_config = {
            'img_size': vit_img_size,
            'in_chans': input_channels,
            'embed_dim': vit_embed_dim,
            'patch_size': vit_patch_size,
            'depth': vit_depth,
            'num_heads': vit_num_heads,
            'mlp_ratio': vit_mlp_ratio,
            'dropout': vit_dropout,
            'num_classes': num_classes
        }
        model = SimpleViTSegmentation(**vit_config)
        return model, model_hyperparameters
    # --------------------------------------------------
<<<<<<< HEAD
    # MambaHSISeg 
=======
    # Tseg
    # --------------------------------------------------
    elif model_name == 'Tseg':
        return AttentionDeepLabV3Plus(
            num_classes=num_classes,
            in_channels=input_channels,
            backbone='resnet50',
            pretrained=True
        )
    # --------------------------------------------------
    # MambaHSISeg (new addition)
>>>>>>> 64040deaa131247df5240dbcfd55693b79946b81
    # --------------------------------------------------
    elif model_name == 'MambaHSISeg':
        mamba_config = {
            'in_channels': input_channels,
            'num_classes': num_classes,
            'base_dim': 64,           # c1 dimension
            'mamba_type': 'both',     # 'spa', 'spe', or 'both'
            'token_num': 16,           # for SpeMamba
            'use_residual': True,
            'group_num': 16,
            'use_att': True,          # attention fusion for BothMamba
            'use_stem': True          # whether to use initial downsampling
        }
        return MambaHSISegWrapper(mamba_config)
    # --------------------------------------------------
    # SwinUNet
    # --------------------------------------------------
    elif model_name == 'SwinUNet':
        swin_config = {
            'img_size': 224,
            'in_channels': input_channels,
            'num_classes': num_classes,
            'embed_dim': 96,
            'depths': [1, 1, 2,1],
            'heads': [1, 2, 4, 8],
            'patch_size': 4,
            'window_size': 7
        }
        return SwinUNetWrapper(**swin_config)
    # --------------------------------------------------
    # Basic UNet
    # --------------------------------------------------
    elif model_name == 'BasicUNet':
        basic_config = {
            'in_channels': input_channels,
            'num_classes': num_classes,
            'stem_dim': 32,
            'stem_kernel': 3,
            'stem_padding': 1,
            'stem_downsampling': False,
            'dims': [32, 64, 128, 256],
            'depths': [1, 1, 2, 1],
        }
        return BasicUNet(basic_config)

    # --------------------------------------------------
    # Other models (unchanged)
    # --------------------------------------------------
    model_config = {}
    if model_name == 'MIMUNet':
        return MIMUNet(**model_config)
    elif model_name == 'FocalUNet':
        return FocalUNet(**model_config)
    elif model_name == 'SepViTUNet':
        return SepViTUNet(**model_config)
    elif model_name == 'CATUNet':
        return CATUNet(**model_config)
    elif model_name == 'TwinsUNet':
        return TwinsUNet(**model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_rgb_label_prediction(image_64band, label_map, pred_map, 
                              save_path=None, figsize=(18, 6)):
    """
    Plot RGB image (first 3 bands), ground truth label, and prediction side by side
    
    Args:
        image_64band: numpy array of shape (H, W, 64) or (64, H, W) - hyperspectral image
        label_map: numpy array of shape (H, W) with values -99, 0, 1, ..., 12
        pred_map: numpy array of shape (H, W) with predicted class values
        save_path: optional path to save the figure
        figsize: figure size tuple
    """
    CLASS_DEFINITIONS = {
        -99: {'name': 'Background', 'color': (0, 0, 0)},
        0: {'name': 'Temperate needleleaf forest', 'color': (0, 61, 0)},
        1: {'name': 'Sub-polar taiga forest', 'color': (148, 156, 112)},
        2: {'name': 'Temperate broadleaf forest', 'color': (20, 140, 61)},
        3: {'name': 'Mixed forest', 'color': (91, 117, 43)},
        4: {'name': 'Temperate shrubland', 'color': (179, 138, 51)},
        5: {'name': 'Temperate grassland', 'color': (225, 207, 138)},
        6: {'name': 'Polar grassland-lichen', 'color': (186, 212, 143)},
        7: {'name': 'Wetland', 'color': (107, 163, 138)},
        8: {'name': 'Cropland', 'color': (230, 174, 102)},
        9: {'name': 'Barren lands', 'color': (168, 171, 174)},
        10: {'name': 'Urban', 'color': (220, 33, 38)},
        11: {'name': 'Water', 'color': (76, 112, 163)},
        12: {'name': 'Snow/ice', 'color': (255, 250, 255)}
    }
    
    # Handle image dimension (channels first or last)
    
    
    # Extract first 3 bands for RGB visualization
    rgb_image = image_64band[:3, :, :].transpose(1, 2, 0)
    
    # Normalize RGB image for better visualization (per channel)
    rgb_normalized = np.zeros_like(rgb_image)
    for i in range(3):
        channel = rgb_image[:, :, i]
        vmin, vmax = np.percentile(channel[channel > 0], (2, 98))  # Use 2-98 percentile to avoid outliers
        rgb_normalized[:, :, i] = np.clip((channel - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Prepare colormap for labels
    class_values = sorted(CLASS_DEFINITIONS.keys())
    colors_rgb = [[c/255.0 for c in CLASS_DEFINITIONS[v]['color']] for v in class_values]
    cmap = ListedColormap(colors_rgb)
    
    # Create boundaries for labels
    boundaries = []
    for i, val in enumerate(class_values):
        if i == 0:
            boundaries.append(val - 0.5)
        boundaries.append(val + 0.5)
    norm = BoundaryNorm(boundaries, cmap.N)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. RGB Image (first 3 bands)
    ax1 = axes[0]
    im1 = ax1.imshow(rgb_normalized)
    ax1.set_title('RGB Image (Bands 1-3)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Ground Truth
    ax2 = axes[1]
    im2 = ax2.imshow(label_map, cmap=cmap, norm=norm, interpolation='none')
    ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Prediction
    ax3 = axes[2]
    im3 = ax3.imshow(pred_map, cmap=cmap, norm=norm, interpolation='none')
    ax3.set_title('Prediction', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Add colorbar for the label maps (shared)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cbar_ax, ticks=class_values)
    cbar.ax.set_yticklabels([CLASS_DEFINITIONS[v]['name'] for v in class_values], 
                            fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Land Cover Classes', fontsize=10)
    
    plt.suptitle('Hyperspectral Image Classification Results', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for colorbar
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig, axes

class Trainer:
    """Main training class with Config Saving"""
    
    def __init__(self, config):
        self.config = config
        self.split_image = config.get('split_image', False)
        self.margin = 5
        self.num_classes = config.get('num_classes', 13)

        # Remove the scaler setup
        self.scaler = None  # Explicitly set to None

        self.device = torch.device('cuda:0' if torch.cuda.device_count() > 1 else 'cpu')
        print(f"Using device: {self.device}")
        
        # Get label type (default to 'filtered' if not specified)
        label_type = config.get('label_type', 'filtered')
        
        # Create output directory with timestamp and label type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir']) / f"{config['model_name']}_{config['sensor_name']}_{label_type}_{timestamp}_{config['image_patch_size']}x{config['image_patch_size']}_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.margin = 5
        self.num_classes = config.get('num_classes', 13)


        # Load dataset config BEFORE saving configuration
        with open(config['dataset_config'], 'r') as f:
            self.dataset_config = json.load(f)
        
        # ========== ADD VERIFICATION CODE HERE ==========
        # Verify data root exists and has the correct structure
        data_root = Path(self.config['data_root'])
        label_type = self.config.get('label_type', 'filtered')
        sensor_name = self.config['sensor_name']

        print(f"\n🔍 Verifying dataset paths:")
        test_path = data_root / 'train' / label_type / sensor_name / 'img'
        print(f"  Should look for images in: {test_path}")
        if test_path.exists():
            print(f"  ✓ Path exists")
            sample_files = list(test_path.glob('*.tif'))[:3]
            if sample_files:
                print(f"  ✓ Found sample files: {[f.name for f in sample_files]}")
        else:
            print(f"  ✗ Path does not exist!")
        # ========== END OF VERIFICATION CODE ==========

        # Save configuration files immediately
        self.save_configuration()
        
        # Initialize wandb AFTER saving config
        if config['use_wandb']:
            wandb.init(
                project=config['wandb_project'],
                entity=config.get('wandb_entity', None),
                name=f"{config['model_name']}_{config['sensor_name']}_{label_type}_{timestamp}",
                config=config,
                tags=[config['model_name'], config['sensor_name'], "segmentation", label_type],
                dir=str(self.output_dir)  # Save wandb logs to experiment directory
            )
            print(f"WandB Run: {wandb.run.url}")
        
        # Create model - now using your specific config
        # Create model - now returns both model and hyperparameters
        self.model, model_hparams = create_model(
            config['model_name'],
            config['sensor_name'],
            dataset_config=self.dataset_config,
            training_config=self.config
        )
        # Add model hyperparameters to config
        self.config['model_hyperparameters'] = model_hparams
        self.model = self.model.to(self.device)
        
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {config['model_name']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Log model architecture to wandb
        if config['use_wandb']:
            wandb.watch(self.model, log="parameters", log_freq=100)
        
        # Loss function
        if config['loss_fn'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=config['ignore_index'],
                weight=config.get('class_weights', None)
            )
        else:
            raise ValueError(f"Unknown loss function: {config['loss_fn']}")
        
        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate']
                # weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate']
                # weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        
        # Scheduler
        warmup_epochs = config.get('warmup_epochs', 5)

        if config['scheduler'] == 'cosine':
            base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'] - warmup_epochs
            )
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[warmup_epochs]
            )

        elif config['scheduler'] == 'reduce_on_plateau':
            # No warmup for plateau — it's reactive and warmup would interfere
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
            )

        elif config['scheduler'] == 'step':
            base_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[warmup_epochs]
            )

        else:
            self.scheduler = None

        # Mixed precision training
        if config['use_amp']:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Get number of classes based on label type
        label_type = self.config.get('label_type', 'filtered')
        if label_type == 'filtered' and 'filtered_labels' in self.dataset_config:
            num_classes = len(self.dataset_config['filtered_labels'])
            # Get class names in the correct order for filtered labels
            class_names = []
            for label in self.dataset_config['filtered_labels']:
                class_names.append(self.dataset_config['class_names'].get(str(label), f"Class_{label}"))
        elif label_type == 'unfiltered' and 'unfiltered_labels' in self.dataset_config:
            num_classes = len(self.dataset_config['unfiltered_labels'])
            # Get class names in the correct order for unfiltered labels
            class_names = []
            for label in self.dataset_config['unfiltered_labels']:
                class_names.append(self.dataset_config['class_names'].get(str(label), f"Class_{label}"))
        else:
            # Fallback
            num_classes = self.config.get('num_classes', 13)
            class_names = [f"Class_{i}" for i in range(num_classes)]

        print(f"Number of classes for {label_type} dataset: {num_classes}")
        print(f"Class names: {class_names}")

        # Metrics
        self.metrics = SegmentationMetrics(
            num_classes=num_classes,
            ignore_index=config['ignore_index'],
            class_names=class_names
        )
        
        # Best model tracking
        self.best_val_iou = 0
        self.best_model_state = None
        
        # Save initial configuration again (with model info)
        self.save_configuration(with_model_info=True)
    
    def save_configuration(self, with_model_info=False):
        """Save configuration as both .txt and .json files"""
        
        # Save as JSON (machine-readable)
        config_json_path = self.output_dir / "config.json"
        with open(config_json_path, 'w') as f:
            # Convert any non-serializable objects
            serializable_config = {}
            for key, value in self.config.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    serializable_config[key] = value
                else:
                    serializable_config[key] = str(value)
            json.dump(serializable_config, f, indent=2)
        print(f"✓ Configuration saved as JSON: {config_json_path}")
        
        # Save as TXT (human-readable)
        config_txt_path = self.output_dir / "config.txt"
        with open(config_txt_path, 'w') as f:
            self._write_config_txt(f, with_model_info)
        print(f"✓ Configuration saved as TXT: {config_txt_path}")
    
    def _write_config_txt(self, file_obj, with_model_info=False):
        """Write configuration in human-readable format"""
        file_obj.write("=" * 80 + "\n")
        file_obj.write("TRAINING CONFIGURATION\n")
        file_obj.write("=" * 80 + "\n\n")
        
        file_obj.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file_obj.write(f"Output Directory: {self.output_dir}\n\n")
        
        # Model information
        file_obj.write("MODEL CONFIGURATION\n")
        file_obj.write("-" * 40 + "\n")
        file_obj.write(f"Model Name: {self.config['model_name']}\n")
        file_obj.write(f"Sensor Name: {self.config['sensor_name']}\n")
        file_obj.write(f"Label Type: {self.config.get('label_type', 'filtered')}\n") 
        # Get number of classes from filtered_labels or unfiltered_labels
        if hasattr(self, 'config') and 'label_type' in self.config:
            label_type = self.config['label_type']
            if label_type == 'filtered' and 'filtered_labels' in self.dataset_config:
                num_classes = len(self.dataset_config['filtered_labels'])
            elif label_type == 'unfiltered' and 'unfiltered_labels' in self.dataset_config:
                num_classes = len(self.dataset_config['unfiltered_labels'])
            else:
                # Fallback
                num_classes = 13
        else:
            num_classes = 13
        file_obj.write(f"Number of Classes: {num_classes}\n")
        file_obj.write(f"Ignore Index: {self.config.get('ignore_index', -99)}\n")
        file_obj.write(f"Class Weights: {self.config.get('class_weights', 'None')}\n")
        
        if with_model_info and hasattr(self, 'model'):
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            file_obj.write(f"Total Parameters: {total_params:,}\n")
            file_obj.write(f"Trainable Parameters: {trainable_params:,}\n")
        
        file_obj.write("\n")
        
        # Training parameters
        file_obj.write("TRAINING HYPERPARAMETERS\n")
        file_obj.write("-" * 40 + "\n")
        file_obj.write(f"Epochs: {self.config['epochs']}\n")
        file_obj.write(f"Batch Size: {self.config['batch_size']}\n")
        file_obj.write(f"Learning Rate: {self.config['learning_rate']}\n")
        file_obj.write(f"Weight Decay: {self.config['weight_decay']}\n")
        file_obj.write(f"Loss Function: {self.config['loss_fn']}\n")
        file_obj.write(f"Optimizer: {self.config['optimizer']}\n")
        file_obj.write(f"Scheduler: {self.config['scheduler']}\n")
        file_obj.write(f"Use AMP: {self.config['use_amp']}\n")
        file_obj.write(f"Number of Workers: {self.config['num_workers']}\n")
        file_obj.write(f"Save Every: {self.config['save_every']} epochs\n\n")
        
        # Dataset information
        file_obj.write("DATASET INFORMATION\n")
        file_obj.write("-" * 40 + "\n")
        # Get number of classes from filtered_labels or unfiltered_labels
        if 'filtered_labels' in self.dataset_config:
            num_classes = len(self.dataset_config['filtered_labels'])
        elif 'unfiltered_labels' in self.dataset_config:
            num_classes = len(self.dataset_config['unfiltered_labels'])
        elif 'num_classes' in self.dataset_config:
            num_classes = self.dataset_config['num_classes']
        else:
            # Fallback to config value
            num_classes = self.config.get('num_classes', 13)

        file_obj.write(f"Number of Classes: {num_classes}\n")
        file_obj.write(f"Window Size: {self.dataset_config['window_size']}\n")
        file_obj.write(f"Background Label: {self.dataset_config['background_label']}\n")
        file_obj.write(f"Sensor Bands: {self.dataset_config['sensors'][self.config['sensor_name']]['bands']}\n")
        
        # List all classes
        file_obj.write("\nCLASS MAPPING:\n")
        # Also fix the class mapping section to use the correct label list
        label_type = self.config.get('label_type', 'filtered')
        if label_type == 'filtered' and 'filtered_labels' in self.dataset_config:
            labels = self.dataset_config['filtered_labels']
        elif label_type == 'unfiltered' and 'unfiltered_labels' in self.dataset_config:
            labels = self.dataset_config['unfiltered_labels']
        else:
            # Fallback - try to determine from class_names length
            labels = list(range(len(self.dataset_config.get('class_names', {}))))

        for i, label in enumerate(labels):
            original_id = self.dataset_config['class_mapping'].get(str(label), label)
            class_name = self.dataset_config['class_names'].get(str(label), f"Class_{label}")
            file_obj.write(f"  Label {i} (new {label}) → Original {original_id} ({class_name})\n")
        file_obj.write("\n")
        
        # Dataset paths
        file_obj.write("DATASET PATHS\n")
        file_obj.write("-" * 40 + "\n")
        file_obj.write(f"Data Root: {self.config['data_root']}\n")
        file_obj.write(f"Dataset Config: {self.config['dataset_config']}\n")
        file_obj.write(f"Output Directory: {self.config['output_dir']}\n\n")
        
        # WandB configuration
        file_obj.write("WEIGHTS & BIASES CONFIGURATION\n")
        file_obj.write("-" * 40 + "\n")
        file_obj.write(f"Use WandB: {self.config['use_wandb']}\n")
        if self.config['use_wandb']:
            file_obj.write(f"WandB Project: {self.config['wandb_project']}\n")
            if self.config.get('wandb_entity'):
                file_obj.write(f"WandB Entity: {self.config['wandb_entity']}\n")
            if wandb.run:
                file_obj.write(f"WandB Run URL: {wandb.run.url}\n")
                file_obj.write(f"WandB Run ID: {wandb.run.id}\n")
        file_obj.write("\n")
        
        # Hardware information
        file_obj.write("HARDWARE INFORMATION\n")
        file_obj.write("-" * 40 + "\n")
        file_obj.write(f"Device: {self.device}\n")
        file_obj.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            file_obj.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            file_obj.write(f"CUDA Version: {torch.version.cuda}\n")
            file_obj.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        file_obj.write(f"PyTorch Version: {torch.__version__}\n")
        file_obj.write(f"Number of CPU cores: {os.cpu_count()}\n\n")
        
        # System information
        file_obj.write("SYSTEM INFORMATION\n")
        file_obj.write("-" * 40 + "\n")
        file_obj.write(f"OS: {platform.system()} {platform.release()}\n")
        file_obj.write(f"Python Version: {platform.python_version()}\n")
        file_obj.write(f"Machine: {platform.machine()}\n")
        file_obj.write(f"Processor: {platform.processor()}\n")
        
        file_obj.write("\n" + "=" * 80 + "\n")
        file_obj.write("END OF CONFIGURATION\n")
        file_obj.write("=" * 80 + "\n")
    
    def create_dataloaders(self):
        """Create train, val, and test dataloaders"""
        
        from torchvision import transforms
        # No augmentation for validation/test
        val_transform = None

        # Get label type from config
        label_type = self.config.get('label_type', 'filtered')
        print(f"\n📁 Loading {label_type.upper()} dataset from: {self.config['data_root']}")
        print(f"batch size: {self.config['batch_size']}")
        # Create datasets - paths are handled in MultisensorDataset
        train_dataset = MultisensorDataset(
            root_dir=self.config['data_root'],
            sensor_name=self.config['sensor_name'],
            split='train',
            label_type=label_type,
            # transform=train_transform
        )
        
        val_dataset = MultisensorDataset(
            root_dir=self.config['data_root'],
            sensor_name=self.config['sensor_name'],
            split='val',
            label_type=label_type,
            transform=val_transform
        )
        
        test_dataset = MultisensorDataset(
            root_dir=self.config['data_root'],
            sensor_name=self.config['sensor_name'],
            split='test',
            label_type=label_type,
            transform=val_transform
        )
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_epoch_loss = 0
        self.metrics.reset()
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        # Gradient accumulation settings
        accumulation_steps = 4  # Adjust based on memory
         # Zero gradients at the beginning
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)                       
            
            # For other models
            # outputs, loss2 = self.model(images)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels.long()) 

            total_loss = loss

            # Normalize loss for gradient accumulation
            total_loss = total_loss
            
            # Backward pass

            self.optimizer.zero_grad() 
            total_loss.backward()
            self.optimizer.step()


            
            # For logging, use the original loss value (before normalization)
            avg_loss = loss.item()  # This is the classification loss
            total_epoch_loss += avg_loss
            
            # Calculate predictions
            preds = outputs.argmax(dim=1)
            #plot_rgb_label_prediction(images.cpu().numpy()[0], labels.cpu().numpy()[0, :, :], preds.cpu().numpy()[0, :, :], 'label1.png')
            # Update metrics
            self.metrics.update(preds, labels)
            
            # Update progress bar
            pbar.set_postfix({'loss': avg_loss})
            
            # Log batch metrics to wandb
            if self.config['use_wandb'] and batch_idx % 10 == 0:
                step = (
                    epoch * len(train_loader.dataset)
                    + batch_idx * train_loader.batch_size
                )

                log_dict = {
                    'train/batch_loss': avg_loss,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                }
                
                wandb.log(log_dict, step=step)
        
        # Compute metrics
        metrics = self.metrics.compute()
        metrics['loss'] = total_epoch_loss / len(train_loader)
        
        
        
        # Log epoch metrics to wandb
        if self.config['use_wandb']:
            log_dict = {
                'train/epoch_loss': metrics['loss'],
                'train/mean_iou': metrics['mean_iou'],
                'train/mean_f1': metrics['mean_f1'],
                'train/overall_accuracy': metrics['overall_accuracy'],
                'train/mean_precision': metrics['mean_precision'],
                'train/mean_recall': metrics['mean_recall'],
                'train/kappa': metrics['kappa'],
                'epoch': epoch
            }
            
        
            
            wandb.log(log_dict)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader, epoch, mode='val'):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        self.metrics.reset()
        
        pbar = tqdm(val_loader, desc=f"{mode.capitalize()} Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Calculate split boundaries
            B, C, H, W = images.shape
            margin = self.margin
            h_mid = H // 2
            w_mid = W // 2
            
            if self.split_image:
                # Split image into 4 overlapping parts EXACTLY like training
                # Top-left
                x_part1 = images[:, :, :h_mid + margin, :w_mid + margin]
                y_part1 = labels[:, :h_mid + margin, :w_mid + margin]

                # Top-right
                x_part2 = images[:, :, :h_mid + margin, w_mid - margin:]
                y_part2 = labels[:, :h_mid + margin, w_mid - margin:]

                # Bottom-left
                x_part3 = images[:, :, h_mid - margin:, :w_mid + margin]
                y_part3 = labels[:, h_mid - margin:, :w_mid + margin]

                # Bottom-right
                x_part4 = images[:, :, h_mid - margin:, w_mid - margin:]
                y_part4 = labels[:, h_mid - margin:, w_mid - margin:]

                # Forward passes for all parts
                y_pred_part1 = self.model(x_part1)
                ls1 = self.criterion(y_pred_part1, y_part1.long())
                
                y_pred_part2 = self.model(x_part2)
                ls2 = self.criterion(y_pred_part2, y_part2.long())
                
                y_pred_part3 = self.model(x_part3)
                ls3 = self.criterion(y_pred_part3, y_part3.long())
                
                y_pred_part4 = self.model(x_part4)
                ls4 = self.criterion(y_pred_part4, y_part4.long())
                
                # Calculate average loss
                batch_loss = (ls1 + ls2 + ls3 + ls4) / 4
                total_loss += batch_loss.item()
                
                # Need to reconstruct full prediction for accurate metrics
                # Get number of classes from output shape
                num_classes = y_pred_part1.shape[1]
                
                # Initialize full logits tensor
                full_logits = torch.zeros(B, num_classes, H, W, device=self.device)
                count_map = torch.zeros(B, 1, H, W, device=self.device)
                
                # Reconstruct full prediction from parts
                # Top-left
                full_logits[:, :, :h_mid + margin, :w_mid + margin] += y_pred_part1
                count_map[:, :, :h_mid + margin, :w_mid + margin] += 1
                
                # Top-right
                full_logits[:, :, :h_mid + margin, w_mid - margin:] += y_pred_part2
                count_map[:, :, :h_mid + margin, w_mid - margin:] += 1
                
                # Bottom-left
                full_logits[:, :, h_mid - margin:, :w_mid + margin] += y_pred_part3
                count_map[:, :, h_mid - margin:, :w_mid + margin] += 1
                
                # Bottom-right
                full_logits[:, :, h_mid - margin:, w_mid - margin:] += y_pred_part4
                count_map[:, :, h_mid - margin:, w_mid - margin:] += 1
                
                # Average overlapping regions
                full_logits = full_logits / count_map
                
                # Get final predictions
                preds = full_logits.argmax(dim=1)
                
                # Update metrics with full reconstructed predictions
                self.metrics.update(preds, labels)
                
                # Update progress bar
                pbar.set_postfix({'loss': batch_loss.item()})
                
            else:
                # Original full-image validation
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                preds = outputs.argmax(dim=1)
                self.metrics.update(preds, labels)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        plot_rgb_label_prediction(images.cpu().numpy()[0], labels.cpu().numpy()[0, :, :], preds.cpu().numpy()[0, :, :], 'val_vis1.png')
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / len(val_loader)
        
        # Log metrics to wandb
        if self.config['use_wandb']:
            wandb.log({
                f'{mode}/loss': metrics['loss'],
                f'{mode}/mean_iou': metrics['mean_iou'],
                f'{mode}/mean_f1': metrics['mean_f1'],
                f'{mode}/overall_accuracy': metrics['overall_accuracy'],
                f'{mode}/mean_precision': metrics['mean_precision'],
                f'{mode}/mean_recall': metrics['mean_recall'],
                f'{mode}/kappa': metrics['kappa'],
                f'{mode}/freq_weighted_iou': metrics['freq_weighted_iou'],
                'epoch': epoch
            })
        
        return metrics
    
    def train(self):
        """Main training loop"""
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders()
        
        print(f"\nStarting training for {self.config['model_name']} on {self.config['sensor_name']}")
        print(f"Training samples:   {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples:       {len(test_loader.dataset)}")
        
        # Training history
        history = {
            'train_loss': [], 'train_iou': [], 'train_accuracy': [],
            'val_loss':   [], 'val_iou':   [], 'val_accuracy':   [],
            'test_metrics': None
        }
        
        # ------------------------------------------------------------------ #
        #  Epoch loop                                                          #
        # ------------------------------------------------------------------ #
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'='*60}")

            # ── Train ──────────────────────────────────────────────────────
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"mIoU: {train_metrics['mean_iou']:.4f}, "
                f"Acc: {train_metrics['overall_accuracy']:.4f}")

            # ── Validate ───────────────────────────────────────────────────
            val_metrics = self.validate(val_loader, epoch, mode='val')
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"mIoU: {val_metrics['mean_iou']:.4f}, "
                f"Acc: {val_metrics['overall_accuracy']:.4f}")

            # ── Scheduler step ─────────────────────────────────────────────
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mean_iou'])
                else:
                    self.scheduler.step()

            # ── Log LR to wandb ────────────────────────────────────────────
            if self.config['use_wandb']:
                wandb.log({
                    'train/learning_rate_epoch': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })

            # ── Update history ─────────────────────────────────────────────
            history['train_loss'].append(train_metrics['loss'])
            history['train_iou'].append(train_metrics['mean_iou'])
            history['train_accuracy'].append(train_metrics['overall_accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_iou'].append(val_metrics['mean_iou'])
            history['val_accuracy'].append(val_metrics['overall_accuracy'])

            # ── Save best model ────────────────────────────────────────────
            if val_metrics['mean_iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['mean_iou']
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                best_path = self.output_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_iou': self.best_val_iou,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, best_path)
                print(f"✓ Saved best model (mIoU: {self.best_val_iou:.4f}) to {best_path}")

                if self.config['use_wandb']:
                    wandb.log({
                        'best/epoch': epoch,
                        'best/val_iou': self.best_val_iou,
                        'best/val_accuracy': val_metrics['overall_accuracy']
                    })

            # ── Save periodic checkpoint ───────────────────────────────────
            if epoch % self.config['save_every'] == 0:
                ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_iou': val_metrics['mean_iou'],
                    'config': self.config
                }, ckpt_path)
                print(f"✓ Saved checkpoint (epoch {epoch}) to {ckpt_path}")

        # ------------------------------------------------------------------ #
        #  Test with best model                                                #
        # ------------------------------------------------------------------ #
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        print(f"\n{'='*60}")
        print("Testing on test set with best model")
        print(f"{'='*60}")
        test_metrics = self.validate(test_loader, epoch, mode='test')
        history['test_metrics'] = test_metrics

        print(f"\n📊 FINAL TEST RESULTS:")
        print(f"  Mean IoU:          {test_metrics['mean_iou']:.4f}")
        print(f"  Mean F1 Score:     {test_metrics['mean_f1']:.4f}")
        print(f"  Overall Accuracy:  {test_metrics['overall_accuracy']:.4f}")
        print(f"  Kappa:             {test_metrics['kappa']:.4f}")
        print(f"  Mean Precision:    {test_metrics['mean_precision']:.4f}")
        print(f"  Mean Recall:       {test_metrics['mean_recall']:.4f}")
        print(f"  Freq Weighted IoU: {test_metrics['freq_weighted_iou']:.4f}")

        # ── Save final model ───────────────────────────────────────────────
        final_model_path = self.output_dir / "final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'test_metrics': test_metrics,
            'history': history
        }, final_model_path)

        # ── Save metrics and history ───────────────────────────────────────
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)

        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # ── Save final config with results ─────────────────────────────────
        final_config_path = self.output_dir / "final_config.txt"
        with open(final_config_path, 'w') as f:
            self._write_config_txt(f, with_model_info=True)
            f.write("\n" + "=" * 80 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Best Validation IoU:  {self.best_val_iou:.4f}\n")
            f.write(f"Test Mean IoU:        {test_metrics['mean_iou']:.4f}\n")
            f.write(f"Test Overall Accuracy:{test_metrics['overall_accuracy']:.4f}\n")
            f.write(f"Test Kappa:           {test_metrics['kappa']:.4f}\n")
            f.write(f"Training Completed:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # ── WandB final logging ────────────────────────────────────────────
        if self.config['use_wandb']:
            wandb.log({
                'test/final_iou':      test_metrics['mean_iou'],
                'test/final_f1':       test_metrics['mean_f1'],
                'test/final_accuracy': test_metrics['overall_accuracy'],
                'test/final_kappa':    test_metrics['kappa'],
                'test/final_precision':test_metrics['mean_precision'],
                'test/final_recall':   test_metrics['mean_recall']
            })

            artifact = wandb.Artifact(
                name=f"{self.config['model_name']}_{self.config['sensor_name']}",
                type="model",
                description=f"Best model trained on {self.config['sensor_name']} data",
                metadata={
                    'test_iou':      test_metrics['mean_iou'],
                    'test_accuracy': test_metrics['overall_accuracy'],
                    'best_val_iou':  self.best_val_iou,
                    'epochs':        self.config['epochs']
                }
            )
            artifact.add_file(str(final_model_path))
            wandb.log_artifact(artifact)
            wandb.finish()

        print(f"\n✅ Training completed!")
        print(f"   Model saved to:   {final_model_path}")
        print(f"   Metrics saved to: {metrics_path}")
        print(f"   Config saved to:  {final_config_path}")
        print(f"   Best val mIoU:    {self.best_val_iou:.4f}")

        return test_metrics

def parse_args():
    """Parse command line arguments - only config file needed"""
    parser = argparse.ArgumentParser(description='Train multi-sensor segmentation model')
    
    # Only config file argument
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    
    return parser.parse_args()

def load_config_from_yaml(config_path):
    """Load configuration from YAML file and normalize paths"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get patch size
    patch_size = config.get('image_patch_size', 128)
    
    # Expand placeholders in paths
    path_keys = ['data_root', 'dataset_config', 'output_dir']
    for key in path_keys:
        if key in config and isinstance(config[key], str):
            # Replace {patch_size} with actual value
            config[key] = config[key].format(patch_size=patch_size)
            # Convert backslashes to forward slashes
            config[key] = config[key].replace('\\', '/')
    
    return config

def parse_class_weights(class_weights_str):
    """Parse class weights from string"""
    if class_weights_str is None or class_weights_str == "null":
        return None
    
    if os.path.exists(class_weights_str):
        # Load from JSON file
        with open(class_weights_str, 'r') as f:
            weights = json.load(f)
        # Convert to tensor if it's a list
        if isinstance(weights, list):
            return torch.tensor(weights, dtype=torch.float32)
        return weights
    else:
        # Check if it's a JSON string
        if class_weights_str.startswith('[') and class_weights_str.endswith(']'):
            try:
                weights = json.loads(class_weights_str)
                return torch.tensor(weights, dtype=torch.float32)
            except:
                pass
        
        # Parse comma-separated list
        try:
            weights = [float(w) for w in class_weights_str.split(',')]
            return torch.tensor(weights, dtype=torch.float32)
        except:
            print(f"Warning: Could not parse class weights: {class_weights_str}")
            return None
        
        
def main():
    """Main function to run training - YAML config only"""
    
    args = parse_args()
    
    # Load configuration from YAML file
    config = load_config_from_yaml(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Set seed from config
    set_seed(config.get('seed', 42))
    
    # Ensure required fields are present
    required_fields = ['model_name', 'sensor_name', 'label_type', 
                       'data_root', 'dataset_config', 'output_dir',
                       'epochs', 'batch_size', 'learning_rate']

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' not found in config file")
    
    # Set defaults for optional fields if not provided
    config.setdefault('weight_decay', 1e-5)
    config.setdefault('loss_fn', 'cross_entropy')
    config.setdefault('optimizer', 'adamw')
    config.setdefault('scheduler', 'reduce_on_plateau')
    config.setdefault('ignore_index', -99)
    config.setdefault('class_weights', None)
    config.setdefault('use_amp', True)
    config.setdefault('num_workers', 4)
    config.setdefault('save_every', 10)
    config.setdefault('use_wandb', True)
    config.setdefault('wandb_project', 'AlphaEarth_Alberta_2020')
    config.setdefault('wandb_entity', 'saeid_taleghani')
    config.setdefault('seed', 42)
    
    # Handle special cases
    if config.get('scheduler') == 'none':
        config['scheduler'] = None
    
    # Parse class weights if provided as string
    if isinstance(config['class_weights'], str):
        config['class_weights'] = parse_class_weights(config['class_weights'])
    
    # Setup WandB login if enabled
    if config['use_wandb']:
        print("\n" + "="*60)
        print("Setting up Weights & Biases")
        print("="*60)
        
        # Setup API key
        setup_wandb_api_key()
        
        # Try to initialize wandb to see if it works
        try:
            # Test if we can initialize wandb
            test_run = wandb.init(
                project=config['wandb_project'],
                entity=config.get('wandb_entity', None),
                name="test_init",
                mode="disabled"  # Start in disabled mode for testing
            )
            test_run.finish()
            print("✓ WandB initialization successful")
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}")
            print("Disabling WandB for this run.")
            config['use_wandb'] = False
        
        print("="*60 + "\n")
    
    # Create trainer and train
    trainer = Trainer(config)
    test_metrics = trainer.train()
    
    End= time.time() 
    print(f"\nTotal training time: {(End - start) / 60:.2f} minutes")
    return test_metrics

if __name__ == '__main__':

    main()
    