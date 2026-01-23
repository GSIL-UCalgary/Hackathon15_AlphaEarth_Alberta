"""
Multi-sensor semantic segmentation training pipeline
Supports multiple models with comprehensive metrics and Weights & Biases logging
"""

import os
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
# Import your models
from models import (
    MIMUNet, FocalUNet, SepViTUNet, SwinUNetWrapper, 
    CATUNet, TwinsUNet, BasicUNet, HRNetWrapper,MambaHSISegWrapper, ImageHyperConnectionTransformerWrapper, 
    SSRNForSegmentation, ConvNeXtForSegmentation, Global_superxiel_model, ViTForSegmentation, AttentionDeepLabV3Plus
)

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
            root_dir: Root directory with patches
            sensor_name: Name of sensor ('landsat8', 'sentinel2', 'alphaearth')
            split: 'train', 'val', or 'test'
            label_type: 'filtered' or 'unfiltered'
            transform: Optional transforms
        """
        self.root_dir = Path(root_dir)
        self.sensor_name = sensor_name
        self.split = split
        self.transform = transform
        
        # Paths
        self.img_dir = self.root_dir / split / sensor_name / 'img'
        self.label_dir = self.root_dir / split / 'labels' / label_type
        
        # Get all patch files
        self.img_files = sorted(list(self.img_dir.glob('*.tif')))
        self.label_files = sorted(list(self.label_dir.glob('*.tif')))
        
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
        image = image.astype(np.float32) / 255.0

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


def create_model(model_name, sensor_name, config):
    """Create model based on name and sensor configuration"""
    print(f"The {model_name} model is being trained using {sensor_name} dataset")

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
    # HRNet
    # --------------------------------------------------
    if model_name == 'HRNet':
        hrnet_config = {
            'in_channels': input_channels,
            'num_classes': num_classes
        }
        return HRNetWrapper(hrnet_config)
    # --------------------------------------------------
    # mHC_cluster
    # --------------------------------------------------
    elif model_name == 'mHC_cluster':
        return ImageHyperConnectionTransformerWrapper(
            in_channels=input_channels,       # 64 for alphaearth
            num_classes=num_classes,          # 13 classes
            image_size=128,                   # Input image size
            dim=64,                           # ✅ Increased from 12 to 64 (must be divisible by 4)
            n_layers=4,                       # ✅ Reduced layers for memory efficiency
            n_heads=4,                        # Number of attention heads (64/4 = 16 per head)
            rate=4,                           # ✅ Reduced from 4 to 2 for memory
            patch_size=1,                     # No downsampling for segmentation
            dropout=0.1,
            drop_path=0.1,
            mask_ratio=0.0,                   # ✅ Disable masking for segmentation
            dynamic=True                      # Dynamic hyper-connections
        )
    # --------------------------------------------------
    # SSRN
    # --------------------------------------------------
    elif model_name == 'ssrn'
        return SSRNForSegmentation(
            in_channels=input_channels,
            num_classes=num_classes,
            msize=18,
            inter_size=49
        ):
    # --------------------------------------------------
    # ConvNeXt
    # --------------------------------------------------
    elif model_name == 'convnext':
        convnext_config = {
            'in_chans': input_channels,
            'depths': [3, 3],
            'dims': [96, 192],
            'num_classes': num_classes,
            'patch_size': 1
        }
        return ConvNeXtForSegmentation(**convnext_config)
    # --------------------------------------------------
    # OBIA Mamba
    # --------------------------------------------------
    elif model_name == 'OBIA_Mamba':
        return Global_superxiel_model(num_classes=num_classes, num_superpixel=500, dim=64, d_conv=6, in_channels=input_channels,
        )
    # --------------------------------------------------
    # ViT
    # --------------------------------------------------
    elif model_name == 'ViT':
        vit_config = {
            'img_size': 128,
            'patch_size': 16,
            'in_chans': input_channels,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'num_classes': num_classes
        }
        return ViTForSegmentation(**vit_config)
    # --------------------------------------------------
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
    # --------------------------------------------------
    elif model_name == 'MambaHSISeg':
        mamba_config = {
            'in_channels': input_channels,
            'num_classes': num_classes,
            'base_dim': 32,           # c1 dimension
            'mamba_type': 'both',     # 'spa', 'spe', or 'both'
            'token_num': 4,           # for SpeMamba
            'use_residual': True,
            'group_num': 4,
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
            'embed_dim': 32,
            'depths': [2, 2, 6, 2],
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


class Trainer:
    """Main training class with Config Saving"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Get label type (default to 'filtered' if not specified)
        label_type = config.get('label_type', 'filtered')
        
        # Create output directory with timestamp and label type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_dir']) / f"{config['model_name']}_{config['sensor_name']}_{label_type}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset config BEFORE saving configuration
        with open(config['dataset_config'], 'r') as f:
            self.dataset_config = json.load(f)
        
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
        self.model = create_model(
            config['model_name'],
            config['sensor_name'],
            self.dataset_config
        ).to(self.device)
        
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
        elif config['loss_fn'] == 'focal':
            try:
                from focal_loss.focal_loss import FocalLoss
                self.criterion = FocalLoss(ignore_index=config['ignore_index'])
            except ImportError:
                print("Warning: focal-loss-torch not installed. Using cross entropy instead.")
                self.criterion = nn.CrossEntropyLoss(ignore_index=config['ignore_index'])
        else:
            raise ValueError(f"Unknown loss function: {config['loss_fn']}")
        
        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        
        # Scheduler
        if config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs']
            )
        elif config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
            )
        elif config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Mixed precision training
        if config['use_amp']:
            self.scaler = torch.amp.GradScaler('cuda')  # Updated API
        else:
            self.scaler = None
        
        # Get class names from your dataset config
        class_names = []
        if 'class_names' in self.dataset_config:
            # Extract class names in order
            class_names = [self.dataset_config['class_names'][str(i)] for i in range(self.dataset_config['num_classes'])]
        else:
            class_names = [f"Class_{i}" for i in range(self.dataset_config['num_classes'])]
        
        # Metrics - using 13 classes from your config
        self.metrics = SegmentationMetrics(
            num_classes=self.dataset_config['num_classes'],  # Should be 13
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
        file_obj.write(f"Number of Classes: {self.dataset_config['num_classes']}\n")
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
        file_obj.write(f"Number of Classes: {self.dataset_config['num_classes']}\n")
        file_obj.write(f"Window Size: {self.dataset_config['window_size']}\n")
        file_obj.write(f"Background Label: {self.dataset_config['background_label']}\n")
        file_obj.write(f"Sensor Bands: {self.dataset_config['sensors'][self.config['sensor_name']]['bands']}\n")
        
        # List all classes
        file_obj.write("\nCLASS MAPPING:\n")
        for i in range(self.dataset_config['num_classes']):
            original_id = self.dataset_config['class_mapping'].get(str(i), i)
            class_name = self.dataset_config['class_names'].get(str(i), f"Class_{i}")
            file_obj.write(f"  Label {i} → Original {original_id} ({class_name})\n")
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
        
        # Data augmentation for training
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomVerticalFlip(p=0.5),
        #     transforms.RandomRotation(degrees=30),
        # ])
        
        # No augmentation for validation/test
        val_transform = None

        # Get label type from config
        label_type = self.config.get('label_type', 'filtered')

        # Create datasets
        train_dataset = MultisensorDataset(
        root_dir=self.config['data_root'],
        sensor_name=self.config['sensor_name'],
        split='train',
        label_type=label_type,
        #transform=train_transform
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
        total_loss = 0
        self.metrics.reset()
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast('cuda'):  # Updated API
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            preds = outputs.argmax(dim=1)
            self.metrics.update(preds, labels)
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log batch metrics to wandb
            if self.config['use_wandb'] and batch_idx % 10 == 0:
                step = epoch * len(train_loader) + batch_idx
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'step': step
                })
        
        # Compute metrics
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / len(train_loader)
        
        # Log epoch metrics to wandb
        if self.config['use_wandb']:
            wandb.log({
                'train/epoch_loss': metrics['loss'],
                'train/mean_iou': metrics['mean_iou'],
                'train/mean_f1': metrics['mean_f1'],
                'train/overall_accuracy': metrics['overall_accuracy'],
                'train/mean_precision': metrics['mean_precision'],
                'train/mean_recall': metrics['mean_recall'],
                'train/kappa': metrics['kappa'],
                'epoch': epoch
            })
        
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
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Update metrics
            preds = outputs.argmax(dim=1)
            self.metrics.update(preds, labels)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
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
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Training history for saving
        history = {
            'train_loss': [], 'train_iou': [], 'train_accuracy': [],
            'val_loss': [], 'val_iou': [], 'val_accuracy': [],
            'test_metrics': None
        }
        
        # Training loop
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"mIoU: {train_metrics['mean_iou']:.4f}, "
                  f"Acc: {train_metrics['overall_accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader, epoch, mode='val')
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"mIoU: {val_metrics['mean_iou']:.4f}, "
                  f"Acc: {val_metrics['overall_accuracy']:.4f}")
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mean_iou'])
                else:
                    self.scheduler.step()
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_iou'].append(train_metrics['mean_iou'])
            history['train_accuracy'].append(train_metrics['overall_accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_iou'].append(val_metrics['mean_iou'])
            history['val_accuracy'].append(val_metrics['overall_accuracy'])
            
            # Save best model
            if val_metrics['mean_iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['mean_iou']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                
                # Save checkpoint
                checkpoint_path = self.output_dir / f"best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_iou': self.best_val_iou,
                    'val_metrics': val_metrics,
                    'config': self.config
                }, checkpoint_path)
                print(f"✓ Saved best model (mIoU: {self.best_val_iou:.4f}) to {checkpoint_path}")
                
                # Log best metrics to wandb
                if self.config['use_wandb']:
                    wandb.log({
                        'best/epoch': epoch,
                        'best/val_iou': self.best_val_iou,
                        'best/val_accuracy': val_metrics['overall_accuracy']
                    })
            
            # Save regular checkpoint
            if epoch % self.config['save_every'] == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'val_iou': val_metrics['mean_iou'],
                    'config': self.config
                }, checkpoint_path)
        
        # Load best model for testing
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Test on test set
        print(f"\n{'='*60}")
        print("Testing on test set with best model")
        print(f"{'='*60}")
        test_metrics = self.validate(test_loader, epoch, mode='test')
        history['test_metrics'] = test_metrics
        
        print(f"\n📊 FINAL TEST RESULTS:")
        print(f"  Mean IoU:        {test_metrics['mean_iou']:.4f}")
        print(f"  Mean F1 Score:   {test_metrics['mean_f1']:.4f}")
        print(f"  Overall Accuracy: {test_metrics['overall_accuracy']:.4f}")
        print(f"  Kappa:           {test_metrics['kappa']:.4f}")
        print(f"  Mean Precision:  {test_metrics['mean_precision']:.4f}")
        print(f"  Mean Recall:     {test_metrics['mean_recall']:.4f}")
        print(f"  Freq Weighted IoU: {test_metrics['freq_weighted_iou']:.4f}")
        
        # Save final model
        final_model_path = self.output_dir / "final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'test_metrics': test_metrics,
            'history': history
        }, final_model_path)
        
        # Save test metrics and history
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save final configuration with results
        final_config_path = self.output_dir / "final_config.txt"
        with open(final_config_path, 'w') as f:
            self._write_config_txt(f, with_model_info=True)
            f.write("\n" + "="*80 + "\n")
            f.write("FINAL RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Best Validation IoU: {self.best_val_iou:.4f}\n")
            f.write(f"Test Mean IoU: {test_metrics['mean_iou']:.4f}\n")
            f.write(f"Test Overall Accuracy: {test_metrics['overall_accuracy']:.4f}\n")
            f.write(f"Test Kappa: {test_metrics['kappa']:.4f}\n")
            f.write(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Log final test metrics to wandb
        if self.config['use_wandb']:
            wandb.log({
                'test/final_iou': test_metrics['mean_iou'],
                'test/final_f1': test_metrics['mean_f1'],
                'test/final_accuracy': test_metrics['overall_accuracy'],
                'test/final_kappa': test_metrics['kappa'],
                'test/final_precision': test_metrics['mean_precision'],
                'test/final_recall': test_metrics['mean_recall']
            })
            
            # Save model artifact to wandb
            artifact = wandb.Artifact(
                name=f"{self.config['model_name']}_{self.config['sensor_name']}",
                type="model",
                description=f"Best model trained on {self.config['sensor_name']} data",
                metadata={
                    'test_iou': test_metrics['mean_iou'],
                    'test_accuracy': test_metrics['overall_accuracy'],
                    'best_val_iou': self.best_val_iou,
                    'epochs': self.config['epochs']
                }
            )
            artifact.add_file(str(final_model_path))
            wandb.log_artifact(artifact)
            
            wandb.finish()
        
        print(f"\n✅ Training completed!")
        print(f"   Model saved to: {final_model_path}")
        print(f"   Metrics saved to: {metrics_path}")
        print(f"   Config saved to: {final_config_path}")
        print(f"   Best validation IoU: {self.best_val_iou:.4f}")
        
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
    
    # Normalize paths to use forward slashes
    path_keys = ['data_root', 'dataset_config', 'output_dir']
    for key in path_keys:
        if key in config:
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
    
    return test_metrics

if __name__ == '__main__':
    main()