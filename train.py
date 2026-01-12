"""
Multi-sensor semantic segmentation training pipeline
Supports multiple models, metrics, and Weights & Biases logging
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
from datetime import datetime
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.cuda.amp import GradScaler, autocast

# Import your models
from models import (
    MIMUNet, FocalUNet, SepViTUNet, SwinUNet, 
    CATUNet, TwinsUNet, BasicUNet, HRNetWrapper
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

class MultisensorDataset(Dataset):
    """Dataset for multi-sensor semantic segmentation"""
    
    def __init__(self, root_dir, sensor_name, split='train', transform=None):
        """
        Args:
            root_dir: Root directory with patches
            sensor_name: Name of sensor ('landsat8', 'sentinel2', 'alphaearth')
            split: 'train', 'val', or 'test'
            transform: Optional transforms
        """
        self.root_dir = Path(root_dir)
        self.sensor_name = sensor_name
        self.split = split
        self.transform = transform
        
        # Paths
        self.img_dir = self.root_dir / split / sensor_name / 'img'
        self.label_dir = self.root_dir / split / 'labels' / 'filtered'
        
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
        
        # Read GeoTIFF files (simplified - you might want to use rasterio or gdal)
        # For now, assume they're already preprocessed to numpy arrays
        # You'll need to implement proper GeoTIFF reading
        image = self.load_geotiff(img_path)  # Shape: (C, H, W)
        label = self.load_geotiff(label_path)  # Shape: (H, W)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # Apply transforms if any
        if self.transform:
            image, label = self.transform(image, label)
        
        return image, label
    
    def load_geotiff(self, path):
        """Load GeoTIFF file as numpy array"""
        # Implement using rasterio or gdal
        # For now, return dummy data
        import rasterio
        with rasterio.open(path) as src:
            data = src.read()
        return data

class SegmentationMetrics:
    """Calculate various segmentation metrics"""
    
    def __init__(self, num_classes, ignore_index=-99):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
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
        
        # Per-class accuracy
        accuracy_per_class = tp / (tp + fp + fn + epsilon)
        
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
        metrics['mean_accuracy'] = np.mean(accuracy_per_class)
        metrics['mean_precision'] = np.mean(precision_per_class)
        metrics['mean_recall'] = np.mean(recall_per_class)
        metrics['mean_f1'] = np.mean(f1_per_class)
        metrics['mean_iou'] = np.mean(iou_per_class)
        
        # Frequency weighted IoU
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + epsilon)
        metrics['freq_weighted_iou'] = np.sum(freq * iou_per_class)
        
        # Per-class metrics for logging
        for i in range(self.num_classes):
            metrics[f'class_{i}_iou'] = iou_per_class[i]
            metrics[f'class_{i}_f1'] = f1_per_class[i]
            metrics[f'class_{i}_precision'] = precision_per_class[i]
            metrics[f'class_{i}_recall'] = recall_per_class[i]
        
        return metrics
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

def create_model(model_name, num_classes, sensor_name, config):
    """Create model based on name and configuration"""
    
    model_config = {
        'num_classes': num_classes,
        'input_channels': config['sensors'][sensor_name]['bands'],
        'img_size': 224,  # Your patch size
        'ignore_index': -99
    }
    
    if model_name == 'MIMUNet':
        return MIMUNet(**model_config)
    elif model_name == 'FocalUNet':
        return FocalUNet(**model_config)
    elif model_name == 'SepViTUNet':
        return SepViTUNet(**model_config)
    elif model_name == 'SwinUNet':
        return SwinUNet(**model_config)
    elif model_name == 'CATUNet':
        return CATUNet(**model_config)
    elif model_name == 'TwinsUNet':
        return TwinsUNet(**model_config)
    elif model_name == 'BasicUNet':
        return BasicUNet(**model_config)
    elif model_name == 'HRNet':
        return HRNetWrapper(**model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

class Trainer:
    """Main training class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize wandb
        if config['use_wandb']:
            wandb.init(
                project=config['wandb_project'],
                name=f"{config['model_name']}_{config['sensor_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config
            )
        
        # Load dataset config
        with open(config['dataset_config'], 'r') as f:
            self.dataset_config = json.load(f)
        
        # Create model
        self.model = create_model(
            config['model_name'],
            self.dataset_config['num_classes'],
            config['sensor_name'],
            self.dataset_config
        ).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {config['model_name']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss function
        if config['loss_fn'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=config['ignore_index'],
                weight=config.get('class_weights', None)
            )
        elif config['loss_fn'] == 'focal':
            from focal_loss import FocalLoss
            self.criterion = FocalLoss(
                ignore_index=config['ignore_index']
            )
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
                verbose=True
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
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # Metrics
        self.metrics = SegmentationMetrics(
            num_classes=self.dataset_config['num_classes'],
            ignore_index=config['ignore_index']
        )
        
        # Best model tracking
        self.best_val_iou = 0
        self.best_model_state = None
        
        # Create output directory
        self.output_dir = Path(config['output_dir']) / f"{config['model_name']}_{config['sensor_name']}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataloaders(self):
        """Create train, val, and test dataloaders"""
        
        # You'll need to implement proper data augmentation
        from torchvision import transforms
        
        # Simple transforms for now
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
        ])
        
        val_transform = None
        
        # Create datasets
        train_dataset = MultisensorDataset(
            root_dir=self.config['data_root'],
            sensor_name=self.config['sensor_name'],
            split='train',
            transform=train_transform
        )
        
        val_dataset = MultisensorDataset(
            root_dir=self.config['data_root'],
            sensor_name=self.config['sensor_name'],
            split='val',
            transform=val_transform
        )
        
        test_dataset = MultisensorDataset(
            root_dir=self.config['data_root'],
            sensor_name=self.config['sensor_name'],
            split='test',
            transform=val_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
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
                with autocast():
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
        
        # Compute metrics
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / len(train_loader)
        
        # Log to wandb
        if self.config['use_wandb']:
            wandb.log({
                'train/loss': metrics['loss'],
                'train/mean_iou': metrics['mean_iou'],
                'train/mean_f1': metrics['mean_f1'],
                'train/overall_accuracy': metrics['overall_accuracy'],
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
        
        # Store sample predictions for visualization
        sample_images = []
        sample_predictions = []
        sample_labels = []
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Update metrics
            preds = outputs.argmax(dim=1)
            self.metrics.update(preds, labels)
            
            # Store samples for visualization (first batch only)
            if batch_idx == 0 and self.config['use_wandb']:
                sample_images.append(images[:3].cpu())
                sample_predictions.append(preds[:3].cpu())
                sample_labels.append(labels[:3].cpu())
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / len(val_loader)
        
        # Log to wandb
        if self.config['use_wandb']:
            wandb.log({
                f'{mode}/loss': metrics['loss'],
                f'{mode}/mean_iou': metrics['mean_iou'],
                f'{mode}/mean_f1': metrics['mean_f1'],
                f'{mode}/overall_accuracy': metrics['overall_accuracy'],
                'epoch': epoch
            })
            
            # Log sample predictions
            if sample_images:
                self.log_sample_predictions(
                    sample_images[0], 
                    sample_predictions[0], 
                    sample_labels[0],
                    mode
                )
        
        return metrics
    
    def log_sample_predictions(self, images, predictions, labels, mode):
        """Log sample predictions to wandb"""
        
        # Convert to numpy
        images_np = images.numpy()
        preds_np = predictions.numpy()
        labels_np = labels.numpy()
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        
        # Get class colors from dataset config
        class_colors = self.get_class_colors()
        
        for i in range(3):
            # Show RGB image (assuming first 3 bands)
            if images_np[i].shape[0] >= 3:
                rgb_img = np.transpose(images_np[i][:3], (1, 2, 0))
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                axes[i, 0].imshow(rgb_img)
            
            # Show ground truth
            gt_img = self.colorize_mask(labels_np[i], class_colors)
            axes[i, 1].imshow(gt_img)
            
            # Show prediction
            pred_img = self.colorize_mask(preds_np[i], class_colors)
            axes[i, 2].imshow(pred_img)
        
        # Set titles
        axes[0, 0].set_title('Input (RGB)')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 2].set_title('Prediction')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"{mode}/sample_predictions": wandb.Image(fig)})
        plt.close(fig)
    
    def get_class_colors(self):
        """Get class colors from dataset config"""
        # You might want to store colors in your dataset config
        # For now, use a simple color map
        import matplotlib.cm as cm
        colors = cm.tab20(np.linspace(0, 1, self.dataset_config['num_classes']))
        return colors
    
    def colorize_mask(self, mask, colors):
        """Colorize a mask with class colors"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx in range(self.dataset_config['num_classes']):
            colored[mask == class_idx] = (colors[class_idx][:3] * 255).astype(np.uint8)
        
        return colored
    
    def train(self):
        """Main training loop"""
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders()
        
        print(f"\nStarting training for {self.config['model_name']} on {self.config['sensor_name']}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
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
            
            # Save best model
            if val_metrics['mean_iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['mean_iou']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                
                # Save checkpoint
                checkpoint_path = self.output_dir / f"best_model_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_iou': self.best_val_iou,
                    'config': self.config
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            
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
        print("Testing on test set")
        print(f"{'='*60}")
        test_metrics = self.validate(test_loader, epoch, mode='test')
        
        print(f"\nTest Results:")
        print(f"  mIoU: {test_metrics['mean_iou']:.4f}")
        print(f"  Mean F1: {test_metrics['mean_f1']:.4f}")
        print(f"  Overall Accuracy: {test_metrics['overall_accuracy']:.4f}")
        print(f"  Frequency Weighted IoU: {test_metrics['freq_weighted_iou']:.4f}")
        
        # Save final model
        final_model_path = self.output_dir / "final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'test_metrics': test_metrics
        }, final_model_path)
        
        # Save test metrics
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Log test metrics to wandb
        if self.config['use_wandb']:
            wandb.log({'test/best_iou': test_metrics['mean_iou']})
            wandb.finish()
        
        return test_metrics

def main():
    """Main function to run training"""
    
    # Configuration
    config = {
        # Model and dataset
        'model_name': 'BasicUNet',  # Choose from: MIMUNet, FocalUNet, etc.
        'sensor_name': 'landsat8',  # 'landsat8', 'sentinel2', 'alphaearth'
        
        # Paths
        'data_root': r'D:\Hackathon15_AlphaEarth\train_val_test_patches\patches',
        'dataset_config': r'D:\Hackathon15_AlphaEarth\train_val_test_patches\multisensor_dataset_config.json',
        'output_dir': './experiments',
        
        # Training hyperparameters
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        
        # Loss and optimizer
        'loss_fn': 'cross_entropy',  # 'cross_entropy' or 'focal'
        'optimizer': 'adamw',  # 'adam', 'adamw', 'sgd'
        'scheduler': 'reduce_on_plateau',  # 'cosine', 'reduce_on_plateau', 'step', None
        
        # Model settings
        'ignore_index': -99,
        'class_weights': None,  # Optional: list of class weights
        
        # Training settings
        'use_amp': True,  # Mixed precision training
        'num_workers': 4,
        'save_every': 10,
        
        # Weights & Biases
        'use_wandb': True,
        'wandb_project': 'multisensor-segmentation',
    }
    
    # Create trainer and train
    trainer = Trainer(config)
    test_metrics = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Test mIoU: {test_metrics['mean_iou']:.4f}")

if __name__ == '__main__':
    main()