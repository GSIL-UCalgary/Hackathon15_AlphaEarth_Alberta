"""
Multi-sensor semantic segmentation training pipeline - Modified for cluster_MambaHSI
WITH FIXED MIXED PRECISION
"""

import os
# ==================== CRITICAL: GPU MEMORY FIX ====================
# Must be before any CUDA operations
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
# ================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
import sys

# Add the current directory to path to import your model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your model - ONLY cluster_MambaHSI
try:
    from models.mamba_cluster_hackathon14 import cluster_MambaHSI
    print("✓ Successfully imported cluster_MambaHSI model")
except ImportError as e:
    print(f"✗ Error importing model: {e}")
    try:
        import mamba_cluster_hackathon14
        cluster_MambaHSI = mamba_cluster_hackathon14.cluster_MambaHSI
        print("✓ Successfully imported model from local file")
    except ImportError:
        raise ImportError("Could not import cluster_MambaHSI model")

# Import your dataset class
from utils.dataloaders import Sentinel2ClusterDataset, create_data_loaders, collate_fn

# Enable memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
torch.backends.cudnn.benchmark = True

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ==================== METRICS CALCULATION ====================
class SegmentationMetrics:
    """Calculate comprehensive segmentation metrics"""
    
    def __init__(self, num_classes, ignore_index=-99, class_names=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
    def update(self, pred, target):
        """Update confusion matrix for a batch"""
        pred = pred.flatten().cpu().numpy()
        target = target.flatten().cpu().numpy()
        
        # Remove ignored pixels
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        if len(pred) > 0:
            cm = confusion_matrix(
                target, 
                pred, 
                labels=range(self.num_classes)
            )
            self.confusion_matrix += cm
        
        return len(pred)
    
    def compute(self):
        """Compute all metrics"""
        metrics = {}
        
        # Per-class metrics
        tp = np.diag(self.confusion_matrix).astype(np.float32)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        
        epsilon = 1e-10
        
        # Vectorized calculations
        precision_per_class = tp / (tp + fp + epsilon)
        recall_per_class = tp / (tp + fn + epsilon)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)
        iou_per_class = tp / (tp + fp + fn + epsilon)
        
        # Overall metrics
        total_pixels = np.sum(self.confusion_matrix)
        metrics['overall_accuracy'] = np.sum(tp) / (total_pixels + epsilon)
        metrics['mean_precision'] = np.mean(precision_per_class)
        metrics['mean_recall'] = np.mean(recall_per_class)
        metrics['mean_f1'] = np.mean(f1_per_class)
        metrics['mean_iou'] = np.mean(iou_per_class)
        
        # Frequency weighted IoU
        freq = np.sum(self.confusion_matrix, axis=1) / (total_pixels + epsilon)
        metrics['freq_weighted_iou'] = np.sum(freq * iou_per_class)
        
        # Per-class metrics
        for i in range(self.num_classes):
            metrics[f'{self.class_names[i]}_iou'] = iou_per_class[i]
            metrics[f'{self.class_names[i]}_f1'] = f1_per_class[i]
        
        # Kappa coefficient
        if total_pixels > 0:
            pe = np.sum(np.sum(self.confusion_matrix, axis=0) * np.sum(self.confusion_matrix, axis=1)) / (total_pixels ** 2)
            metrics['kappa'] = (metrics['overall_accuracy'] - pe) / (1 - pe + epsilon)
        else:
            metrics['kappa'] = 0.0
        
        return metrics
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

# ==================== OPTIMIZED TRAINER ====================

class ClusterMambaTrainer:
    """Main training class for cluster_MambaHSI model"""
    
    def __init__(self, config):
        self.config = config
        self.num_classes = config.get('num_classes', 13)
        self.ignore_index = config.get('ignore_index', -1)
        self.cluster_loss_weight = config.get('cluster_loss_weight', 0.1)
        #self.use_amp = config.get('mixed_precision', True) and torch.cuda.is_available()
        self.use_amp = False
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_type = config.get('label_type', 'filtered')
        self.output_dir = Path(config['output_dir']) / f"cluster_mamba_{config['sensor_name']}_{label_type}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset config
        if 'dataset_config' in config and os.path.exists(config['dataset_config']):
            with open(config['dataset_config'], 'r') as f:
                self.dataset_config = json.load(f)
        else:
            self.dataset_config = {
                'num_classes': 13,
                'window_size': 224,
                'background_label': -1,
                'class_names': {str(i): f"Class_{i}" for i in range(13)}
            }
        
        # Save configuration
        self.save_configuration()
        
        # Initialize wandb
        if config.get('use_wandb', False):
            self.init_wandb(timestamp, label_type)
        
        # Create model
        print(f"\nCreating cluster_MambaHSI model for {config['sensor_name']}")
        bands_config = {
            'landsat8': 6,
            'sentinel2': 10,
            'alphaearth': 64
        }
        
        input_channels = bands_config.get(config['sensor_name'], 10)
        num_classes = self.dataset_config.get('num_classes', 13)
        
        self.model = cluster_MambaHSI(
            in_channels=input_channels,
            hidden_dim=config.get('hidden_dim', 64),
            num_classes=num_classes,
            use_residual=True,
            mamba_type='both',
            token_num=config.get('token_num', 4),
            group_num=config.get('group_num', 4),
            use_att=config.get('use_att', True),
            sparsity_ratio=config.get('sparsity_ratio', 1.0),
            attention_heads=config.get('attention_heads', 4),
            selection_mode=config.get('selection_mode', 'cluster')
        ).to(self.device)
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
            print(f"GPU Memory: {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Loss functions
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        # Mixed precision setup (PyTorch 2.0+ syntax)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print("✓ Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        if config.get('scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs']
            )
        elif config.get('scheduler') == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif config.get('warmrestarts', False):
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=20,
                T_mult=2
            )
        else:
            self.scheduler = None
        
        # Metrics
        class_names = []
        if 'class_names' in self.dataset_config:
            class_names = [self.dataset_config['class_names'][str(i)] 
                          for i in range(self.dataset_config['num_classes'])]
        else:
            class_names = [f"Class_{i}" for i in range(self.dataset_config['num_classes'])]
        
        self.metrics = SegmentationMetrics(
            num_classes=self.dataset_config['num_classes'],
            ignore_index=self.ignore_index,
            class_names=class_names
        )
        
        # Best model tracking
        self.best_val_iou = 0
        self.best_model_state = None
        
        print(f"\n✓ Trainer initialized successfully")
    
    def init_wandb(self, timestamp, label_type):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.get('wandb_project', 'AlphaEarth_Alberta_2020_ClusterMamba'),
            entity=self.config.get('wandb_entity', None),
            name=f"cluster_mamba_{self.config['sensor_name']}_{label_type}_{timestamp}",
            config=self.config,
            tags=["cluster_MambaHSI", self.config['sensor_name'], "segmentation", label_type],
            dir=str(self.output_dir)
        )
        print(f"WandB Run: {wandb.run.url}")
    
    def save_configuration(self):
        """Save configuration files"""
        config_json_path = self.output_dir / "config.json"
        with open(config_json_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"✓ Configuration saved to: {self.output_dir}")
    
    def create_dataloaders(self):
        """Create dataloaders for cluster_MambaHSI"""
        print("\nCreating dataloaders...")
        
        # Get label type
        label_type = self.config.get('label_type', 'filtered')
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_data_loaders(
            patches_root=self.config['data_root'],
            clusters_root=self.config['clusters_root'],
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', min(4, os.cpu_count())),
            pin_memory=self.config.get('pin_memory', True if torch.cuda.is_available() else False),
            label_type=label_type,
        )
        
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch - OPTIMIZED for batch_size > 1"""
        self.model.train()
        total_seg_loss = 0
        total_cluster_loss = 0
        total_correct = 0
        total_pixels = 0
        self.metrics.reset()
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare data - batch_size can be > 1 now
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True).long()
            
            # Get cluster inputs - these are already batched
            per_cluster_num = batch['per_cluster_num'][0]  # Same for all batches
            cluster50 = batch['cluster50'].to(self.device, non_blocking=True).long()
            cluster30 = batch['cluster30'].to(self.device, non_blocking=True).long()
            cluster20 = batch['cluster20'].to(self.device, non_blocking=True).long()
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, cluster_loss = self.model(
                images, per_cluster_num, cluster50, cluster30, cluster20
            )
            
            # Calculate segmentation loss
            seg_loss = self.seg_criterion(logits, labels)
            
            # Combine losses
            total_loss = seg_loss + self.cluster_loss_weight * cluster_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Get predictions
            preds = logits.argmax(dim=1)
            
            # Calculate batch accuracy
            valid_mask = labels != self.ignore_index
            if valid_mask.any():
                batch_correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
                batch_pixels = valid_mask.sum().item()
                total_correct += batch_correct
                total_pixels += batch_pixels
                batch_accuracy = batch_correct / batch_pixels if batch_pixels > 0 else 0
            else:
                batch_accuracy = 0
            
            # Update metrics
            self.metrics.update(preds, labels)
            
            # Track losses
            total_seg_loss += seg_loss.item()
            total_cluster_loss += cluster_loss.item()
            
            # Calculate running average accuracy
            running_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
            
            # Update progress bar with MORE metrics
            pbar.set_postfix({
                'seg_loss': f"{seg_loss.item():.3f}",
                'clust_loss': f"{cluster_loss.item():.3f}",
                'total_loss': f"{total_loss.item():.3f}",
                'acc': f"{batch_accuracy:.3f}",  # Batch accuracy
                'run_acc': f"{running_accuracy:.3f}",  # Running accuracy
            })
        
        # Compute epoch metrics
        metrics = self.metrics.compute()
        metrics['seg_loss'] = total_seg_loss / len(train_loader)
        metrics['cluster_loss'] = total_cluster_loss / len(train_loader)
        metrics['total_loss'] = metrics['seg_loss'] + self.cluster_loss_weight * metrics['cluster_loss']
        
        # ADD THIS: Log epoch metrics to wandb
        if self.config.get('use_wandb', False) and wandb.run:
            wandb.log({
                'train/epoch_seg_loss': metrics['seg_loss'],
                'train/epoch_cluster_loss': metrics['cluster_loss'],
                'train/epoch_total_loss': metrics['total_loss'],
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
        """Validate the model - OPTIMIZED for batch_size > 1"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        self.metrics.reset()
        
        pbar = tqdm(val_loader, desc=f"{mode.capitalize()} Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare data
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True).long()
            
            # Get cluster inputs
            per_cluster_num = batch['per_cluster_num'][0]
            cluster50 = batch['cluster50'].to(self.device, non_blocking=True).long()
            cluster30 = batch['cluster30'].to(self.device, non_blocking=True).long()
            cluster20 = batch['cluster20'].to(self.device, non_blocking=True).long()
            
            # Forward pass
            logits = self.model(
                images, per_cluster_num, cluster50, cluster30, cluster20
            )
            
            # Calculate loss
            loss = self.seg_criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions
            preds = logits.argmax(dim=1)
            
            # Calculate batch accuracy
            valid_mask = labels != self.ignore_index
            if valid_mask.any():
                batch_correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
                batch_pixels = valid_mask.sum().item()
                total_correct += batch_correct
                total_pixels += batch_pixels
                batch_accuracy = batch_correct / batch_pixels if batch_pixels > 0 else 0
            else:
                batch_accuracy = 0
            
            # Update metrics
            self.metrics.update(preds, labels)
            
            # Calculate running average accuracy
            running_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
            
            # Update progress bar with MORE metrics
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'acc': f"{batch_accuracy:.3f}",
                'run_acc': f"{running_accuracy:.3f}",
            })
        
        # Compute metrics
        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / len(val_loader)
        
        if self.config.get('use_wandb', False) and wandb.run:
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
        
        print(f"\nStarting training for cluster_MambaHSI")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Training history
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
            print(f"Train - Total Loss: {train_metrics['total_loss']:.4f} "
              f"(Seg: {train_metrics['seg_loss']:.4f}, "
              f"Cluster: {train_metrics['cluster_loss']:.4f}), "
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
            history['train_loss'].append(train_metrics['total_loss'])
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
                print(f"✓ Saved best model (mIoU: {self.best_val_iou:.4f})")
            
            # Save regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
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
        
        # Save final model and results
        self.save_final_results(test_metrics, history)
        
        return test_metrics
    
    def save_final_results(self, test_metrics, history):
        """Save final model and results"""
        # Save final model
        final_model_path = self.output_dir / "final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'test_metrics': test_metrics,
            'history': history,
            'best_val_iou': self.best_val_iou
        }, final_model_path)
        
        # Save test metrics
        metrics_path = self.output_dir / "test_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        # ADD THIS: Log final test metrics to wandb
        if self.config.get('use_wandb', False) and wandb.run:
            wandb.log({
                'test/final_iou': test_metrics['mean_iou'],
                'test/final_f1': test_metrics['mean_f1'],
                'test/final_accuracy': test_metrics['overall_accuracy'],
                'test/final_kappa': test_metrics['kappa']
            })
            
            # Save model artifact to wandb
            artifact = wandb.Artifact(
                name=f"cluster_mamba_{self.config['sensor_name']}",
                type="model",
                description=f"Best cluster_MambaHSI model trained on {self.config['sensor_name']} filtered data",
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
        print(f"   Best validation IoU: {self.best_val_iou:.4f}")

# ==================== MAIN FUNCTION ====================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train cluster_MambaHSI model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML config file')
    return parser.parse_args()

def load_config_from_yaml(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Normalize paths
    path_keys = ['data_root', 'clusters_root', 'dataset_config', 'output_dir']
    for key in path_keys:
        if key in config:
            config[key] = config[key].replace('\\', '/')
    
    # Set defaults
    config.setdefault('batch_size', 20)
    config.setdefault('hidden_dim', 64)
    config.setdefault('token_num', 4)
    config.setdefault('group_num', 4)
    config.setdefault('sparsity_ratio', 1.0)
    config.setdefault('attention_heads', 4)
    config.setdefault('selection_mode', 'cluster')
    config.setdefault('cluster_loss_weight', 0.1)
    config.setdefault('ignore_index', -1)
    config.setdefault('num_workers', min(4, os.cpu_count()))
    config.setdefault('pin_memory', True if torch.cuda.is_available() else False)
    config.setdefault('mixed_precision', True)
    
    return config

def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration
    config = load_config_from_yaml(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Verify required fields
    required_fields = ['model_name', 'sensor_name', 'data_root', 'clusters_root']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' not found in config")
    
    # Verify model is cluster_MambaHSI
    if config['model_name'] != 'cluster_MambaHSI':
        raise ValueError(f"This trainer is only for cluster_MambaHSI, got {config['model_name']}")
    
    # # Verify batch_size is 1
    # if config.get('batch_size', 1) != 1:
    #     print(f"⚠ Warning: cluster_MambaHSI requires batch_size=1. Setting to 1.")
    #     config['batch_size'] = 1
    
    # Create trainer and train
    trainer = ClusterMambaTrainer(config)
    test_metrics = trainer.train()
    
    return test_metrics

if __name__ == '__main__':
    main()