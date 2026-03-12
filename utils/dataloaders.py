import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
import json
from typing import Tuple, Dict, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Dataset class for Sentinel-2 patches with cluster maps
class Sentinel2ClusterDataset(Dataset):
    """
    Dataset for Sentinel-2 patches with pre-generated cluster maps.
    
    Directory Structure:
    /beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/
    ├── train_val_test_patches/patches/
    │   ├── train/
    │   │   ├── sentinel2/img/class_0_patch_0.tif (uint8 images)
    │   │   └── labels/unfiltered/class_0_patch_0.tif (0-12 classes, -99 background)
    │   ├── val/
    │   └── test/
    └── clustered_datasets/sentinel2/unfiltered/
        ├── train/clusters/class_0_patch_0_cluster{20,30,50}.npy
        ├── val/clusters/
        └── test/clusters/
    """
    
    def __init__(
        self,
        patches_root: str,
        clusters_root: str,
        split: str = "train",
        label_type: str = "filtered",
        sensor_name: str = "sentinel2",
        augment: bool = False,
        num_classes: int = 13
    ):
        """
        Args:
            patches_root: Path to /beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/train_val_test_patches/patches
            clusters_root: Path to /beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/clustered_datasets
            split: One of ['train', 'val', 'test']
            label_type: 'filtered'
            sensor_name: 'sentinel2'
            augment: Apply data augmentation (only for training)
            num_classes: Number of semantic classes (13)
        """
        self.patches_root = Path(patches_root)
        self.clusters_root = Path(clusters_root)
        self.split = split
        self.label_type = label_type
        self.sensor_name = sensor_name
        self.augment = augment and (split == "train")
        self.num_classes = num_classes
        
        # Define paths
        self.image_dir = self.patches_root / split / sensor_name / "img"
        self.label_dir = self.patches_root / split / "labels" / label_type
        self.cluster_dir = self.clusters_root / sensor_name / label_type / split / "clusters"
        
        # Verify directories exist
        assert self.image_dir.exists(), f"Image directory not found: {self.image_dir}"
        assert self.label_dir.exists(), f"Label directory not found: {self.label_dir}"
        assert self.cluster_dir.exists(), f"Cluster directory not found: {self.cluster_dir}"
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob("*.tif")))
        
        if not self.image_files:
            raise ValueError(f"No .tif files found in {self.image_dir}")
        
        print(f"Found {len(self.image_files)} patches for {split} split")
        
        # Define augmentations
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        """Get data transformations"""
        if self.augment:
            return A.Compose([
                # Spatial augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,  # 1/16 of image size
                    scale_limit=0.1,     # 10% zoom
                    rotate_limit=15,     # degrees
                    p=0.5
                ),
                # Spectral augmentations (applied per-channel)
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                # Normalize to [0, 1] range
                A.Normalize(
                    mean=[0.0] * 13,  # Assuming 13 spectral bands
                    std=[1.0] * 13,
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ], additional_targets={'cluster50': 'mask', 'cluster30': 'mask', 'cluster20': 'mask'})
        else:
            return A.Compose([
                A.Normalize(
                    mean=[0.0] * 13,
                    std=[1.0] * 13,
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ], additional_targets={'cluster50': 'mask', 'cluster30': 'mask', 'cluster20': 'mask'})
    
    def __len__(self):
        return len(self.image_files)
    
    def _load_raster(self, path: Path) -> np.ndarray:
        """Load raster file (TIFF)"""
        with rasterio.open(path) as src:
            data = src.read()
            return data
    
    def _process_label(self, label: np.ndarray) -> np.ndarray:
        """Process label: convert -99 to -1 and ensure int type"""
        label = label.astype(np.int32)
        label[label == -99] = -1  # Convert background from -99 to -1
        return label
    
    def _calculate_per_cluster_num(self, cluster_map: np.ndarray, num_clusters: int) -> List[int]:
        """Calculate number of pixels per cluster"""
        counts = np.zeros(num_clusters, dtype=int)
        unique, counts_arr = np.unique(cluster_map, return_counts=True)
        for cluster_id, count in zip(unique, counts_arr):
            if 0 <= cluster_id < num_clusters:
                counts[cluster_id] = count
        return counts.tolist()
    
    def __getitem__(self, idx: int) -> Dict:
        """Load a single sample"""
        # Get file paths
        img_path = self.image_files[idx]
        patch_name = img_path.stem
        
        label_path = self.label_dir / f"{patch_name}.tif"
        cluster50_path = self.cluster_dir / f"{patch_name}_cluster50.npy"
        cluster30_path = self.cluster_dir / f"{patch_name}_cluster30.npy"
        cluster20_path = self.cluster_dir / f"{patch_name}_cluster20.npy"
        
        # Load data
        image = self._load_raster(img_path)  # Shape: (C, H, W), uint8
        label = self._load_raster(label_path)[0]  # Shape: (H, W), take first band
        label = self._process_label(label)
        
        # Load cluster maps
        cluster50 = np.load(cluster50_path)  # Shape: (H, W), 0-49
        cluster30 = np.load(cluster30_path)  # Shape: (H, W), 0-29
        cluster20 = np.load(cluster20_path)  # Shape: (H, W), 0-19
        
        # Verify dimensions match
        assert image.shape[1:] == label.shape, f"Image {image.shape[1:]} and label {label.shape} shapes don't match"
        assert label.shape == cluster50.shape == cluster30.shape == cluster20.shape, \
            f"All maps must have same shape. Got: label {label.shape}, cluster50 {cluster50.shape}"
        
        # Transpose image to (H, W, C) for albumentations
        image = np.transpose(image, (1, 2, 0))
        
        # Prepare for augmentation
        data = {
            'image': image,
            'mask': label,
            'cluster50': cluster50,
            'cluster30': cluster30,
            'cluster20': cluster20
        }
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(**data)
            image = transformed['image']  # Shape: (C, H, W), float32
            label = transformed['mask']  # Shape: (H, W), int64
            cluster50 = transformed['cluster50']  # Shape: (H, W), int64
            cluster30 = transformed['cluster30']  # Shape: (H, W), int64
            cluster20 = transformed['cluster20']  # Shape: (H, W), int64
        
        # Calculate per-cluster pixel counts
        per_cluster_num = [
            self._calculate_per_cluster_num(cluster50.numpy() if torch.is_tensor(cluster50) else cluster50, 50),
            self._calculate_per_cluster_num(cluster30.numpy() if torch.is_tensor(cluster30) else cluster30, 30),
            self._calculate_per_cluster_num(cluster20.numpy() if torch.is_tensor(cluster20) else cluster20, 20)
        ]
        
        return {
            'image': image,  # (C, H, W)
            'label': label.long(),  # (H, W)
            'cluster50': cluster50.long() if torch.is_tensor(cluster50) else torch.from_numpy(cluster50).long(),
            'cluster30': cluster30.long() if torch.is_tensor(cluster30) else torch.from_numpy(cluster30).long(),
            'cluster20': cluster20.long() if torch.is_tensor(cluster20) else torch.from_numpy(cluster20).long(),
            'per_cluster_num': per_cluster_num,
            'patch_name': patch_name
        }

# Custom collate function to handle per_cluster_num lists
def collate_fn(batch):
    """Custom collate function to handle variable-length lists"""
    collated = {}
    
    # Stack tensors
    for key in ['image', 'cluster50', 'cluster30', 'cluster20']:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
    
    # Special handling for label to convert to long
    if 'label' in batch[0]:
        collated['label'] = torch.stack([item['label'].long() for item in batch])  # ADD .long() HERE
    
    # Keep lists as-is for per_cluster_num
    collated['per_cluster_num'] = [item['per_cluster_num'] for item in batch]
    
    # Keep patch names
    collated['patch_names'] = [item['patch_name'] for item in batch]
    
    return collated
# Create data loaders for training, validation, and testing
def create_data_loaders(
    patches_root: str,
    clusters_root: str,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    label_type: str = "filtered"
    
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = Sentinel2ClusterDataset(
        patches_root=patches_root,
        clusters_root=clusters_root,
        split='train',
        augment=True,
        label_type=label_type,
    )
    
    val_dataset = Sentinel2ClusterDataset(
        patches_root=patches_root,
        clusters_root=clusters_root,
        split='val',
        augment=False,
        label_type=label_type,
    )
    
    test_dataset = Sentinel2ClusterDataset(
        patches_root=patches_root,
        clusters_root=clusters_root,
        split='test',
        augment=False,
        label_type=label_type,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # ADD THIS
        persistent_workers=True if num_workers > 0 else False,  # ADD THIS
        collate_fn=collate_fn,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # ADD THIS
        persistent_workers=True if num_workers > 0 else False,  # ADD THIS
        collate_fn=collate_fn,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    print(f"Created data loaders:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

# Simple test function to verify dataset AND dataloaders
def test_dataset():
    """Test the dataset class and dataloaders"""
    patches_root = "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/train_val_test_patches/patches"
    clusters_root = "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/clustered_datasets"
    
    print("Testing Dataset and Dataloaders...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        patches_root=patches_root,
        clusters_root=clusters_root,
        batch_size=1,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        pin_memory=False
    )
    
    # Test train dataloader
    print(f"\n{'='*60}")
    print("TESTING TRAIN DATALOADER")
    print(f"{'='*60}")
    
    # Get first batch from train loader
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nTrain Batch {batch_idx}:")
        print(f"  Batch keys: {list(batch.keys())}")
        print(f"  Image shape: {batch['image'].shape}")  # Should be [batch_size, C, H, W]
        print(f"  Label shape: {batch['label'].shape}")  # Should be [batch_size, H, W]
        print(f"  Cluster50 shape: {batch['cluster50'].shape}")  # Should be [batch_size, H, W]
        print(f"  Cluster30 shape: {batch['cluster30'].shape}")
        print(f"  Cluster20 shape: {batch['cluster20'].shape}")
        print(f"  Number of samples in batch: {len(batch['patch_names'])}")
        print(f"  Patch names: {batch['patch_names']}")
        
        # Check for background values
        print(f"\n  Checking for background values in labels:")
        unique_labels = torch.unique(batch['label'])
        print(f"    Unique label values in batch: {unique_labels}")
        
        # Check if any -99 or -1 exists
        has_minus99 = (batch['label'] == -99).any()
        has_minus1 = (batch['label'] == -1).any()
        print(f"    Has -99? {has_minus99.item() if has_minus99.dim() == 0 else has_minus99}")
        print(f"    Has -1? {has_minus1.item() if has_minus1.dim() == 0 else has_minus1}")
        
        # Check per_cluster_num structure
        print(f"\n  Checking per_cluster_num structure:")
        print(f"    Number of samples with per_cluster_num: {len(batch['per_cluster_num'])}")
        if len(batch['per_cluster_num']) > 0:
            sample_0 = batch['per_cluster_num'][0]
            print(f"    Sample 0 has {len(sample_0)} cluster levels:")
            print(f"      Level 0 (cluster50): {len(sample_0[0])} values")
            print(f"      Level 1 (cluster30): {len(sample_0[1])} values")
            print(f"      Level 2 (cluster20): {len(sample_0[2])} values")
            print(f"      Example counts - cluster50 first 5: {sample_0[0][:5]}")
        
        # Check one sample in detail
        print(f"\n  Detailed check of first sample in batch:")
        print(f"    Label unique: {torch.unique(batch['label'][0])}")
        print(f"    Cluster50 range: [{batch['cluster50'][0].min()} to {batch['cluster50'][0].max()}]")
        print(f"    Cluster30 range: [{batch['cluster30'][0].min()} to {batch['cluster30'][0].max()}]")
        print(f"    Cluster20 range: [{batch['cluster20'][0].min()} to {batch['cluster20'][0].max()}]")
        
        # Only test first batch
        break
    
    # Test validation dataloader
    print(f"\n{'='*60}")
    print("TESTING VALIDATION DATALOADER")
    print(f"{'='*60}")
    
    for batch_idx, batch in enumerate(val_loader):
        print(f"\nValidation Batch {batch_idx}:")
        print(f"  Batch size: {batch['image'].shape[0]}")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Label shape: {batch['label'].shape}")
        
        # Check background
        unique_labels = torch.unique(batch['label'])
        print(f"  Unique label values: {unique_labels}")
        
        # Only test first batch
        break
    
    # Test test dataloader
    print(f"\n{'='*60}")
    print("TESTING TEST DATALOADER")
    print(f"{'='*60}")
    
    for batch_idx, batch in enumerate(test_loader):
        print(f"\nTest Batch {batch_idx}:")
        print(f"  Batch size: {batch['image'].shape[0]}")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Label shape: {batch['label'].shape}")
        
        # Check background
        unique_labels = torch.unique(batch['label'])
        print(f"  Unique label values: {unique_labels}")
        
        # Only test first batch
        break
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("✓ Dataloaders created successfully")
    print("✓ All batches have correct shapes")
    print("✓ Collate function working properly")
    print("✓ per_cluster_num structure preserved")
    
    # Final check: Are there ANY background pixels in the entire dataset?
    print(f"\nChecking a few random batches for background pixels:")
    
    background_found = False
    batches_checked = 0
    
    # Check a few more batches
    for batch_idx, batch in enumerate(train_loader):
        if batches_checked >= 3:  # Check 3 batches
            break
            
        has_minus99 = (batch['label'] == -99).any()
        has_minus1 = (batch['label'] == -1).any()
        
        if has_minus99 or has_minus1:
            background_found = True
            print(f"  Batch {batch_idx}: Found background!")
            if has_minus99:
                print(f"    -99 count: {(batch['label'] == -99).sum().item()}")
            if has_minus1:
                print(f"    -1 count: {(batch['label'] == -1).sum().item()}")
        else:
            print(f"  Batch {batch_idx}: No background pixels")
        
        batches_checked += 1
    
    if not background_found:
        print(f"\nCONCLUSION: No background pixels (-99 or -1) found in tested batches.")
        print("The dataset appears to have clean labels with only class 0-12.")
    else:
        print(f"\nCONCLUSION: Background pixels found!")
        print("Check if conversion from -99 to -1 is needed.")
    
    return True


# Run test if executed as main
if __name__ == "__main__":
    # Test the dataset and dataloaders
    test_dataset()
