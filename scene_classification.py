"""
Scene Classification for Multi-Sensor Data
Classifies entire scenes and saves in the same folder as trained model
"""

import os
import torch
import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import yaml
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import your models
from models import (
    MIMUNet, FocalUNet, SepViTUNet, SwinUNet, 
    CATUNet, TwinsUNet, BasicUNet, HRNetWrapper
)

# ==================== PREDEFINED SCENE PATHS ====================
SCENE_PATHS = {
    'landsat8': "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/preprocessing/Alberta_L8_2020/Alberta_2020_NAD83_StatsCan_L8_30m_Mosaics_EPSG_3979_Clipped_Stack/Alberta_2020_L8_Stacked_6Bands.tif",
    'sentinel2': "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/preprocessing/Alberta_Sentinel2_2020/Alberta_2020_NAD83_StatsCan_Sentinel2_30m_Mosaics_EPSG_3979_Clipped_Stack/Alberta_2020_S2_Stacked_10Bands.tif",
    'alphaearth': "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/preprocessing/AlphaEarth_Dataset/Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979_Clipped_Stack/Alberta_2020_AlphaEarth_Stacked_64Bands.tif"
}

def create_model(model_name, sensor_name, config):
    """Create model based on name and sensor configuration"""
    
    # Get number of bands from config
    bands_config = {
        'landsat8': 6,
        'sentinel2': 10,
        'alphaearth': 64
    }
    
    input_channels = bands_config[sensor_name]
    num_classes = config['num_classes']
    
    # Create model configuration dictionary
    model_config = {
        'model': {
            'in_channels': input_channels,
            'num_classes': num_classes,
            'img_size': 224,
            'stem_dim': 32,
            'stem_kernel': 3,
            'stem_padding': 1,
            'stem_downsampling': False,
            'dims': [32, 64, 128, 256],
            'depths': [1, 1, 2, 1],
        }
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
        model_config = {
            'in_channels': input_channels,
            'num_classes': num_classes,
            'stem_dim': 32,
            'stem_kernel': 3,
            'stem_padding': 1,
            'stem_downsampling': False,
            'dims': [32, 64, 128, 256],
            'depths': [1, 1, 2, 1],
        }
        return BasicUNet(model_config)
    elif model_name == 'HRNet':
        return HRNetWrapper(**model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def patch_generator(scene_array, patch_size, overlap=0):
    """
    Generator that yields patches from the scene with optional overlap.
    
    Args:
        scene_array (np.ndarray): (H, W, C) array
        patch_size (int): Size of square patch
        overlap (int): Overlap between patches (default: 0)
    
    Yields:
        (start_row, start_col, patch_array): Patch coordinates and data
    """
    H, W, _ = scene_array.shape
    stride = patch_size - overlap
    
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            # Get patch with overlap
            patch = scene_array[
                r:min(r + patch_size, H),
                c:min(c + patch_size, W),
                :
            ]
            yield r, c, patch
def classify_full_scene(
    scene_path,
    model,
    model_config,
    experiment_dir,
    patch_size=224,
    batch_size=16,
    device=None,
    overlap=0,
    save_probabilities=True,
    target_crs='EPSG:3979'  # NEW: Specify target CRS
):
    """
    Classify a full scene and save in the experiment directory.
    
    Args:
        scene_path (str): Path to input GeoTIFF
        model: Trained PyTorch model
        model_config: Model configuration dictionary
        experiment_dir (str): Experiment directory where model is saved
        patch_size (int): Size of patches (must match training)
        batch_size (int): Batch size for inference
        device: PyTorch device (CPU/GPU)
        overlap (int): Overlap between patches (for smoother edges)
        save_probabilities (bool): Save class probabilities instead of labels
        target_crs (str): Target coordinate reference system (default: EPSG:3979)
    
    Returns:
        output_path: Path to saved classified scene
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Loading scene: {scene_path}")
    print(f"Experiment directory: {experiment_dir}")
    
    # Create classified_scenes subdirectory in experiment folder
    output_dir = Path(experiment_dir) / "classified_scenes"
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filename
    scene_name = Path(scene_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_probabilities:
        output_filename = f"{scene_name}_probabilities_{timestamp}.tif"
    else:
        output_filename = f"{scene_name}_classified_{timestamp}.tif"
    
    output_path = output_dir / output_filename
    
    # 1. Load scene with geospatial metadata
    with rasterio.open(scene_path) as src:
        # Read all bands
        scene_array = src.read()  # Shape: (C, H, W)
        
        # Get geospatial metadata
        transform = src.transform
        crs = src.crs
        profile = src.profile
        
        print(f"Scene shape: {scene_array.shape}")
        print(f"Input CRS: {crs}")
        print(f"Target CRS: {target_crs}")
        print(f"Transform: {transform}")
        
        # IMPROVED CRS CHECKING
    if crs is not None:
        crs_str = str(crs).upper()
        
        # Check if it's actually EPSG:3979 by looking for keywords
        is_epsg_3979 = False
        
        # Check for EPSG:3979 code
        if 'EPSG:3979' in crs_str:
            is_epsg_3979 = True
        
        # Check for NAD83(CSRS) / Canada Atlas Lambert (the actual name)
        elif ('NAD83' in crs_str and 'CSRS' in crs_str and 
              'CANADA ATLAS LAMBERT' in crs_str):
            is_epsg_3979 = True
            print(f"✓ CRS is NAD83(CSRS) / Canada Atlas Lambert (EPSG:3979)")
        
        # Check for just the local name
        elif 'NAD83(CSRS) / CANADA ATLAS LAMBERT' in crs_str:
            is_epsg_3979 = True
            print(f"✓ CRS is NAD83(CSRS) / Canada Atlas Lambert (EPSG:3979)")
        
        if not is_epsg_3979:
            print(f"⚠️  WARNING: Input CRS may not be EPSG:3979")
            print(f"   CRS: {crs}")
            print(f"   Expected: {target_crs}")
        else:
            print(f"✅ Input CRS matches EPSG:3979")
    else:
        print("⚠️  WARNING: Input scene has no CRS information!")
        print(f"   Assuming it's in {target_crs}")
        
        # Transpose to (H, W, C) for easier patch extraction
        scene_array = scene_array.transpose(1, 2, 0)  # (H, W, C)
    
    H, W, C = scene_array.shape
    print(f"Transposed shape: {scene_array.shape}")
    
    # 2. Initialize output arrays
    if save_probabilities:
        num_classes = model_config['num_classes']
        classified_scene = np.zeros((H, W, num_classes), dtype=np.float32)
    else:
        classified_scene = np.zeros((H, W), dtype=np.uint8)
    
    # 3. Prepare model
    model.to(device)
    model.eval()
    
    # 4. Calculate total patches
    stride = patch_size - overlap
    total_patches = ((H - overlap) // stride + (1 if (H - overlap) % stride > 0 else 0)) * \
                    ((W - overlap) // stride + (1 if (W - overlap) % stride > 0 else 0))
    
    print(f"\nClassification parameters:")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Overlap: {overlap} pixels")
    print(f"  Stride: {stride} pixels")
    print(f"  Total patches: {total_patches}")
    print(f"  Batch size: {batch_size}")
    print(f"  Save probabilities: {save_probabilities}")
    print(f"  Target CRS: {target_crs}")
    print(f"  Output path: {output_path}")
    
    # 5. Process patches
    patches = []
    coords = []
    patch_counter = 0
    
    with tqdm(total=total_patches, desc="Classifying patches") as pbar:
        for r, c, patch in patch_generator(scene_array, patch_size, overlap):
            # Store original patch dimensions
            ph, pw, pc = patch.shape
            
            # Pad patch if needed (at borders)
            if ph < patch_size or pw < patch_size:
                pad_h = patch_size - ph
                pad_w = patch_size - pw
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), 
                              mode='reflect')
            
            # Normalize: UInt8 → [0, 1]
            patch = patch.astype(np.float32) / 255.0
            
            patches.append(patch)
            coords.append((r, c, ph, pw))
            patch_counter += 1
            
            # Process batch when full
            if len(patches) == batch_size:
                # Convert to PyTorch tensor: (batch, C, H, W)
                batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2)).float().to(device)
                
                with torch.no_grad():
                    if save_probabilities:
                        # Get softmax probabilities
                        logits = model(batch)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        preds = probs  # Shape: (batch, num_classes, H, W)
                    else:
                        # Get class labels
                        logits = model(batch)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()  # Shape: (batch, H, W)
                
                # Place predictions back in scene
                for idx, (row, col, ph_orig, pw_orig) in enumerate(coords):
                    if save_probabilities:
                        # For probabilities, place all class channels
                        for class_idx in range(num_classes):
                            classified_scene[row:row+ph_orig, col:col+pw_orig, class_idx] = \
                                preds[idx, class_idx, :ph_orig, :pw_orig]
                    else:
                        # For labels, just place the single channel
                        classified_scene[row:row+ph_orig, col:col+pw_orig] = \
                            preds[idx, :ph_orig, :pw_orig]
                
                patches = []
                coords = []
                pbar.update(batch_size)
    
    # 6. Process remaining patches
    if patches:
        batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2)).float().to(device)
        
        with torch.no_grad():
            if save_probabilities:
                logits = model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs
            else:
                logits = model(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        for idx, (row, col, ph_orig, pw_orig) in enumerate(coords):
            if save_probabilities:
                for class_idx in range(num_classes):
                    classified_scene[row:row+ph_orig, col:col+pw_orig, class_idx] = \
                        preds[idx, class_idx, :ph_orig, :pw_orig]
            else:
                classified_scene[row:row+ph_orig, col:col+pw_orig] = \
                    preds[idx, :ph_orig, :pw_orig]
        
        pbar.update(len(patches))
    
    # 7. Save classified scene with geospatial information
    print(f"\nSaving classified scene to: {output_path}")
    
    # Update profile for output - FORCE target CRS
    output_profile = profile.copy()
    
    # Set the CRS to target CRS
    try:
        from rasterio.crs import CRS
        target_crs_obj = CRS.from_string(target_crs)
        output_profile['crs'] = target_crs_obj
        print(f"Setting output CRS to: {target_crs}")
    except Exception as e:
        print(f"Warning: Could not parse target CRS {target_crs}: {e}")
        print(f"Using input CRS: {crs}")
        output_profile['crs'] = crs
    
    if save_probabilities:
        # For probability maps: save as Float32 with multiple bands
        output_profile.update(
            dtype=rasterio.float32,
            count=num_classes,
            compress='LZW',
            nodata=np.nan
        )
        
        # Transpose back to (C, H, W) for writing
        classified_scene = classified_scene.transpose(2, 0, 1)  # (num_classes, H, W)
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(classified_scene)
            # Write band descriptions
            for i in range(num_classes):
                dst.set_band_description(i+1, f"Class_{i}_probability")
                
    else:
        # For label maps: save as UInt8
        output_profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='LZW',
            nodata=255  # Use 255 for no data
        )
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(classified_scene, 1)
            # Add colormap for visualization
            try:
                from rasterio.colormap import ColorMap
                # Create a simple colormap (adjust based on your classes)
                cmap = {i: (i*10, i*20, i*30) for i in range(num_classes)}
                dst.write_colormap(1, cmap)
            except:
                pass
    
    print("✓ Classification complete!")
    
    # 8. Save classification metadata with CRS details
    metadata_path = output_path.with_suffix('.json')
    metadata = {
        'input_scene': scene_path,
        'output_scene': str(output_path),
        'experiment_dir': str(experiment_dir),
        'model_name': model_config['model_name'],
        'sensor_name': model_config['sensor_name'],
        'patch_size': patch_size,
        'overlap': overlap,
        'batch_size': batch_size,
        'save_probabilities': save_probabilities,
        'num_classes': num_classes,
        'input_crs': str(crs) if crs else None,
        'target_crs': target_crs,
        'output_crs': str(output_profile['crs']) if 'crs' in output_profile else None,
        'transform': [transform.a, transform.b, transform.c, 
                     transform.d, transform.e, transform.f],
        'dimensions': {'height': H, 'width': W, 'channels': C},
        'processing_date': str(datetime.now()),
        'processing_time_seconds': pbar.format_dict['elapsed']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    
    # 9. Verify the saved file has correct CRS
    print(f"\nVerifying saved file CRS...")
    try:
        with rasterio.open(output_path) as dst:
            saved_crs = dst.crs
            saved_transform = dst.transform
            saved_shape = dst.shape
            saved_count = dst.count
            
        print(f"✓ Saved file verification:")
        print(f"  CRS: {saved_crs}")
        print(f"  Shape: {saved_shape}")
        print(f"  Bands: {saved_count}")
        print(f"  Transform: {saved_transform}")
        
        # Check if CRS matches target
        if saved_crs:
            saved_crs_str = str(saved_crs).upper()
            if target_crs.upper() in saved_crs_str:
                print(f"✅ CRS correctly set to {target_crs}")
            else:
                print(f"⚠️  WARNING: Saved CRS ({saved_crs}) doesn't match target ({target_crs})")
        else:
            print("⚠️  WARNING: Saved file has no CRS information!")
            
    except Exception as e:
        print(f"⚠️  Could not verify saved file: {e}")
    
    return output_path

def load_model_from_checkpoint(checkpoint_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    try:
        # Try with weights_only=False (older PyTorch compatibility)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # For older PyTorch versions that don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        # If still fails, try the safe_globals approach for PyTorch 2.6+
        print(f"Standard load failed, trying safe_globals approach: {e}")
        import numpy
        from torch.serialization import add_safe_globals
        add_safe_globals([numpy.core.multiarray.scalar])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Get model configuration from checkpoint
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        model_name = checkpoint_config['model_name']
        sensor_name = checkpoint_config['sensor_name']
        
        # Try to get num_classes from config, default to 13
        if 'dataset_config' in checkpoint_config:
            try:
                with open(checkpoint_config['dataset_config'], 'r') as f:
                    dataset_config = json.load(f)
                    num_classes = dataset_config.get('num_classes', 13)
            except:
                num_classes = 13
        else:
            num_classes = 13
    else:
        raise ValueError("Checkpoint does not contain configuration information")
    
    # Create model
    model = create_model(model_name, sensor_name, 
                        {'num_classes': num_classes})
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✓ Loaded {model_name} model for {sensor_name} ({num_classes} classes)")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best validation IoU: {checkpoint.get('best_val_iou', 'unknown'):.4f}")
    
    return model, model_name, sensor_name, num_classes

def find_experiment_dirs(base_path, sensor_name=None):
    """Find experiment directories containing trained models"""
    base_dir = Path(base_path)
    experiment_dirs = []
    
    for exp_dir in base_dir.iterdir():
        if exp_dir.is_dir() and "BasicUNet" in exp_dir.name:
            # Check if it contains a trained model
            model_path = exp_dir / "best_model.pth"
            if model_path.exists():
                # Optional: filter by sensor name
                if sensor_name and sensor_name in exp_dir.name:
                    experiment_dirs.append(exp_dir)
                elif not sensor_name:
                    experiment_dirs.append(exp_dir)
    
    return sorted(experiment_dirs)

def get_scene_path_for_sensor(sensor_name):
    """Get the predefined scene path for a sensor"""
    if sensor_name in SCENE_PATHS:
        scene_path = SCENE_PATHS[sensor_name]
        if Path(scene_path).exists():
            return scene_path
        else:
            raise FileNotFoundError(f"Scene path not found for {sensor_name}: {scene_path}")
    else:
        raise ValueError(f"Unknown sensor: {sensor_name}. Available sensors: {list(SCENE_PATHS.keys())}")

def main():
    """Main function for scene classification"""
    
    parser = argparse.ArgumentParser(description='Classify full scene with trained model')
    
    # Either provide experiment directory or let it auto-detect
    parser.add_argument('--experiment_dir', type=str, default=None,
                       help='Path to experiment directory (auto-detected if not provided)')
    
    # Sensor name is now REQUIRED to know which scene to use
    parser.add_argument('--sensor_name', type=str, required=True,
                       choices=['landsat8', 'sentinel2', 'alphaearth'],
                       help='Sensor name (determines which scene to classify)')
    
    # Classification parameters
    parser.add_argument('--patch_size', type=int, default=224,
                       help='Patch size for classification (must match training)')
    parser.add_argument('--overlap', type=int, default=0,
                       help='Overlap between patches (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--save_probabilities', action='store_true', default = True,
                       help='Save class probabilities instead of labels')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: "cuda", "cpu", or auto-detect')
    
    # Batch processing
    parser.add_argument('--process_all_experiments', action='store_true',
                       help='Process all found experiment directories for this sensor')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get the predefined scene path for this sensor
    scene_path = get_scene_path_for_sensor(args.sensor_name)
    print(f"Sensor: {args.sensor_name}")
    print(f"Scene path: {scene_path}")
    
    # Find experiment directories
    base_experiments_dir = Path("/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/experiments")
    
    if args.experiment_dir:
        # Use provided experiment directory
        experiment_dirs = [Path(args.experiment_dir)]
    elif args.process_all_experiments:
        # Process all experiment directories for this sensor
        experiment_dirs = find_experiment_dirs(base_experiments_dir, args.sensor_name)
        print(f"\nFound {len(experiment_dirs)} experiment directories for {args.sensor_name}:")
        for exp_dir in experiment_dirs:
            print(f"  - {exp_dir.name}")
    else:
        # Find matching experiment directory (most recent)
        experiment_dirs = find_experiment_dirs(base_experiments_dir, args.sensor_name)
        
        if len(experiment_dirs) == 0:
            raise ValueError(f"No experiment directory found for sensor: {args.sensor_name}")
        
        # Sort by creation time (newest first)
        experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"\nFound {len(experiment_dirs)} experiment directories for {args.sensor_name}:")
        for i, exp_dir in enumerate(experiment_dirs[:5]):  # Show top 5
            print(f"  {i+1}. {exp_dir.name} (modified: {datetime.fromtimestamp(exp_dir.stat().st_mtime)})")
        
        # Use the most recent one
        experiment_dirs = [experiment_dirs[0]]
        print(f"\nUsing most recent: {experiment_dirs[0].name}")
    
    # Process each experiment directory
    for experiment_dir in experiment_dirs:
        print(f"\n{'='*60}")
        print(f"Processing experiment: {experiment_dir.name}")
        print(f"{'='*60}")
        
        # Determine checkpoint path
        checkpoint_path = experiment_dir / "best_model.pth"
        
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
            continue
        
        # Load model
        model, model_name, sensor_name, num_classes = load_model_from_checkpoint(checkpoint_path)
        
        # Verify sensor matches
        if sensor_name != args.sensor_name:
            print(f"WARNING: Model trained on {sensor_name} but classifying {args.sensor_name} scene!")
            print(f"         This may cause issues if the number of bands doesn't match.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Skipping this experiment...")
                continue
        
        # Classify scene
        try:
            output_path = classify_full_scene(
                scene_path=scene_path,
                model=model,
                model_config={
                    'model_name': model_name,
                    'sensor_name': sensor_name,
                    'num_classes': num_classes
                },
                experiment_dir=experiment_dir,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                device=device,
                overlap=args.overlap,
                save_probabilities=args.save_probabilities,
                target_crs='EPSG:3979'  # ADD THIS LINE
            )
            
            print(f"\n✅ Scene classification completed for {experiment_dir.name}!")
            print(f"   Output saved to: {output_path}")
            
            # Also create a symbolic link for easy access
            scene_stem = Path(scene_path).stem
            link_path = experiment_dir / f"classified_{scene_stem}.tif"
            if not link_path.exists():
                try:
                    os.symlink(output_path.relative_to(experiment_dir), link_path)
                    print(f"   Symbolic link created: {link_path}")
                except:
                    pass
            
        except Exception as e:
            print(f"ERROR processing {experiment_dir.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All processing completed!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()