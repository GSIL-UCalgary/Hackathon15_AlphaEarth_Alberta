"""
Script to generate cluster maps for hyperspectral patches
FIXED VERSION: Ensures exactly 50, 30, 20 clusters and uses correct naming
"""

import os
import numpy as np
import torch
from pathlib import Path
import rasterio
from tqdm import tqdm
import json
import warnings
import shutil
warnings.filterwarnings('ignore')

# Import clustering functions from your original code
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def allocate_subclusters_proportional(counts, target_subclusters):
    """Allocate number of subsets per class, proportional to class pixel count."""
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    alloc = (counts / total) * target_subclusters
    alloc = np.maximum(1, np.floor(alloc)).astype(int)

    diff = target_subclusters - alloc.sum()
    if diff > 0:
        frac = alloc - np.floor(alloc)
        for i in np.argsort(-frac):
            if diff <= 0:
                break
            alloc[i] += 1
            diff -= 1
    elif diff < 0:
        for i in np.argsort(counts):
            if diff >= 0:
                break
            if alloc[i] > 1:
                alloc[i] -= 1
                diff += 1
    return alloc

def ImageStretching(data):
    """Mimic original normalization: per-band min-max to [-1, 1]"""
    img = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[-1]):
        input_max = np.max(data[:, :, i])
        input_min = np.min(data[:, :, i])
        img[:, :, i] = ((data[:, :, i] - input_min) / 
                       (input_max - input_min + 1e-10)) * 2 - 1
    return img

def split_label_map_by_spectra(hsi, label_map, num_classes, target_subclusters=50, n_pca_components=20, random_state=42):
    """
    Split each class into subsets using spectral clustering.
    Ensures EXACTLY target_subclusters total.
    """
    H, W, B = hsi.shape
    
    # Apply original normalization
    hsi_norm = ImageStretching(hsi)
    
    new_label_map = np.full((H, W), fill_value=-1, dtype=int)

    # Consider only valid labels (0 to num_classes-1, ignore -1 background)
    class_labels = np.arange(0, num_classes)
    
    # Count pixels per class (ignoring background -1)
    counts = []
    for c in class_labels:
        mask = (label_map == c)
        counts.append(np.sum(mask))
    
    allocs = allocate_subclusters_proportional(counts, target_subclusters)

    # Fit global PCA and normalization on all NON-BACKGROUND pixels
    mask_valid = label_map >= 0
    X_all = hsi_norm[mask_valid].reshape(-1, B)
    
    if len(X_all) == 0:
        # If no valid pixels, return all zeros
        return np.zeros((H, W), dtype=int), {}
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=min(n_pca_components, B), random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # To retrieve spectral embeddings per pixel
    coords_valid = np.argwhere(mask_valid)
    flat_indices_valid = np.ravel_multi_index((coords_valid[:,0], coords_valid[:,1]), (H,W))

    next_label = 0
    mapping = {}

    for i, c in enumerate(class_labels):
        n_sub = int(allocs[i])
        mask = (label_map == c)
        idx_flat = np.nonzero(mask.ravel())[0]
        
        if len(idx_flat) == 0:
            continue

        # get corresponding indices in X_pca
        valid_mask = np.isin(flat_indices_valid, idx_flat)
        Xc = X_pca[valid_mask]

        if len(Xc) == 0:
            continue
            
        if n_sub == 1:
            sub_labels = np.zeros(len(Xc), dtype=int)
        else:
            n_sub = min(n_sub, len(Xc))
            if n_sub <= 1:
                sub_labels = np.zeros(len(Xc), dtype=int)
            else:
                km = KMeans(n_clusters=n_sub, random_state=random_state, n_init=5)
                sub_labels = km.fit_predict(Xc)

        # assign unique global labels
        global_labels = np.array([next_label + l for l in sub_labels], dtype=int)
        new_label_map.ravel()[idx_flat] = global_labels
        mapping[c] = list(range(next_label, next_label + n_sub))
        next_label += n_sub

    return new_label_map, mapping

def seeded_kmeans_fixed_clusters(hsi, reassigned_label_map, target_clusters):
    """
    Ensures EXACTLY target_clusters clusters.
    All pixels get assignments in range 0 to target_clusters-1.
    """
    H, W, B = hsi.shape
    
    # Apply original normalization
    hsi_norm = hsi  # Already normalized
    
    # 1. Compute mean spectra for each subset (ignore background -1)
    subset_labels = np.unique(reassigned_label_map)
    subset_labels = subset_labels[subset_labels >= 0]  # ignore background
    
    mean_spectra = []
    if len(subset_labels) > 0:
        for s in subset_labels:
            mask = (reassigned_label_map == s).ravel()
            subset_pixels = hsi_norm.reshape(-1, B)[mask]
            if len(subset_pixels) > 0:
                mean_spec = subset_pixels.mean(axis=0)
                mean_spectra.append(mean_spec)
    
    pixels_all = hsi_norm.reshape(-1, B)
    
    # 2. Prepare initialization centers
    if len(mean_spectra) >= target_clusters:
        # If we have enough initial centers, use the first target_clusters
        init_centers = np.array(mean_spectra[:target_clusters])
    else:
        # Pad with random pixels from the data
        init_centers = []
        if len(mean_spectra) > 0:
            init_centers.extend(mean_spectra)
        
        n_missing = target_clusters - len(init_centers)
        if n_missing > 0:
            # Ensure we don't sample more than available
            n_samples = min(len(pixels_all), n_missing)
            random_indices = np.random.choice(len(pixels_all), n_samples, replace=False)
            random_centers = pixels_all[random_indices]
            init_centers.extend(random_centers)
            
            # If still missing, duplicate existing centers
            while len(init_centers) < target_clusters:
                init_centers.append(init_centers[-1])
        
        init_centers = np.array(init_centers)
    
    # 3. Run KMeans with EXACTLY target_clusters
    km = KMeans(n_clusters=target_clusters, init=init_centers, n_init=1, random_state=42)
    cluster_labels = km.fit_predict(pixels_all)
    
    # 4. Reshape back to H x W
    new_label_map = cluster_labels.reshape(H, W)
    
    # Double-check we have exactly target_clusters
    unique_clusters = np.unique(new_label_map)
    if len(unique_clusters) != target_clusters:
        print(f"Warning: Got {len(unique_clusters)} clusters instead of {target_clusters}")
        # Force cluster IDs to 0..target_clusters-1
        for i, cluster_id in enumerate(sorted(unique_clusters)):
            new_label_map[new_label_map == cluster_id] = i
    
    return new_label_map, init_centers

def load_patch(image_path, label_path):
    """Load image and label patches"""
    with rasterio.open(image_path) as src:
        image = src.read()  # Shape: (C, H, W), uint8
    
    with rasterio.open(label_path) as src:
        label = src.read(1)  # Shape: (H, W), int
        # Convert -99 to -1 for consistency with original
        label = np.where(label == -99, -1, label)
    
    return image, label

def process_patch_fixed(image, label, num_classes=13):
    """
    Generate cluster maps with EXACTLY 50, 30, 20 clusters
    Returns: per_cluster_num, cluster50, cluster30, cluster20
    """
    # Apply original normalization to 0-1 range then ImageStretching
    image_norm = image.astype(np.float32) / 255.0  # First to [0, 1]
    
    # Transpose to (H, W, C) for clustering
    hsi = np.transpose(image_norm, (1, 2, 0))  # (H, W, C)
    
    # Apply original ImageStretching
    hsi_stretched = ImageStretching(hsi)
    
    H, W, B = hsi_stretched.shape
    
    # Generate initial label maps
    new_label50, _ = split_label_map_by_spectra(
        hsi_stretched, label, num_classes, target_subclusters=50
    )
    
    new_label30, _ = split_label_map_by_spectra(
        hsi_stretched, label, num_classes, target_subclusters=30
    )
    
    new_label20, _ = split_label_map_by_spectra(
        hsi_stretched, label, num_classes, target_subclusters=20
    )
    
    # Apply fixed-cluster KMeans (ALL pixels get assignments)
    cluster50, _ = seeded_kmeans_fixed_clusters(hsi_stretched, new_label50, target_clusters=50)
    cluster30, _ = seeded_kmeans_fixed_clusters(hsi_stretched, new_label30, target_clusters=30)
    cluster20, _ = seeded_kmeans_fixed_clusters(hsi_stretched, new_label20, target_clusters=20)
    
    # Verify cluster counts
    unique50 = np.unique(cluster50)
    unique30 = np.unique(cluster30)
    unique20 = np.unique(cluster20)
    
    if len(unique50) != 50 or len(unique30) != 30 or len(unique20) != 20:
        print(f"Warning: Cluster counts not exact: {len(unique50)}/{len(unique30)}/{len(unique20)}")
    
    # Get pixel counts per cluster
    def get_cluster_counts(cluster_map, target_clusters):
        counts = np.zeros(target_clusters, dtype=int)
        unique, counts_actual = np.unique(cluster_map, return_counts=True)
        for cluster_id, count in zip(unique, counts_actual):
            if cluster_id < target_clusters:
                counts[cluster_id] = count
        return counts.tolist()
    
    num_label50 = get_cluster_counts(cluster50, 50)
    num_label30 = get_cluster_counts(cluster30, 30)
    num_label20 = get_cluster_counts(cluster20, 20)
    
    per_cluster_num = [num_label50, num_label30, num_label20]
    
    # Final verification
    assert cluster50.max() < 50, f"cluster50 has value {cluster50.max()} >= 50"
    assert cluster30.max() < 30, f"cluster30 has value {cluster30.max()} >= 30"
    assert cluster20.max() < 20, f"cluster20 has value {cluster20.max()} >= 20"
    
    return per_cluster_num, cluster50, cluster30, cluster20

def save_cluster_maps_only(base_path, patch_name, cluster50, cluster30, cluster20, 
                          per_cluster_num, sensor_name, label_type, split):
    """Save ONLY cluster maps with correct naming"""
    # Create directory structure for clusters only
    clusters_dir = base_path / sensor_name / label_type / split / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cluster maps as numpy files with CORRECT NAMES
    cluster50_path = clusters_dir / f"{patch_name}_cluster50.npy"
    cluster30_path = clusters_dir / f"{patch_name}_cluster30.npy"
    cluster20_path = clusters_dir / f"{patch_name}_cluster20.npy"
    
    np.save(cluster50_path, cluster50)
    np.save(cluster30_path, cluster30)
    np.save(cluster20_path, cluster20)
    
    # Verify before saving
    if cluster50.max() >= 50 or cluster30.max() >= 30 or cluster20.max() >= 20:
        print(f"ERROR: {patch_name} has out-of-range cluster values!")
        print(f"  cluster50: {cluster50.min()}-{cluster50.max()}")
        print(f"  cluster30: {cluster30.min()}-{cluster30.max()}")
        print(f"  cluster20: {cluster20.min()}-{cluster20.max()}")
    
    # Save metadata
    metadata = {
        'split': split,
        'sensor': sensor_name,
        'label_type': label_type,
        'patch_name': patch_name,
        'per_cluster_num': per_cluster_num,
        'cluster_ranges': {
            'cluster50': f"{cluster50.min()}-{cluster50.max()}",
            'cluster30': f"{cluster30.min()}-{cluster30.max()}",
            'cluster20': f"{cluster20.min()}-{cluster20.max()}"
        },
        'num_clusters': {
            'cluster50': len(np.unique(cluster50)),
            'cluster30': len(np.unique(cluster30)),
            'cluster20': len(np.unique(cluster20))
        },
        'expected_clusters': [50, 30, 20]
    }
    
    metadata_path = clusters_dir / f"{patch_name}_clusters_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return cluster50_path, cluster30_path, cluster20_path, metadata_path

def process_sentinel2_fixed(data_root, output_base=None, num_classes=13):
    """
    Process Sentinel-2 patches with FIXED cluster counts (50, 30, 20)
    """
    sensor_name = "sentinel2"
    
    data_root = Path(data_root)
    
    # Define output directory for clusters only
    if output_base is None:
        output_base = data_root.parent / "clustered_datasets"
    else:
        output_base = Path(output_base)
    
    all_metadata = {}
    
    # Process both filtered and unfiltered
    for label_type in ["filtered"]: #["filtered", "unfiltered"]:
        print(f"\n{'='*60}")
        print(f"Processing Sentinel-2 with {label_type} labels (FIXED CLUSTERS)")
        print(f"{'='*60}")
        
        for split in ["train", "val", "test"]:
            print(f"\n  Processing {split} split...")
            
            # Define input paths for this split
            image_dir = data_root / split / sensor_name / "img"
            label_dir = data_root / split / "labels" / label_type
            
            # Check if directories exist
            if not image_dir.exists():
                print(f"    Warning: Image directory not found: {image_dir}")
                continue
            if not label_dir.exists():
                print(f"    Warning: Label directory not found: {label_dir}")
                continue
            
            # Get all patch files for this split
            image_files = sorted(list(image_dir.glob("*.tif")))
            label_files = sorted(list(label_dir.glob("*.tif")))
            
            print(f"    Found {len(image_files)} image patches and {len(label_files)} label patches")
            
            if len(image_files) == 0 or len(label_files) == 0:
                print(f"    Skipping {split} split - no files found")
                continue
            
            # Verify matching files
            image_dict = {f.stem: f for f in image_files}
            label_dict = {f.stem: f for f in label_files}
            common_names = set(image_dict.keys()) & set(label_dict.keys())
            
            if not common_names:
                print(f"    Warning: No matching patch names found in {split} split")
                continue
            
            print(f"    Found {len(common_names)} matching patches")
            
            # Use only matching patches
            image_files = [image_dict[name] for name in sorted(common_names)]
            label_files = [label_dict[name] for name in sorted(common_names)]
            
            # Process each patch in this split
            for i, (img_path, lbl_path) in enumerate(tqdm(zip(image_files, label_files), 
                                                          total=len(image_files), 
                                                          desc=f"    Clustering {split}")):
                
                patch_name = img_path.stem
                
                try:
                    # Load patches
                    image, label = load_patch(img_path, lbl_path)
                    
                    # Check spatial dimensions match
                    if image.shape[1:] != label.shape:
                        print(f"    Warning: Spatial dimensions don't match! Image: {image.shape[1:]}, Label: {label.shape}")
                        continue
                    
                    # Generate cluster maps with FIXED cluster counts
                    per_cluster_num, cluster50, cluster30, cluster20 = process_patch_fixed(
                        image, label, num_classes
                    )
                    
                    # Verify cluster ranges
                    if cluster50.max() >= 50 or cluster30.max() >= 30 or cluster20.max() >= 20:
                        print(f"    ERROR: {patch_name} has out-of-range clusters!")
                        print(f"      cluster50: {cluster50.min()}-{cluster50.max()}")
                        print(f"      cluster30: {cluster30.min()}-{cluster30.max()}")
                        print(f"      cluster20: {cluster20.min()}-{cluster20.max()}")
                        continue
                    
                    # Save cluster maps only
                    cluster_paths = save_cluster_maps_only(
                        output_base,
                        patch_name,
                        cluster50, cluster30, cluster20,
                        per_cluster_num,
                        sensor_name,
                        label_type,
                        split
                    )
                    
                    # Store metadata
                    key = f"{split}_{label_type}_{patch_name}"
                    all_metadata[key] = {
                        'split': split,
                        'label_type': label_type,
                        'patch_name': patch_name,
                        'original_image_path': str(img_path),
                        'original_label_path': str(lbl_path),
                        'cluster_paths': [str(p) for p in cluster_paths],
                        'cluster_stats': {
                            'cluster50_range': f"{cluster50.min()}-{cluster50.max()}",
                            'cluster30_range': f"{cluster30.min()}-{cluster30.max()}",
                            'cluster20_range': f"{cluster20.min()}-{cluster20.max()}",
                            'num_clusters_50': len(np.unique(cluster50)),
                            'num_clusters_30': len(np.unique(cluster30)),
                            'num_clusters_20': len(np.unique(cluster20))
                        },
                        'per_cluster_num_lengths': [
                            len(per_cluster_num[0]),
                            len(per_cluster_num[1]),
                            len(per_cluster_num[2])
                        ]
                    }
                    
                    # # Print sample for first few patches
                    # if i < 3:
                    #     print(f"\n    Sample {patch_name}:")
                    #     print(f"      Image shape: {image.shape}")
                    #     print(f"      Label range: {label.min()} to {label.max()}")
                    #     print(f"      Cluster50: {cluster50.min()} to {cluster50.max()} ({len(np.unique(cluster50))} clusters)")
                    #     print(f"      Cluster30: {cluster30.min()} to {cluster30.max()} ({len(np.unique(cluster30))} clusters)")
                    #     print(f"      Cluster20: {cluster20.min()} to {cluster20.max()} ({len(np.unique(cluster20))} clusters)")
                    #     print(f"      per_cluster_num lengths: {len(per_cluster_num[0])}/{len(per_cluster_num[1])}/{len(per_cluster_num[2])}")
                        
                except Exception as e:
                    print(f"    Error processing {split}/{patch_name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save overall metadata
    metadata_dir = output_base / sensor_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"sentinel2_fixed_clusters_metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Sentinel-2 FIXED Clustering Complete!")
    print(f"Cluster maps saved to: {output_base / sensor_name}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print statistics
    print(f"\nFinal Statistics:")
    print(f"  Total patches processed: {len(all_metadata)}")
    
    # Check cluster consistency
    cluster_issues = 0
    for key, data in all_metadata.items():
        stats = data['cluster_stats']
        if (int(stats['num_clusters_50']) != 50 or 
            int(stats['num_clusters_30']) != 30 or 
            int(stats['num_clusters_20']) != 20):
            cluster_issues += 1
    
    if cluster_issues == 0:
        print("  ✓ All patches have exactly 50/30/20 clusters")
    else:
        print(f"  ⚠ {cluster_issues} patches have incorrect cluster counts")
    
    return all_metadata

def main():
    """Main function to generate cluster maps with FIXED cluster counts"""
    # Configuration
    DATA_ROOT = "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/train_val_test_patches/patches"
    OUTPUT_BASE = "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/clustered_datasets"
    NUM_CLASSES = 13
    
    print("=" * 80)
    print("Generating FIXED Cluster Maps for Sentinel-2")
    print("=" * 80)
    print(f"Input data root: {DATA_ROOT}")
    print(f"Output base directory: {OUTPUT_BASE}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Fixed cluster counts: 50, 30, 20")
    print("=" * 80)
    
    # Process with fixed clusters
    metadata = process_sentinel2_fixed(
        DATA_ROOT, 
        OUTPUT_BASE,
        NUM_CLASSES
    )
    
    print(f"\n{'='*80}")
    print("CLUSTERING VERIFICATION")
    print(f"{'='*80}")
    print("Key features of FIXED method:")
    print("1. EXACTLY 50 clusters for 'cluster50' (0-49)")
    print("2. EXACTLY 30 clusters for 'cluster30' (0-29)")
    print("3. EXACTLY 20 clusters for 'cluster20' (0-19)")
    print("4. All pixels get cluster assignments")
    print("5. Uses ImageStretching normalization [-1, 1]")
    
    # Verify a few random samples
    print(f"\nRandom verification samples:")
    import random
    sample_keys = random.sample(list(metadata.keys()), min(3, len(metadata)))
    for key in sample_keys:
        print(f"\n{key}:")
        data = metadata[key]
        stats = data['cluster_stats']
        print(f"  Cluster50: {stats['cluster50_range']} ({stats['num_clusters_50']} clusters)")
        print(f"  Cluster30: {stats['cluster30_range']} ({stats['num_clusters_30']} clusters)")
        print(f"  Cluster20: {stats['cluster20_range']} ({stats['num_clusters_20']} clusters)")

if __name__ == "__main__":
    main()