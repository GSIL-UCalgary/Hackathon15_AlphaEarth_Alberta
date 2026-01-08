# In this code, train, val, and test identical patches
# are extracted from Landsat-8, Sentinel-2, and AlphaEarth images
import numpy as np
from osgeo import gdal, gdalconst, osr
import os
import math
from tqdm import tqdm
from numba import jit, prange
import warnings
import json
warnings.filterwarnings('ignore')

# Class definitions for Alberta
CLASS_DEFINITIONS = {
    0: {'name': 'Unknown', 'color': (0, 0, 0)},
    1: {'name': 'Temperate needleleaf forest', 'color': (0, 61, 0)},
    2: {'name': 'Sub-polar taiga forest', 'color': (148, 156, 112)},
    5: {'name': 'Temperate broadleaf forest', 'color': (20, 140, 61)},
    6: {'name': 'Mixed forest', 'color': (91, 117, 43)},
    8: {'name': 'Temperate shrubland', 'color': (179, 138, 51)},
    10: {'name': 'Temperate grassland', 'color': (225, 207, 138)},
    11: {'name': 'Polar shrubland-lichen', 'color': (156, 117, 84)},
    12: {'name': 'Polar grassland-lichen', 'color': (186, 212, 143)},
    13: {'name': 'Polar barren-lichen', 'color': (64, 138, 112)},
    14: {'name': 'Wetland', 'color': (107, 163, 138)},
    15: {'name': 'Cropland', 'color': (230, 174, 102)},
    16: {'name': 'Barren lands', 'color': (168, 171, 174)},
    17: {'name': 'Urban', 'color': (220, 33, 38)},
    18: {'name': 'Water', 'color': (76, 112, 163)},
    19: {'name': 'Snow/ice', 'color': (255, 250, 255)}
}

def create_alberta_label_mapping(alberta_classes):
    """Create mapping from original Alberta classes to 0-based continuous labels"""
    # Sort the classes to ensure consistent mapping
    alberta_classes_sorted = sorted(alberta_classes)
    
    # Create mapping dictionaries
    original_to_new = {}  # original -> new (0-based)
    new_to_original = {}  # new -> original
    new_class_definitions = {}
    
    # Map 0 (Unknown/NoData) to -99 for background
    original_to_new[0] = -99
    new_class_definitions[-99] = {
        'name': 'Background/NoData',
        'color': (0, 0, 0),
        'original_id': 0
    }
    
    # Map Alberta classes to 0, 1, 2, ..., N-1
    for new_label, original_class in enumerate(alberta_classes_sorted):
        original_to_new[original_class] = new_label
        new_to_original[new_label] = original_class
        
        # Create new class definition
        if original_class in CLASS_DEFINITIONS:
            new_class_definitions[new_label] = {
                'name': CLASS_DEFINITIONS[original_class]['name'],
                'color': CLASS_DEFINITIONS[original_class]['color'],
                'original_id': original_class
            }
        else:
            new_class_definitions[new_label] = {
                'name': f'Class_{original_class}',
                'color': (0, 0, 0),
                'original_id': original_class
            }
    
    return original_to_new, new_to_original, new_class_definitions, alberta_classes_sorted

def remap_ground_truth(input_path, output_path, label_mapping):
    """Remap ground truth image using the label mapping"""
    print(f"Remapping ground truth labels...")
    
    # Open input file
    src_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise ValueError(f"Could not open input file: {input_path}")
    
    # Get dimensions and metadata
    rows = src_ds.RasterYSize
    cols = src_ds.RasterXSize
    gt = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    
    # Read data
    band = src_ds.GetRasterBand(1)
    data = band.ReadAsArray()
    
    # Create output array with remapped values
    remapped_data = np.full_like(data, -99, dtype=np.int16)  # Default to background (-99)
    
    # Apply remapping using vectorized operations for speed
    for original_class, new_label in label_mapping.items():
        remapped_data[data == original_class] = new_label
    
    # Close input
    src_ds = None
    
    # Create output file
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Int16,
                         options=['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(remapped_data)
    out_band.SetNoDataValue(-99)
    
    # Close output
    out_ds.FlushCache()
    out_ds = None
    
    print(f"  Remapped ground truth saved to: {output_path}")
    return output_path

@jit(nopython=True, parallel=True)
def apply_strict_majority_filter_numba(lc_array, min_homogeneity=0.8):
    """Numba-optimized strict majority filter that ignores background (-99) values"""
    rows, cols = lc_array.shape
    filtered = np.full_like(lc_array, -99)  # Use -99 for background
    kernel_size = 9
    half_kernel = kernel_size // 2
    min_required = math.ceil(kernel_size**2 * min_homogeneity)
    
    for i in prange(rows):
        for j in range(cols):
            current_val = lc_array[i,j]
            if current_val == -99:  # Skip background
                filtered[i,j] = -99
                continue
                
            # Get window bounds
            y_start = max(0, i - half_kernel)
            y_end = min(rows, i + half_kernel + 1)
            x_start = max(0, j - half_kernel)
            x_end = min(cols, j + half_kernel + 1)
            
            # Count occurrences of current class (ignoring background)
            count = 0
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    if lc_array[y,x] == current_val and lc_array[y,x] != -99:
                        count += 1
            
            # Only keep if meets homogeneity threshold
            filtered[i,j] = current_val if count >= min_required else -99
    
    return filtered

def apply_strict_majority_filter_batched(input_path, output_path, min_homogeneity=0.8, batch_size=2048):
    """Memory-optimized strict majority filter using batch processing"""
    print("Loading remapped land cover data for strict filtering...")
    src_ds = gdal.Open(input_path)
    
    # Get dimensions and metadata
    rows = src_ds.RasterYSize
    cols = src_ds.RasterXSize
    gt = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    
    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Int16, 
                         options=['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(-99)
    
    # Process in batches with overlap for edge handling
    kernel_size = 9
    overlap = kernel_size // 2
    
    print(f"Processing in batches of {batch_size}x{batch_size} with {overlap} pixel overlap...")
    
    # Track statistics
    total_pixels_removed = 0
    total_pixels_processed = 0
    
    # Calculate number of batches
    num_batches_y = math.ceil(rows / batch_size)
    num_batches_x = math.ceil(cols / batch_size)
    
    for batch_y in tqdm(range(num_batches_y), desc="Processing batches"):
        for batch_x in range(num_batches_x):
            # Calculate batch boundaries with overlap
            y_start = batch_y * batch_size
            y_end = min((batch_y + 1) * batch_size, rows)
            x_start = batch_x * batch_size  
            x_end = min((batch_x + 1) * batch_size, cols)
            
            # Add overlap for edge processing
            y_start_read = max(0, y_start - overlap)
            y_end_read = min(rows, y_end + overlap)
            x_start_read = max(0, x_start - overlap)
            x_end_read = min(cols, x_end + overlap)
            
            # Read batch with overlap
            batch_array = src_ds.GetRasterBand(1).ReadAsArray(
                x_start_read, y_start_read, 
                x_end_read - x_start_read, 
                y_end_read - y_start_read
            )
            
            if batch_array is None:
                continue
                
            # Apply filter to batch
            filtered_batch = apply_strict_majority_filter_numba(batch_array, min_homogeneity)
            
            # Extract the core region (without overlap) for writing
            core_y_start = y_start - y_start_read
            core_y_end = core_y_start + (y_end - y_start)
            core_x_start = x_start - x_start_read
            core_x_end = core_x_start + (x_end - x_start)
            
            core_filtered = filtered_batch[core_y_start:core_y_end, core_x_start:core_x_end]
            core_original = batch_array[core_y_start:core_y_end, core_x_start:core_x_end]
            
            # Update statistics (ignore background pixels)
            pixels_removed = np.sum((core_original != -99) & (core_filtered == -99))
            pixels_processed = np.sum(core_original != -99)
            total_pixels_removed += pixels_removed
            total_pixels_processed += pixels_processed
            
            # Write the core region to output
            out_band.WriteArray(core_filtered, x_start, y_start)
    
    # Final statistics
    print(f"Removed {total_pixels_removed:,} non-homogeneous pixels ({total_pixels_removed/max(total_pixels_processed,1):.1%})")
    
    # Cleanup
    out_ds.FlushCache()
    src_ds = out_ds = None
    
    return output_path

@jit(nopython=True)
def calculate_abundance_batch_numba(lc_batch, class_id, window_size, stride, 
                                   batch_start_y, batch_start_x, out_rows, out_cols):
    """Numba-optimized abundance calculation for a batch"""
    lc_rows, lc_cols = lc_batch.shape
    
    # Calculate which abundance cells this batch affects
    start_i = max(0, batch_start_y // stride)
    end_i = min(out_rows, (batch_start_y + lc_rows) // stride + 1)
    start_j = max(0, batch_start_x // stride)  
    end_j = min(out_cols, (batch_start_x + lc_cols) // stride + 1)
    
    abundance = np.zeros((end_i - start_i, end_j - start_j), dtype=np.int32)
    
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            # Convert to original image coordinates
            y_start_orig = i * stride
            x_start_orig = j * stride
            y_end_orig = min(y_start_orig + window_size, batch_start_y + lc_rows)
            x_end_orig = min(x_start_orig + window_size, batch_start_x + lc_cols)
            
            # Convert to batch coordinates
            y_start_batch = max(0, y_start_orig - batch_start_y)
            x_start_batch = max(0, x_start_orig - batch_start_x)
            y_end_batch = min(lc_rows, y_end_orig - batch_start_y)
            x_end_batch = min(lc_cols, x_end_orig - batch_start_x)
            
            if (y_start_batch < y_end_batch and x_start_batch < x_end_batch):
                window = lc_batch[y_start_batch:y_end_batch, x_start_batch:x_end_batch]
                abundance[i - start_i, j - start_j] = np.sum(window == class_id)
    
    return abundance, start_i, start_j

def create_abundance_maps_batched(lc_path, output_dir, window_size=224, overlap=20, batch_size=2048):
    """Create abundance maps for each class using batch processing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Open dataset
    src_ds = gdal.Open(lc_path)
    rows = src_ds.RasterYSize
    cols = src_ds.RasterXSize
    gt = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    
    print(f"Land cover dimensions: {rows} x {cols}")
    print(f"Window size: {window_size}, Overlap: {overlap}")
    
    # Calculate output dimensions
    stride = window_size - overlap
    max_valid_row = max(0, rows - window_size)
    max_valid_col = max(0, cols - window_size)
    out_rows = max(1, (max_valid_row // stride) + 1)
    out_cols = max(1, (max_valid_col // stride) + 1)
    
    print(f"Abundance map dimensions: {out_rows} x {out_cols}")
    
    # Adjust geotransform for output
    out_gt = list(gt)
    out_gt[1] = gt[1] * stride
    out_gt[5] = gt[5] * stride
    
    # Get classes by scanning in batches (ignore background -99)
    print("Scanning for unique classes...")
    classes = set()
    num_batches_y = math.ceil(rows / batch_size)
    num_batches_x = math.ceil(cols / batch_size)
    
    for batch_y in range(num_batches_y):
        for batch_x in range(num_batches_x):
            y_start = batch_y * batch_size
            y_end = min((batch_y + 1) * batch_size, rows)
            x_start = batch_x * batch_size
            x_end = min((batch_x + 1) * batch_size, cols)
            
            batch_array = src_ds.GetRasterBand(1).ReadAsArray(x_start, y_start, 
                                                             x_end - x_start, y_end - y_start)
            if batch_array is not None:
                batch_classes = np.unique(batch_array)
                # Filter out background (-99) and keep only valid classes
                valid_classes = batch_classes[(batch_classes != -99)]
                classes.update(valid_classes)
    
    classes = sorted([int(c) for c in classes])
    print(f"Found {len(classes)} classes after remapping: {classes}")
    
    # Process each class
    for class_id in tqdm(classes, desc="Creating abundance maps"):
        abundance_path = os.path.join(output_dir, f"class_{class_id}_abundance.tif")
        
        # Skip if already exists
        if os.path.exists(abundance_path):
            continue
            
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(abundance_path, out_cols, out_rows, 1, gdal.GDT_Int32,
                             options=['COMPRESS=LZW', 'TILED=YES'])
        out_ds.SetGeoTransform(out_gt)
        out_ds.SetProjection(projection)
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(-9999)
        
        full_abundance = np.zeros((out_rows, out_cols), dtype=np.int32)
        
        # Process in batches
        for batch_y in range(num_batches_y):
            for batch_x in range(num_batches_x):
                y_start = batch_y * batch_size
                y_end = min((batch_y + 1) * batch_size, rows)
                x_start = batch_x * batch_size
                x_end = min((batch_x + 1) * batch_size, cols)
                
                batch_array = src_ds.GetRasterBand(1).ReadAsArray(x_start, y_start,
                                                                 x_end - x_start, y_end - y_start)
                
                if batch_array is None:
                    continue
                
                # Calculate abundance for this batch
                batch_abundance, start_i, start_j = calculate_abundance_batch_numba(
                    batch_array, class_id, window_size, stride, y_start, x_start, out_rows, out_cols
                )
                
                # Add to full abundance map
                end_i = start_i + batch_abundance.shape[0]
                end_j = start_j + batch_abundance.shape[1]
                if end_i <= out_rows and end_j <= out_cols:
                    full_abundance[start_i:end_i, start_j:end_j] += batch_abundance
        
        # Write abundance map
        out_band.WriteArray(full_abundance)
        out_ds.FlushCache()
        out_ds = None
    
    src_ds = None
    return classes, stride, rows, cols

def select_and_split_patches_stratified(abundance_dir, classes, lc_rows, lc_cols, 
                                      window_size, stride, min_patches_per_class=100,
                                      train_ratio=0.7, val_ratio=0.15):
    """Select patches with stratified sampling and split into train/val/test"""
    import heapq
    import random
    
    print(f"Selecting patches with min {min_patches_per_class} per class")
    print(f"Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={1-train_ratio-val_ratio:.0%}")
    
    all_splits = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for class_id in tqdm(classes, desc="Selecting patches by class"):
        path = os.path.join(abundance_dir, f"class_{class_id}_abundance.tif")
        ds = gdal.Open(path)
        
        if ds is None:
            all_splits['train'][class_id] = []
            all_splits['val'][class_id] = []
            all_splits['test'][class_id] = []
            continue
        
        rows = ds.RasterYSize
        cols = ds.RasterXSize
        
        # Collect all valid patches
        all_patches = []
        chunk_size = 500
        
        for i_start in range(0, rows, chunk_size):
            i_end = min(i_start + chunk_size, rows)
            chunk = ds.GetRasterBand(1).ReadAsArray(0, i_start, cols, i_end - i_start)
            
            if chunk is None:
                continue
                
            for local_i in range(chunk.shape[0]):
                for j in range(chunk.shape[1]):
                    abundance_val = chunk[local_i, j]
                    i = i_start + local_i
                    
                    if abundance_val > 0:
                        y_start = i * stride
                        x_start = j * stride
                        
                        if (y_start + window_size <= lc_rows and 
                            x_start + window_size <= lc_cols):
                            
                            all_patches.append((abundance_val, i, j))
        
        ds = None
        
        # Sort by abundance and take top patches
        all_patches.sort(key=lambda x: x[0], reverse=True)
        available_patches = len(all_patches)
        patches_to_select = min(available_patches, min_patches_per_class)
        
        if patches_to_select == 0:
            all_splits['train'][class_id] = []
            all_splits['val'][class_id] = []
            all_splits['test'][class_id] = []
            print(f"Class {class_id}: No patches available")
            continue
        
        selected_patches = all_patches[:patches_to_select]
        random.shuffle(selected_patches)
        
        # Split into train/val/test
        n_patches = len(selected_patches)
        n_train = int(n_patches * train_ratio)
        n_val = int(n_patches * val_ratio)
        n_test = n_patches - n_train - n_val
        
        # Ensure at least 1 patch in each split if possible
        if n_test == 0 and n_patches >= 3:
            n_train -= 1
            n_test = 1
        if n_val == 0 and n_patches >= 2:
            n_train -= 1
            n_val = 1
        
        train_patches = [(i, j) for _, i, j in selected_patches[:n_train]]
        val_patches = [(i, j) for _, i, j in selected_patches[n_train:n_train + n_val]]
        test_patches = [(i, j) for _, i, j in selected_patches[n_train + n_val:]]
        
        all_splits['train'][class_id] = train_patches
        all_splits['val'][class_id] = val_patches
        all_splits['test'][class_id] = test_patches
        
        print(f"Class {class_id}: {len(train_patches)} train, {len(val_patches)} val, {len(test_patches)} test")
    
    return all_splits

def extract_patches_multisensor(sensor_paths, lc_filtered_path, lc_unfiltered_path,
                               patch_locations, window_size, stride, output_dir, split_name):
    """Extract patches from multiple sensors and organize into sensor-specific folders"""
    from tqdm import tqdm
    
    # Create organized directory structure for each sensor
    sensor_extracted_counts = {}
    
    for sensor_name, sensor_path in sensor_paths.items():
        # Create directories for this sensor
        sensor_img_dir = os.path.join(output_dir, split_name, sensor_name, 'img')
        os.makedirs(sensor_img_dir, exist_ok=True)
        
        # Get number of bands for this sensor
        try:
            sensor_ds = gdal.Open(sensor_path, gdal.GA_ReadOnly)
            if sensor_ds is None:
                print(f"Warning: Could not open sensor image for {sensor_name}")
                continue
            num_bands = sensor_ds.RasterCount
            sensor_ds = None
        except:
            print(f"Warning: Could not determine bands for {sensor_name}, skipping")
            continue
        
        print(f"\n  Extracting {split_name} patches for {sensor_name} ({num_bands} bands)...")
        
        # Extract patches for this sensor
        extracted = extract_patches_for_sensor(
            sensor_path, lc_filtered_path, lc_unfiltered_path,
            patch_locations, window_size, stride, 
            sensor_img_dir, split_name, sensor_name, num_bands
        )
        
        sensor_extracted_counts[sensor_name] = extracted
        print(f"    Extracted {extracted} patches for {sensor_name}")
    
    # Create common label directory (same labels for all sensors)
    if sensor_extracted_counts:  # Only create labels if at least one sensor was processed
        print(f"\n  Creating common label patches for {split_name}...")
        label_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(label_dir, exist_ok=True)
        
        extract_label_patches(
            lc_filtered_path, lc_unfiltered_path,
            patch_locations, window_size, stride, 
            label_dir, split_name
        )
    
    return sensor_extracted_counts

def extract_patches_for_sensor(sensor_path, lc_filtered_path, lc_unfiltered_path,
                              patch_locations, window_size, stride, 
                              output_dir, split_name, sensor_name, num_bands):
    """Extract patches for a specific sensor"""
    # Open land cover datasets
    lc_filtered_ds = gdal.Open(lc_filtered_path, gdal.GA_ReadOnly)
    lc_unfiltered_ds = gdal.Open(lc_unfiltered_path, gdal.GA_ReadOnly)
    
    if lc_filtered_ds is None or lc_unfiltered_ds is None:
        raise ValueError("Failed to open land cover datasets")
    
    total_extracted = 0
    
    for class_id, coords in tqdm(patch_locations.items(), desc=f"{sensor_name}"):
        if not coords:
            continue
        
        # Process each patch
        for patch_idx, (i, j) in enumerate(coords):
            try:
                x_offset = int(j * stride)
                y_offset = int(i * stride)
                
                # Open sensor image for this patch
                sensor_ds = gdal.Open(sensor_path, gdal.GA_ReadOnly)
                if sensor_ds is None:
                    raise ValueError(f"Could not open sensor image: {sensor_path}")
                
                # Read geotransform and projection
                gt = sensor_ds.GetGeoTransform()
                projection = sensor_ds.GetProjection()
                
                # Read all bands for this sensor
                bands_data = []
                for band_num in range(1, num_bands + 1):
                    band = sensor_ds.GetRasterBand(band_num)
                    patch = band.ReadAsArray(x_offset, y_offset, window_size, window_size)
                    if patch is None or patch.shape != (window_size, window_size):
                        raise ValueError(f"Invalid band {band_num} for sensor {sensor_name}")
                    bands_data.append(patch)
                
                sensor_ds = None
                
                # Stack bands and ensure UInt8 format
                image_patch = np.stack(bands_data, axis=0)
                if image_patch.dtype != np.uint8:
                    # Convert to UInt8 if needed (clip to 0-255 range)
                    image_patch = np.clip(image_patch, 0, 255).astype(np.uint8)
                
                # Create patch geotransform
                patch_gt = (
                    gt[0] + x_offset * gt[1],
                    gt[1], gt[2],
                    gt[3] + y_offset * gt[5],
                    gt[4], gt[5]
                )
                
                # Create patch filename
                patch_filename = f"class_{class_id}_patch_{patch_idx}.tif"
                
                # Save image patch for this sensor
                save_geotiff_uint8(image_patch, 
                                 os.path.join(output_dir, patch_filename),
                                 patch_gt, projection)
                
                total_extracted += 1
                
            except Exception as e:
                print(f"Error extracting patch {patch_idx} for class {class_id}, sensor {sensor_name}: {str(e)}")
                continue
    
    lc_filtered_ds = lc_unfiltered_ds = None
    return total_extracted

def extract_label_patches(lc_filtered_path, lc_unfiltered_path,
                         patch_locations, window_size, stride, 
                         output_dir, split_name):
    """Extract label patches (common for all sensors)"""
    from tqdm import tqdm
    
    # Open land cover datasets
    lc_filtered_ds = gdal.Open(lc_filtered_path, gdal.GA_ReadOnly)
    lc_unfiltered_ds = gdal.Open(lc_unfiltered_path, gdal.GA_ReadOnly)
    
    if lc_filtered_ds is None or lc_unfiltered_ds is None:
        raise ValueError("Failed to open land cover datasets")
    
    # Read geotransform and projection from filtered dataset
    gt = lc_filtered_ds.GetGeoTransform()
    projection = lc_filtered_ds.GetProjection()
    
    filtered_dir = os.path.join(output_dir, 'filtered')
    unfiltered_dir = os.path.join(output_dir, 'unfiltered')
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(unfiltered_dir, exist_ok=True)
    
    for class_id, coords in tqdm(patch_locations.items(), desc="Labels"):
        if not coords:
            continue
        
        # Process each patch
        for patch_idx, (i, j) in enumerate(coords):
            try:
                x_offset = int(j * stride)
                y_offset = int(i * stride)
                
                # Read land cover patches
                lc_filtered_patch = lc_filtered_ds.GetRasterBand(1).ReadAsArray(
                    x_offset, y_offset, window_size, window_size)
                
                lc_unfiltered_patch = lc_unfiltered_ds.GetRasterBand(1).ReadAsArray(
                    x_offset, y_offset, window_size, window_size)
                
                if (lc_filtered_patch is None or lc_unfiltered_patch is None or
                    lc_filtered_patch.shape != (window_size, window_size)):
                    continue
                
                # Create patch geotransform
                patch_gt = (
                    gt[0] + x_offset * gt[1],
                    gt[1], gt[2],
                    gt[3] + y_offset * gt[5],
                    gt[4], gt[5]
                )
                
                # Create patch filename
                patch_filename = f"class_{class_id}_patch_{patch_idx}.tif"
                
                # Save filtered label patch
                save_geotiff_int16(lc_filtered_patch.astype(np.int16),
                                 os.path.join(filtered_dir, patch_filename),
                                 patch_gt, projection)
                
                # Save unfiltered label patch
                save_geotiff_int16(lc_unfiltered_patch.astype(np.int16),
                                 os.path.join(unfiltered_dir, patch_filename),
                                 patch_gt, projection)
                
            except Exception as e:
                print(f"Error extracting label patch {patch_idx} for class {class_id}: {str(e)}")
                continue
    
    lc_filtered_ds = lc_unfiltered_ds = None

def save_geotiff_uint8(array, path, geotransform, projection):
    """Save UInt8 array as GeoTIFF"""
    driver = gdal.GetDriverByName('GTiff')
    if array.ndim == 2:
        bands = 1
        height, width = array.shape
    else:
        bands, height, width = array.shape
    
    options = ['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    if height * width > 1000000:
        options.append('BIGTIFF=YES')
    
    ds = driver.Create(path, width, height, bands, gdal.GDT_Byte, options=options)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    
    if array.ndim == 2:
        ds.GetRasterBand(1).WriteArray(array)
    else:
        for b in range(bands):
            ds.GetRasterBand(b+1).WriteArray(array[b])
    
    ds.FlushCache()
    ds = None

def save_geotiff_int16(array, path, geotransform, projection):
    """Save Int16 array as GeoTIFF with -99 as nodata"""
    driver = gdal.GetDriverByName('GTiff')
    height, width = array.shape
    
    options = ['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    if height * width > 1000000:
        options.append('BIGTIFF=YES')
    
    ds = driver.Create(path, width, height, 1, gdal.GDT_Int16, options=options)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    
    band = ds.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(-99)
    
    ds.FlushCache()
    ds = None

def create_manifest_file(output_dir, split_name, sensor_counts, total_labels):
    """Create a manifest file listing all patches in the split"""
    manifest_path = os.path.join(output_dir, split_name, f"{split_name}_manifest.txt")
    
    with open(manifest_path, 'w') as f:
        f.write(f"{split_name.upper()} SET MANIFEST\n")
        f.write("=" * 60 + "\n")
        
        # Write sensor information
        f.write("\nSENSOR PATCH COUNTS:\n")
        f.write("-" * 40 + "\n")
        for sensor_name, count in sensor_counts.items():
            f.write(f"{sensor_name}: {count} patches\n")
        
        f.write(f"\nCOMMON LABELS: {total_labels} patches\n")
        
        # Write directory structure
        f.write("\nDIRECTORY STRUCTURE:\n")
        f.write("-" * 40 + "\n")
        for sensor_name in sensor_counts.keys():
            f.write(f"\n{sensor_name}:\n")
            f.write(f"  Images: {split_name}/{sensor_name}/img/class_X_patch_Y.tif\n")
        
        f.write(f"\nLabels:\n")
        f.write(f"  Filtered: {split_name}/labels/filtered/class_X_patch_Y.tif\n")
        f.write(f"  Unfiltered: {split_name}/labels/unfiltered/class_X_patch_Y.tif\n")
    
    print(f"  Manifest file created: {manifest_path}")

def scan_original_classes(lc_path, batch_size=2048):
    """Scan the original land cover to find all existing classes"""
    print("Scanning original land cover for classes...")
    src_ds = gdal.Open(lc_path)
    if src_ds is None:
        raise ValueError(f"Could not open land cover file: {lc_path}")
    
    rows = src_ds.RasterYSize
    cols = src_ds.RasterXSize
    
    classes = set()
    num_batches_y = math.ceil(rows / batch_size)
    num_batches_x = math.ceil(cols / batch_size)
    
    for batch_y in range(num_batches_y):
        for batch_x in range(num_batches_x):
            y_start = batch_y * batch_size
            y_end = min((batch_y + 1) * batch_size, rows)
            x_start = batch_x * batch_size
            x_end = min((batch_x + 1) * batch_size, cols)
            
            batch_array = src_ds.GetRasterBand(1).ReadAsArray(x_start, y_start, 
                                                             x_end - x_start, y_end - y_start)
            if batch_array is not None:
                batch_classes = np.unique(batch_array)
                classes.update(batch_classes)
    
    src_ds = None
    
    # Convert to sorted list, remove 0 (background)
    classes = sorted([int(c) for c in classes if c != 0])
    print(f"Found {len(classes)} original classes (excluding 0): {classes}")
    return classes

def save_label_mapping_metadata(output_dir, label_mapping_metadata):
    """Save label mapping metadata to JSON file"""
    metadata_path = os.path.join(output_dir, 'label_mapping_metadata.json')
    
    # Convert numpy types to Python types for JSON serialization
    serializable_metadata = {
        'original_to_new': {int(k): int(v) for k, v in label_mapping_metadata['original_to_new'].items()},
        'new_to_original': {int(k): int(v) for k, v in label_mapping_metadata['new_to_original'].items()},
        'new_class_definitions': label_mapping_metadata['new_class_definitions'],
        'original_alberta_classes': [int(c) for c in label_mapping_metadata['original_alberta_classes']]
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(serializable_metadata, f, indent=2)
    
    print(f"Label mapping metadata saved to: {metadata_path}")
    return metadata_path

def verify_sensor_dimensions(sensor_paths, reference_path):
    """Verify that all sensor images have the same dimensions and geotransform"""
    print("Verifying sensor dimensions...")
    
    # Open reference (land cover) to get dimensions
    ref_ds = gdal.Open(reference_path, gdal.GA_ReadOnly)
    ref_rows = ref_ds.RasterYSize
    ref_cols = ref_ds.RasterXSize
    ref_gt = ref_ds.GetGeoTransform()
    ref_proj = ref_ds.GetProjection()
    ref_ds = None
    
    print(f"Reference dimensions: {ref_rows} x {ref_cols}")
    
    sensor_info = {}
    for sensor_name, sensor_path in sensor_paths.items():
        try:
            ds = gdal.Open(sensor_path, gdal.GA_ReadOnly)
            if ds is None:
                print(f"  ERROR: Could not open {sensor_name} at {sensor_path}")
                continue
                
            rows = ds.RasterYSize
            cols = ds.RasterXSize
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            bands = ds.RasterCount
            
            # Check dimensions
            if rows != ref_rows or cols != ref_cols:
                print(f"  ERROR: {sensor_name} dimensions mismatch: {rows}x{cols} (expected {ref_rows}x{ref_cols})")
                continue
            
            # Check geotransform (allow small floating point differences)
            gt_match = all(abs(gt[i] - ref_gt[i]) < 0.001 for i in range(6))
            if not gt_match:
                print(f"  WARNING: {sensor_name} geotransform differs slightly")
            
            sensor_info[sensor_name] = {
                'path': sensor_path,
                'bands': bands,
                'rows': rows,
                'cols': cols,
                'geotransform': gt,
                'projection': proj
            }
            
            print(f"  ✓ {sensor_name}: {rows}x{cols}, {bands} bands")
            
            ds = None
        except Exception as e:
            print(f"  ERROR processing {sensor_name}: {str(e)}")
    
    if len(sensor_info) != len(sensor_paths):
        print(f"\nWARNING: Only {len(sensor_info)} out of {len(sensor_paths)} sensors passed verification")
    
    return sensor_info

def main():
    """Main function for multi-sensor patch extraction with identical locations"""
    # Configuration
    landcover_path = r'D:\Hackathon15_AlphaEarth\GroundTruth_Landsat_Canada\landcover-2020-classification_CLIPPED.tif'
    
    # Define all sensor paths
    sensor_paths = {
        'landsat8': r'D:\Hackathon15_AlphaEarth\Alberta_L8_2020\Alberta_2020_NAD83_StatsCan_L8_30m_Mosaics_EPSG_3979_Clipped_Stack\Alberta_2020_L8_Stacked_6Bands.tif',  # Landsat-8: 6 bands
        'sentinel2': r'D:\Hackathon15_AlphaEarth\Alberta_Sentinel2_2020\Alberta_2020_NAD83_StatsCan_Sentinel2_30m_Mosaics_EPSG_3979_Clipped_Stack\Alberta_2020_S2_Stacked_10Bands.tif',      # Sentinel-2: 10 bands
        'alphaearth': r'D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979_Clipped_Stack\Alberta_2020_AlphaEarth_Stacked_64Bands.tif'  # AlphaEarth: 64 bands
    }
    
    output_dir = r'D:\Hackathon15_AlphaEarth\train_val_test_patches'
    
    # Parameters
    window_size = 224
    overlap = 20
    min_patches_per_class = 100
    train_ratio = 0.7
    val_ratio = 0.15
    min_homogeneity = 0.8
    batch_size = 2048
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MULTI-SENSOR PATCH EXTRACTION - IDENTICAL LOCATIONS")
    print("=" * 80)
    
    # Step 0: Verify all sensors have same dimensions as landcover
    print("\n0. Verifying sensor dimensions and alignment...")
    sensor_info = verify_sensor_dimensions(sensor_paths, landcover_path)
    
    if not sensor_info:
        print("ERROR: No sensors passed verification. Exiting.")
        return
    
    print(f"\n  Processing {len(sensor_info)} sensors: {list(sensor_info.keys())}")
    
    # Step 1: Scan original classes and create label mapping
    print("\n1. Scanning original land cover and creating label mapping...")
    original_classes = scan_original_classes(landcover_path, batch_size)
    
    # Create label mapping
    original_to_new, new_to_original, new_class_definitions, sorted_original_classes = \
        create_alberta_label_mapping(original_classes)
    
    # Save label mapping metadata
    label_mapping_metadata = {
        'original_to_new': original_to_new,
        'new_to_original': new_to_original,
        'new_class_definitions': new_class_definitions,
        'original_alberta_classes': sorted_original_classes
    }
    
    save_label_mapping_metadata(output_dir, label_mapping_metadata)
    
    print(f"\nLabel mapping created:")
    print(f"  Original classes: {sorted_original_classes}")
    print(f"  New labels: {list(new_to_original.keys())}")
    print(f"  Background (original 0) mapped to: -99")
    
    # Step 2: Remap original ground truth
    print("\n2. Remapping original ground truth labels...")
    lc_remapped_path = os.path.join(output_dir, 'LC_remapped.tif')
    if not os.path.exists(lc_remapped_path):
        remap_ground_truth(landcover_path, lc_remapped_path, original_to_new)
    else:
        print("  Remapped ground truth already exists")
    
    # Step 3: Create filtered version of remapped land cover
    lc_filtered_path = os.path.join(output_dir, 'LC_filtered.tif')
    if not os.path.exists(lc_filtered_path):
        print("\n3. Creating filtered land cover from remapped data...")
        apply_strict_majority_filter_batched(lc_remapped_path, lc_filtered_path,
                                           min_homogeneity, batch_size)
    else:
        print("\n3. Filtered land cover already exists")
    
    # Step 4: Create abundance maps based on filtered data
    print("\n4. Creating abundance maps...")
    abundance_dir = os.path.join(output_dir, 'abundance_maps')
    classes, stride, lc_rows, lc_cols = create_abundance_maps_batched(
        lc_filtered_path, abundance_dir, window_size, overlap, batch_size
    )
    
    # Note: 'classes' now contains the new 0-based labels
    print(f"\n  Using {len(classes)} remapped classes: {classes}")
    
    # Step 5: Select and split patches (identical for all sensors)
    print("\n5. Selecting and splitting patches (identical for all sensors)...")
    all_splits = select_and_split_patches_stratified(
        abundance_dir, classes, lc_rows, lc_cols,
        window_size, stride, min_patches_per_class,
        train_ratio, val_ratio
    )
    
    # Calculate total patches for each split
    train_total = sum(len(p) for p in all_splits['train'].values())
    val_total = sum(len(p) for p in all_splits['val'].values())
    test_total = sum(len(p) for p in all_splits['test'].values())
    
    print(f"\n  Total patches selected:")
    print(f"    Training: {train_total}")
    print(f"    Validation: {val_total}")
    print(f"    Testing: {test_total}")
    print(f"    Total: {train_total + val_total + test_total}")
    
    # Step 6: Extract patches for all sensors (identical locations)
    print("\n6. Extracting patches for all sensors (identical locations)...")
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    
    # Prepare sensor paths dictionary for extraction
    sensor_paths_for_extraction = {}
    for sensor_name, info in sensor_info.items():
        sensor_paths_for_extraction[sensor_name] = info['path']
    
    total_extracted_by_sensor = {}
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n  Extracting {split_name} patches...")
        
        # Extract patches for all sensors
        sensor_counts = extract_patches_multisensor(
            sensor_paths_for_extraction, lc_filtered_path, lc_remapped_path,
            all_splits[split_name], window_size, stride, 
            patches_dir, split_name
        )
        
        # Calculate total labels for this split
        split_total_labels = sum(len(p) for p in all_splits[split_name].values())
        
        # Create manifest file for this split
        create_manifest_file(patches_dir, split_name, sensor_counts, split_total_labels)
        
        # Accumulate totals
        for sensor_name, count in sensor_counts.items():
            if sensor_name not in total_extracted_by_sensor:
                total_extracted_by_sensor[sensor_name] = 0
            total_extracted_by_sensor[sensor_name] += count
    
    # Step 7: Save summary and class info
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    
    # Print class mapping information
    print(f"\nCLASS MAPPING SUMMARY:")
    print(f"  Number of classes: {len(classes)}")
    print(f"  New labels: {sorted(classes)}")
    print(f"\n  Detailed mapping:")
    for new_label in sorted(classes):
        original_id = new_to_original[new_label]
        class_name = new_class_definitions[new_label]['name']
        print(f"    New label {new_label} → Original {original_id} ({class_name})")
    
    # Print sensor information
    print(f"\nSENSOR INFORMATION:")
    for sensor_name, info in sensor_info.items():
        print(f"  {sensor_name}:")
        print(f"    Bands: {info['bands']}")
        print(f"    Dimensions: {info['rows']} x {info['cols']}")
        print(f"    Patches extracted: {total_extracted_by_sensor.get(sensor_name, 0)}")
    
    print(f"\nDATASET STATISTICS:")
    print(f"  Window size: {window_size}x{window_size}")
    print(f"  Stride: {stride}")
    print(f"  Image patches: UInt8 format (sensor-specific)")
    print(f"  Label patches: Int16 format (common for all sensors)")
    print(f"\n  Total patches per split:")
    print(f"    Training: {train_total}")
    print(f"    Validation: {val_total}")
    print(f"    Testing: {test_total}")
    print(f"    Total: {train_total + val_total + test_total}")
    
    # Display directory structure
    print(f"\nOUTPUT DIRECTORY STRUCTURE:")
    print(f"  {output_dir}/")
    print(f"  ├── LC_remapped.tif")
    print(f"  ├── LC_filtered.tif")
    print(f"  ├── abundance_maps/")
    print(f"  ├── patches/")
    print(f"  │   ├── train/")
    print(f"  │   │   ├── landsat8/")
    print(f"  │   │   │   └── img/           # Landsat-8 patches (6 bands)")
    print(f"  │   │   ├── sentinel2/")
    print(f"  │   │   │   └── img/           # Sentinel-2 patches (10 bands)")
    print(f"  │   │   ├── alphaearth/")
    print(f"  │   │   │   └── img/           # AlphaEarth patches (64 bands)")
    print(f"  │   │   └── labels/            # Common labels")
    print(f"  │   │       ├── filtered/")
    print(f"  │   │       └── unfiltered/")
    print(f"  │   ├── val/                   (same structure as train)")
    print(f"  │   └── test/                  (same structure as train)")
    print(f"  └── label_mapping_metadata.json")
    
    # Save dataset configuration for training
    dataset_config = {
        'num_classes': len(classes),
        'new_labels': sorted(classes),
        'class_mapping': {int(k): int(v) for k, v in new_to_original.items()},
        'class_names': {str(new_label): new_class_definitions[new_label]['name'] 
                       for new_label in classes},
        'window_size': window_size,
        'background_label': -99,
        'common_labels': {
            'train': os.path.join(patches_dir, 'train', 'labels'),
            'val': os.path.join(patches_dir, 'val', 'labels'),
            'test': os.path.join(patches_dir, 'test', 'labels')
        },
        'sensors': {}
    }
    
    # Add sensor-specific configurations
    for sensor_name, info in sensor_info.items():
        dataset_config['sensors'][sensor_name] = {
            'bands': info['bands'],
            'image_dtype': 'uint8',
            'train_img_path': os.path.join(patches_dir, 'train', sensor_name, 'img'),
            'val_img_path': os.path.join(patches_dir, 'val', sensor_name, 'img'),
            'test_img_path': os.path.join(patches_dir, 'test', sensor_name, 'img')
        }
    
    config_path = os.path.join(output_dir, 'multisensor_dataset_config.json')
    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    print(f"\nDataset configuration saved to: {config_path}")
    
    # Print patch naming convention
    print(f"\nPATCH NAMING CONVENTION:")
    print(f"  Images: class_<class_id>_patch_<patch_id>.tif")
    print(f"  Labels: class_<class_id>_patch_<patch_id>.tif (same name for all sensors)")
    print(f"\n  Example for training:")
    print(f"    Landsat-8: patches/train/landsat8/img/class_0_patch_0.tif")
    print(f"    Sentinel-2: patches/train/sentinel2/img/class_0_patch_0.tif")
    print(f"    AlphaEarth: patches/train/alphaearth/img/class_0_patch_0.tif")
    print(f"    Labels: patches/train/labels/filtered/class_0_patch_0.tif")
    
    print(f"\nAll patches are extracted from identical geographic locations!")
    print(f"You can now train models for Landsat-8, Sentinel-2, and AlphaEarth using the same ground truth.")

if __name__ == '__main__':
    main()