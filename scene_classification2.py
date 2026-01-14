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
import sys
warnings.filterwarnings('ignore')

# Import your models
from models import (
    MIMUNet, FocalUNet, SepViTUNet, SwinUNet, 
    CATUNet, TwinsUNet, BasicUNet, HRNetWrapper
)

# Import sklearn metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, jaccard_score
)

# ==================== PREDEFINED SCENE PATHS ====================
SCENE_PATHS = {
    'landsat8': "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/preprocessing/Alberta_L8_2020/Alberta_2020_NAD83_StatsCan_L8_30m_Mosaics_EPSG_3979_Clipped_Stack/Alberta_2020_L8_Stacked_6Bands.tif",
    'sentinel2': "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/preprocessing/Alberta_Sentinel2_2020/Alberta_2020_NAD83_StatsCan_Sentinel2_30m_Mosaics_EPSG_3979_Clipped_Stack/Alberta_2020_S2_Stacked_10Bands.tif",
    'alphaearth': "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/preprocessing/AlphaEarth_Dataset/Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979_Clipped_Stack/Alberta_2020_AlphaEarth_Stacked_64Bands.tif"
}

# ==================== LABEL MAP PATHS ====================
# Add your label map paths here
LABEL_MAP_PATHS = {
    'landsat8': "/path/to/landsat8_label_map.tif",  # Update with actual path
    'sentinel2': "/path/to/sentinel2_label_map.tif",  # Update with actual path
    'alphaearth': "/path/to/alphaearth_label_map.tif"  # Update with actual path
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


def save_classification_results_with_gdal(classified_scene, transform, output_dir, scene_name, timestamp, save_probabilities=False):
    """
    Save classification results using GDAL with exact EPSG:3979 CRS.
    Clips to Alberta boundary using gdal.Warp.
    
    Args:
        classified_scene: 2D numpy array of class labels (0-12 after prediction)
        transform: Geotransform from original scene
        output_dir: Directory to save outputs
        scene_name: Name of the scene
        timestamp: Timestamp string
        save_probabilities: If True, save probabilities instead of labels
    
    Returns:
        label_path, rgb_path: Paths to saved label map and RGB visualization
    """
    from osgeo import gdal, osr
    import numpy as np
    
    H, W = classified_scene.shape
    
    # CANADA LAND COVER CLASS DEFINITIONS from the PDF (Official Canada classes)
    CANADA_CLASS_DEFINITIONS = {
        1: {'name': 'Temperate or sub-polar needleleaf forest', 'color': (0, 61, 0)},
        2: {'name': 'Sub-polar taiga needleleaf forest', 'color': (148, 156, 112)},
        5: {'name': 'Temperate or sub-polar broadleaf deciduous forest', 'color': (20, 140, 61)},
        6: {'name': 'Mixed forest', 'color': (91, 117, 43)},
        8: {'name': 'Temperate or sub-polar Shrubland', 'color': (179, 138, 51)},
        10: {'name': 'Temperate or sub-polar grassland', 'color': (225, 207, 138)},
        11: {'name': 'Sub-polar or polar shrubland-lichen-moss', 'color': (156, 117, 84)},
        12: {'name': 'Sub-polar or polar grassland-lichen-moss', 'color': (186, 212, 143)},
        13: {'name': 'Sub-polar or polar barren-lichen-moss', 'color': (64, 138, 112)},
        14: {'name': 'Wetland', 'color': (107, 163, 138)},
        15: {'name': 'Cropland', 'color': (230, 174, 102)},
        16: {'name': 'Barren land', 'color': (168, 171, 174)},
        17: {'name': 'Urban and built-up', 'color': (220, 33, 38)},
        18: {'name': 'Water', 'color': (76, 112, 163)},
        19: {'name': 'Snow and ice', 'color': (255, 250, 255)}
    }
    
    # Alberta classes (13 classes) and their mapping to Canada-wide IDs
    ALBERTA_TO_CANADA_MAPPING = {
        0: 1,   # Temperate needleleaf forest -> Temperate or sub-polar needleleaf forest
        1: 2,   # Sub-polar taiga forest -> Sub-polar taiga needleleaf forest
        2: 5,   # Temperate broadleaf forest -> Temperate or sub-polar broadleaf deciduous forest
        3: 6,   # Mixed forest -> Mixed forest
        4: 8,   # Temperate shrubland -> Temperate or sub-polar Shrubland
        5: 10,  # Temperate grassland -> Temperate or sub-polar grassland
        6: 12,  # Polar grassland-lichen -> Sub-polar or polar grassland-lichen-moss
        7: 14,  # Wetland -> Wetland
        8: 15,  # Cropland -> Cropland
        9: 16,  # Barren lands -> Barren land
        10: 17, # Urban -> Urban and built-up
        11: 18, # Water -> Water
        12: 19  # Snow/ice -> Snow and ice
    }
    
    # Create a list of ALL Canada classes that should appear in Alberta
    ALBERTA_CANADA_CLASSES = sorted(ALBERTA_TO_CANADA_MAPPING.values())
    
    # Remap the classified scene from Alberta classes (0-12) to Canada-wide classes
    print("Remapping Alberta classes (0-12) to Canada Land Cover classes...")
    remapped_scene = np.zeros_like(classified_scene, dtype=np.int16)  # Use int16 to support -99
    
    # First, set all pixels to -99 (outside boundary)
    remapped_scene.fill(-99)
    
    # Then map each Alberta class to its Canada-wide equivalent
    for alberta_class, canada_class in ALBERTA_TO_CANADA_MAPPING.items():
        mask = classified_scene == alberta_class
        if mask.any():
            remapped_scene[mask] = canada_class
            canada_name = CANADA_CLASS_DEFINITIONS[canada_class]['name']
            print(f"  Mapped Alberta class {alberta_class} -> Canada class {canada_class} ({canada_name})")
    
    # ==================== SAVE TEMPORARY UNCLIPPED FILE ====================
    print("\nCreating temporary unclipped raster...")
    
    # Create temporary directory
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file = temp_dir / f"temp_unclipped_{timestamp}.tif"
    
    driver = gdal.GetDriverByName('GTiff')
    
    # Save unclipped raster with int16 to support -99
    ds_temp = driver.Create(str(temp_file), W, H, 1, gdal.GDT_Int16, 
                   options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES',
                            'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    
    # Set geotransform and projection
    ds_temp.SetGeoTransform((transform.c, transform.a, transform.b, 
                            transform.f, transform.d, transform.e))
    
    # Set CRS
    EPSG_3979_WKT = '''PROJCRS["NAD83(CSRS) / Canada Atlas Lambert",
    BASEGEOGCRS["NAD83(CSRS)",
        DATUM["NAD83 Canadian Spatial Reference System",
            ELLIPSOID["GRS 1980",6378137,298.257222101,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433]],
        ID["EPSG",4617]],
    CONVERSION["Canada Atlas Lambert",
        METHOD["Lambert Conic Conformal (2SP)",
            ID["EPSG",9802]],
        PARAMETER["Latitude of false origin",49,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8821]],
        PARAMETER["Longitude of false origin",-95,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8822]],
        PARAMETER["Latitude of 1st standard parallel",49,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8823]],
        PARAMETER["Latitude of 2nd standard parallel",77,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8824]],
        PARAMETER["Easting at false origin",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8826]],
        PARAMETER["Northing at false origin",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8827]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Transformation of coordinates at 5m level of accuracy."],
        AREA["Canada - onshore and offshore - Alberta; British Columbia; Manitoba; New Brunswick; Newfoundland and Labrador; Northwest Territories; Nova Scotia; Nunavut; Ontario; Prince Edward Island; Quebec; Saskatchewan; Yukon."],
        BBOX[38.21,-141.01,86.46,-40.73]],
    ID["EPSG",3979]]'''
    
    srs = osr.SpatialReference()
    srs.ImportFromWkt(EPSG_3979_WKT)
    ds_temp.SetProjection(srs.ExportToWkt())
    
    band_temp = ds_temp.GetRasterBand(1)
    band_temp.WriteArray(remapped_scene)
    band_temp.SetNoDataValue(-99)  # Set -99 as NoData for outside boundary
    band_temp.FlushCache()
    ds_temp = None  # Close file
    
    print(f"  Temporary file saved: {temp_file}")
    
    # ==================== CLIP WITH ALBERTA BOUNDARY ====================
    print("\nClipping with Alberta boundary using gdal.Warp...")
    
    alberta_shapefile = "/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/preprocessing/Alberta_EPSG_3979.gpkg"
    
    # Output file path for clipped raster
    label_path = output_dir / f"{scene_name}_labels_{timestamp}_CLIPPED.tif"
    
    if not Path(alberta_shapefile).exists():
        print(f"  ERROR: Alberta shapefile not found at: {alberta_shapefile}")
        print(f"  Skipping clipping...")
        # Just rename temp file to output
        temp_file.rename(label_path)
        is_clipped = False
    else:
        print(f"  Using Alberta boundary: {alberta_shapefile}")
        
        try:
            # Define Warp options for clipping - use -99 as NoData
            warp_options = gdal.WarpOptions(
                format="GTiff",
                cutlineDSName=alberta_shapefile,
                cropToCutline=True,
                dstNodata=-99,  # Set -99 as NoData for outside boundary
                resampleAlg='near',
                creationOptions=[
                    "COMPRESS=LZW",
                    "PREDICTOR=2",
                    "TILED=YES",
                    "BLOCKXSIZE=256",
                    "BLOCKYSIZE=256",
                    "BIGTIFF=YES"
                ],
                # Preserve original resolution
                xRes=transform.a,
                yRes=transform.e,
                targetAlignedPixels=False
            )
            
            # Perform the clipping
            print("  Running gdal.Warp to clip raster...")
            ds_clipped = gdal.Warp(str(label_path), str(temp_file), options=warp_options)
            
            if ds_clipped is None:
                raise Exception("gdal.Warp returned None")
            
            # Get clipped raster info
            width = ds_clipped.RasterXSize
            height = ds_clipped.RasterYSize
            gt = ds_clipped.GetGeoTransform()
            band = ds_clipped.GetRasterBand(1)
            no_data = band.GetNoDataValue()
            
            # Read the clipped data
            clipped_data = band.ReadAsArray()
            
            ds_clipped = None  # Close file
            
            print(f"  ✓ Successfully clipped raster")
            print(f"    Output size: {width} x {height} pixels")
            print(f"    Resolution: {abs(gt[1]):.1f}m x {abs(gt[5]):.1f}m")
            print(f"    NoData value: {no_data}")
            
            # Count statistics after clipping
            unique_classes = np.unique(clipped_data)
            
            total_pixels = width * height
            nodata_pixels = np.sum(clipped_data == -99)
            classified_pixels = total_pixels - nodata_pixels
            
            print(f"\n  Statistics after clipping:")
            print(f"    Total pixels: {total_pixels:,}")
            print(f"    NoData (outside Alberta): {nodata_pixels:,} ({nodata_pixels/total_pixels*100:.1f}%)")
            print(f"    Classified (inside Alberta): {classified_pixels:,} ({classified_pixels/total_pixels*100:.1f}%)")
            
            for class_id in sorted(unique_classes):
                if class_id == -99:
                    continue
                count = np.sum(clipped_data == class_id)
                if count > 0:
                    name = CANADA_CLASS_DEFINITIONS.get(class_id, {}).get('name', f'Unknown class {class_id}')
                    print(f"    Class {class_id}: {name} - {count:,} pixels ({count/total_pixels*100:.1f}%)")
            
            is_clipped = True
            
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()
                print(f"  Cleaned up temporary file")
            
        except Exception as e:
            print(f"  ERROR during clipping: {str(e)}")
            print(f"  Using unclipped file instead")
            # Use unclipped file as final
            temp_file.rename(label_path)
            is_clipped = False
    
    # Clean up temp directory
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()
    
    # ==================== ADD COLOR TABLE AND METADATA ====================
    print(f"\nAdding color table and metadata...")
    
    # Re-open the file to add color table
    ds_final = gdal.Open(str(label_path), gdal.GA_Update)
    
    if ds_final is None:
        print(f"  ERROR: Could not open final file")
        return None, None
    
    # Change data type to Byte for color table (QGIS works better with Byte)
    print("  Converting to Byte data type for color table...")
    
    # Create a new Byte version
    label_path_byte = output_dir / f"{scene_name}_labels_{timestamp}_CLIPPED_BYTE.tif"
    
    translate_options = gdal.TranslateOptions(
        outputType=gdal.GDT_Byte,
        noData=255,  # Use 255 for -99 in Byte version
        creationOptions=[
            "COMPRESS=LZW",
            "PREDICTOR=1",
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "PHOTOMETRIC=PALETTE"
        ]
    )
    
    ds_byte = gdal.Translate(str(label_path_byte), ds_final, options=translate_options)
    ds_final = None  # Close original
    
    # Now open the Byte version
    ds_byte = gdal.Open(str(label_path_byte), gdal.GA_Update)
    band = ds_byte.GetRasterBand(1)
    
    # Create and set colormap
    colors = gdal.ColorTable()
    
    # Class 255: No Data / Outside Alberta (black)
    colors.SetColorEntry(255, (0, 0, 0, 255))
    
    # Set colors for all Canada classes (1-19)
    # We need to remap values: -99 -> 255, 1-19 stay the same
    for class_id in range(1, 20):
        if class_id in CANADA_CLASS_DEFINITIONS:
            r, g, b = CANADA_CLASS_DEFINITIONS[class_id]['color']
            colors.SetColorEntry(class_id, (r, g, b, 255))
        else:
            # For undefined classes, use gray
            colors.SetColorEntry(class_id, (128, 128, 128, 255))
    
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    
    # Set category names - ONLY class names, no extra info
    category_names = []
    
    # Class 255: No Data / Outside Alberta
    category_names.append("255:0:0:0:255:Outside Alberta")
    
    # Add entries for all Canada classes (1-19)
    for class_id in range(1, 20):
        if class_id in CANADA_CLASS_DEFINITIONS:
            r, g, b = CANADA_CLASS_DEFINITIONS[class_id]['color']
            name = CANADA_CLASS_DEFINITIONS[class_id]['name']
            # Format: value:red:green:blue:opacity:name (name only, no extra text)
            category_names.append(f"{class_id}:{r}:{g}:{b}:255:{name}")
        else:
            # Skip undefined classes
            pass
    
    band.SetRasterCategoryNames(category_names)
    
    # Set dataset metadata
    ds_byte.SetMetadataItem('TIFFTAG_SOFTWARE', 'AlphaEarth Classification')
    ds_byte.SetMetadataItem('TIFFTAG_IMAGEDESCRIPTION', '2020 Canada Land Cover Classification - Alberta')
    ds_byte.SetMetadataItem('TIFFTAG_DATETIME', datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
    
    # Build overviews
    ds_byte.BuildOverviews("NEAREST", [2, 4, 8, 16])
    ds_byte = None  # Close file
    
    # Replace the original with Byte version
    if label_path.exists():
        label_path.unlink()
    label_path_byte.rename(label_path)
    
    print(f"✓ Final label map saved: {label_path}")
    print(f"  Added color table with class names only")
    print(f"  Outside Alberta: -99 -> 255 (black)")
    print(f"  {'CLIPPED to Alberta boundary' if is_clipped else 'UNCLIPPED (boundary file not found)'}")
    
    # ==================== CREATE RGB VISUALIZATION ====================
    print(f"\nCreating RGB visualization...")
    
    rgb_path = output_dir / f"{scene_name}_rgb_{timestamp}.tif"
    
    # Open the label file to read data for RGB
    ds_label = gdal.Open(str(label_path))
    rgb_data = ds_label.GetRasterBand(1).ReadAsArray()
    H_rgb, W_rgb = rgb_data.shape
    
    rgb_array = np.zeros((H_rgb, W_rgb, 3), dtype=np.uint8)
    
    # Set colors for Canada classes (1-19)
    for class_id in range(1, 20):
        if class_id in CANADA_CLASS_DEFINITIONS:
            mask = rgb_data == class_id
            if mask.any():
                rgb_array[mask] = CANADA_CLASS_DEFINITIONS[class_id]['color']
    
    # Outside Alberta (255): black
    outside_mask = rgb_data == 255
    if outside_mask.any():
        rgb_array[outside_mask] = (0, 0, 0)
    
    ds_label = None
    
    # Save RGB file
    ds_rgb = driver.Create(str(rgb_path), W_rgb, H_rgb, 3, gdal.GDT_Byte,
                       options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES', 
                                'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
                                'PHOTOMETRIC=RGB'])
    
    # Get geotransform from label file
    ds_label = gdal.Open(str(label_path))
    gt = ds_label.GetGeoTransform()
    proj = ds_label.GetProjection()
    ds_label = None
    
    ds_rgb.SetGeoTransform(gt)
    ds_rgb.SetProjection(proj)
    
    for i in range(3):
        band = ds_rgb.GetRasterBand(i + 1)
        band.WriteArray(rgb_array[:, :, i])
        if i == 0:
            band.SetDescription("Red")
            band.SetColorInterpretation(gdal.GCI_RedBand)
        elif i == 1:
            band.SetDescription("Green")
            band.SetColorInterpretation(gdal.GCI_GreenBand)
        elif i == 2:
            band.SetDescription("Blue")
            band.SetColorInterpretation(gdal.GCI_BlueBand)
    
    ds_rgb.SetMetadataItem('TIFFTAG_SOFTWARE', 'Classification')
    ds_rgb.SetMetadataItem('TIFFTAG_IMAGEDESCRIPTION', '2020 Canada Land Cover - RGB Visualization')
    
    ds_rgb.BuildOverviews("AVERAGE", [2, 4, 8, 16])
    ds_rgb = None
    
    print(f"✓ RGB visualization saved: {rgb_path}")
    
    # ==================== CREATE QGIS STYLE FILE ====================
    print(f"\nCreating QGIS style file...")
    
    qml_path = label_path.with_suffix('.qml')
    
    qml_content = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0-Firenze" styleCategories="Symbology">
  <pipe>
    <rasterrenderer opacity="1" alphaBand="-1" classificationMax="19" classificationMin="1" type="paletted" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
'''
    
    # Class 255: Outside Alberta (black)
    qml_content += '        <paletteEntry value="255" color="#000000" label="Outside Alberta" alpha="255"/>\n'
    
    # Add palette entries for Alberta Canada classes
    for class_id in ALBERTA_CANADA_CLASSES:
        color = CANADA_CLASS_DEFINITIONS[class_id]['color']
        name = CANADA_CLASS_DEFINITIONS[class_id]['name']
        r, g, b = color
        qml_content += f'        <paletteEntry value="{class_id}" color="#{r:02x}{g:02x}{b:02x}" label="{name}" alpha="255"/>\n'
    
    qml_content += '''      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeGreen="128" colorizeOn="0" colorizeRed="255" colorizeBlue="128" grayscaleMode="0" saturation="0" colorizeStrength="100"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>'''
    
    with open(qml_path, 'w') as f:
        f.write(qml_content)
    
    print(f"✓ QGIS style file created: {qml_path}")
    print(f"  Legend will show only class names")
    
    return label_path, rgb_path


def calculate_classification_metrics(predicted_labels, ground_truth_labels, num_classes=13):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        predicted_labels: 2D numpy array of predicted class labels (0-12)
        ground_truth_labels: 2D numpy array of ground truth class labels (0-12, -99 for invalid)
        num_classes: Number of classes (default: 13)
    
    Returns:
        Dictionary containing all metrics
    """
    print("\n" + "="*60)
    print("CALCULATING CLASSIFICATION METRICS")
    print("="*60)
    
    # Flatten arrays and filter out invalid pixels (-99)
    pred_flat = predicted_labels.flatten()
    gt_flat = ground_truth_labels.flatten()
    
    # Create mask for valid pixels (not -99 in ground truth)
    valid_mask = gt_flat != -99
    
    # Apply mask
    pred_valid = pred_flat[valid_mask]
    gt_valid = gt_flat[valid_mask]
    
    # Check if there are valid pixels
    if len(pred_valid) == 0:
        print("WARNING: No valid pixels found for metrics calculation!")
        return None
    
    total_valid_pixels = len(pred_valid)
    print(f"Valid pixels for evaluation: {total_valid_pixels:,}")
    
    # Calculate overall metrics
    accuracy = accuracy_score(gt_valid, pred_valid)
    f1_macro = f1_score(gt_valid, pred_valid, average='macro', zero_division=0)
    f1_weighted = f1_score(gt_valid, pred_valid, average='weighted', zero_division=0)
    precision_macro = precision_score(gt_valid, pred_valid, average='macro', zero_division=0)
    recall_macro = recall_score(gt_valid, pred_valid, average='macro', zero_division=0)
    kappa = cohen_kappa_score(gt_valid, pred_valid)
    
    # Calculate IoU (Jaccard score) per class
    iou_per_class = jaccard_score(gt_valid, pred_valid, average=None, labels=range(num_classes))
    mean_iou = np.nanmean(iou_per_class)
    
    # Calculate confusion matrix
    cm = confusion_matrix(gt_valid, pred_valid, labels=range(num_classes))
    
    # Calculate per-class metrics
    class_names = [
        'Temperate needleleaf forest',
        'Sub-polar taiga forest',
        'Temperate broadleaf forest',
        'Mixed forest',
        'Temperate shrubland',
        'Temperate grassland',
        'Polar grassland-lichen',
        'Wetland',
        'Cropland',
        'Barren lands',
        'Urban',
        'Water',
        'Snow/ice'
    ]
    
    per_class_metrics = {}
    for i in range(num_classes):
        # Get TP, FP, FN from confusion matrix
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = iou_per_class[i] if i < len(iou_per_class) else 0
        
        per_class_metrics[i] = {
            'name': class_names[i],
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'support': int(cm[i, :].sum())
        }
    
    # Compile all metrics
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'kappa': float(kappa),
            'mean_iou': float(mean_iou),
            'total_pixels': int(total_valid_pixels)
        },
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    # Print summary
    print(f"\nOVERALL METRICS:")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  F1 Score (macro): {f1_macro:.4f}")
    print(f"  F1 Score (weighted): {f1_weighted:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro):    {recall_macro:.4f}")
    print(f"  Cohen's Kappa:    {kappa:.4f}")
    print(f"  Mean IoU:         {mean_iou:.4f}")
    
    print(f"\nPER-CLASS METRICS:")
    print(f"{'Class':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Support':<10}")
    print("-" * 80)
    
    for i in range(num_classes):
        class_metric = per_class_metrics[i]
        print(f"{class_metric['name'][:30]:<30} "
              f"{class_metric['precision']:.4f}    "
              f"{class_metric['recall']:.4f}    "
              f"{class_metric['f1']:.4f}    "
              f"{class_metric['iou']:.4f}    "
              f"{class_metric['support']:,}")
    
    # Print confusion matrix (abbreviated if too large)
    print(f"\nCONFUSION MATRIX (rows=truth, columns=predicted):")
    if num_classes <= 10:
        # Print full matrix for smaller number of classes
        print("     " + " ".join([f"{i:3d}" for i in range(num_classes)]))
        for i in range(num_classes):
            row_str = f"{i:3d}: " + " ".join([f"{cm[i,j]:3d}" for j in range(num_classes)])
            print(row_str)
    else:
        # Print summary for larger matrices
        print(f"  Shape: {cm.shape}")
        print(f"  Most confused pairs:")
        # Find most confused pairs (excluding diagonal)
        max_confusions = 5
        indices = np.argsort(cm.flatten())[::-1]
        count = 0
        for idx in indices:
            i = idx // num_classes
            j = idx % num_classes
            if i != j and cm[i, j] > 0 and count < max_confusions:
                print(f"    {class_names[i]} → {class_names[j]}: {cm[i, j]:,} pixels")
                count += 1
    
    return metrics


def evaluate_classified_scene(predicted_scene_path, ground_truth_path, output_dir, experiment_name):
    """
    Evaluate classification results against ground truth label map.
    
    Args:
        predicted_scene_path: Path to predicted classification result
        ground_truth_path: Path to ground truth label map
        output_dir: Directory to save evaluation results
        experiment_name: Name of the experiment for reporting
    """
    print(f"\nEvaluating classification results...")
    print(f"Predicted scene: {predicted_scene_path}")
    print(f"Ground truth: {ground_truth_path}")
    
    # Load predicted scene (after clipping and conversion to Canada classes)
    with rasterio.open(predicted_scene_path) as src_pred:
        predicted_data = src_pred.read(1)  # Read first band
        pred_transform = src_pred.transform
        pred_crs = src_pred.crs
    
    # Load ground truth
    with rasterio.open(ground_truth_path) as src_gt:
        ground_truth_data = src_gt.read(1)  # Read first band
        gt_transform = src_gt.transform
        gt_crs = src_gt.crs
    
    print(f"Predicted shape: {predicted_data.shape}")
    print(f"Ground truth shape: {ground_truth_data.shape}")
    
    # Check if CRS and transforms match
    if str(pred_crs) != str(gt_crs):
        print(f"WARNING: CRS mismatch! Predicted: {pred_crs}, Ground truth: {gt_crs}")
        print("Attempting to reproject ground truth to match predicted...")
        
        # You would need to implement reprojection here if needed
        # For now, we'll assume they match or are close enough
    
    # Resample ground truth to match predicted if needed
    if predicted_data.shape != ground_truth_data.shape:
        print(f"WARNING: Shape mismatch! Resampling ground truth...")
        from scipy.ndimage import zoom
        scale_h = predicted_data.shape[0] / ground_truth_data.shape[0]
        scale_w = predicted_data.shape[1] / ground_truth_data.shape[1]
        ground_truth_data = zoom(ground_truth_data, (scale_h, scale_w), order=0)  # Nearest neighbor
    
    # Convert predicted Canada classes back to Alberta classes for comparison
    # Reverse mapping: Canada class -> Alberta class
    CANADA_TO_ALBERTA_MAPPING = {
        1: 0,   # Temperate or sub-polar needleleaf forest -> Temperate needleleaf forest
        2: 1,   # Sub-polar taiga needleleaf forest -> Sub-polar taiga forest
        5: 2,   # Temperate or sub-polar broadleaf deciduous forest -> Temperate broadleaf forest
        6: 3,   # Mixed forest -> Mixed forest
        8: 4,   # Temperate or sub-polar Shrubland -> Temperate shrubland
        10: 5,  # Temperate or sub-polar grassland -> Temperate grassland
        12: 6,  # Sub-polar or polar grassland-lichen-moss -> Polar grassland-lichen
        14: 7,  # Wetland -> Wetland
        15: 8,  # Cropland -> Cropland
        16: 9,  # Barren land -> Barren lands
        17: 10, # Urban and built-up -> Urban
        18: 11, # Water -> Water
        19: 12  # Snow and ice -> Snow/ice
    }
    
    # Convert predicted data (Canada classes) back to Alberta classes (0-12)
    predicted_alberta = np.full_like(predicted_data, -99, dtype=np.int16)
    
    for canada_class, alberta_class in CANADA_TO_ALBERTA_MAPPING.items():
        mask = predicted_data == canada_class
        if mask.any():
            predicted_alberta[mask] = alberta_class
    
    # Also handle the case where predicted is already 255 (NoData/Outside Alberta)
    outside_mask = predicted_data == 255
    if outside_mask.any():
        predicted_alberta[outside_mask] = -99
    
    # Now both arrays should have classes 0-12 and -99 for invalid
    # Calculate metrics
    metrics = calculate_classification_metrics(predicted_alberta, ground_truth_data, num_classes=13)
    
    if metrics is None:
        print("No valid metrics could be calculated.")
        return None
    
    # Save metrics to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"classification_metrics_{experiment_name}_{timestamp}.json"
    metrics_path = output_dir / metrics_filename
    
    # Add metadata to metrics
    metrics_metadata = {
        'evaluation_date': str(datetime.now()),
        'predicted_scene': str(predicted_scene_path),
        'ground_truth_scene': str(ground_truth_path),
        'experiment_name': experiment_name,
        'predicted_shape': predicted_data.shape,
        'ground_truth_shape': ground_truth_data.shape,
        'predicted_crs': str(pred_crs),
        'ground_truth_crs': str(gt_crs)
    }
    
    metrics['metadata'] = metrics_metadata
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Also save a text summary
    summary_path = output_dir / f"classification_summary_{experiment_name}_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Classification Metrics Summary\n")
        f.write(f"=============================\n")
        f.write(f"Evaluation Date: {datetime.now()}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Predicted Scene: {predicted_scene_path}\n")
        f.write(f"Ground Truth: {ground_truth_path}\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:        {metrics['overall']['accuracy']:.4f}\n")
        f.write(f"  F1 Score (macro): {metrics['overall']['f1_macro']:.4f}\n")
        f.write(f"  Precision (macro): {metrics['overall']['precision_macro']:.4f}\n")
        f.write(f"  Recall (macro):    {metrics['overall']['recall_macro']:.4f}\n")
        f.write(f"  Cohen's Kappa:    {metrics['overall']['kappa']:.4f}\n")
        f.write(f"  Mean IoU:         {metrics['overall']['mean_iou']:.4f}\n")
        f.write(f"  Total Valid Pixels: {metrics['overall']['total_pixels']:,}\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        f.write(f"{'Class':<30} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'IoU':<10} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        
        for i in range(13):
            class_metric = metrics['per_class'][i]
            f.write(f"{class_metric['name'][:30]:<30} "
                   f"{class_metric['precision']:.4f}    "
                   f"{class_metric['recall']:.4f}    "
                   f"{class_metric['f1']:.4f}    "
                   f"{class_metric['iou']:.4f}    "
                   f"{class_metric['support']:,}\n")
    
    print(f"✓ Text summary saved to: {summary_path}")
    
    return metrics_path


def classify_full_scene(
    scene_path,
    model,
    model_config,
    experiment_dir,
    patch_size=224,
    batch_size=8,
    device=None,
    overlap=0,
    save_probabilities=False,
    target_crs='EPSG:3979',
    evaluate=False,
    ground_truth_path=None
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
        evaluate (bool): Whether to evaluate against ground truth
        ground_truth_path (str): Path to ground truth label map for evaluation
    
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
        
        # CRS CHECKING
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
    num_classes = model_config['num_classes']  # <-- Define num_classes here, outside the if block
    
    if save_probabilities:
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
    print(f"  Output directory: {output_dir}")
    if evaluate:
        print(f"  Evaluation: ENABLED (will compare with ground truth)")
    
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
    
    # 7. Save classification results using GDAL with exact EPSG:3979 CRS
    print(f"\nSaving classification results...")
    
    if save_probabilities:
        # For probabilities, use rasterio (not implemented with GDAL in this version)
        output_filename = f"{scene_name}_probabilities_{timestamp}.tif"
        output_path = output_dir / output_filename
        
        output_profile = profile.copy()
        output_profile.update(
            dtype=rasterio.float32,
            count=num_classes,
            compress='LZW',
            nodata=np.nan
        )
        
        # Transpose back to (C, H, W) for writing
        classified_scene = classified_scene.transpose(2, 0, 1)
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(classified_scene)
            for i in range(num_classes):
                dst.set_band_description(i+1, f"Class_{i}_probability")
        
        print(f"✓ Probability map saved: {output_path}")
        rgb_path = None
    else:
        # For labels, use GDAL to save both label map and RGB visualization
        label_path, rgb_path = save_classification_results_with_gdal(
            classified_scene=classified_scene,
            transform=transform,
            output_dir=output_dir,
            scene_name=scene_name,
            timestamp=timestamp,
            save_probabilities=save_probabilities
        )
        
        # Use label path as the main output path
        output_path = label_path
    
    print("✓ Classification complete!")
    
    # 8. Evaluate classification if requested
    if evaluate and not save_probabilities:
        if ground_truth_path is None:
            # Try to get ground truth path from sensor name
            sensor_name = model_config['sensor_name']
            if sensor_name in LABEL_MAP_PATHS:
                ground_truth_path = LABEL_MAP_PATHS[sensor_name]
                print(f"\nUsing default ground truth for {sensor_name}: {ground_truth_path}")
            else:
                print(f"\nWARNING: No ground truth path provided for sensor {sensor_name}")
                print("Skipping evaluation. Use --ground_truth_path to specify.")
                evaluate = False
        
        if evaluate and Path(ground_truth_path).exists():
            try:
                experiment_name = Path(experiment_dir).name
                metrics_path = evaluate_classified_scene(
                    predicted_scene_path=output_path,
                    ground_truth_path=ground_truth_path,
                    output_dir=output_dir,
                    experiment_name=experiment_name
                )
                if metrics_path:
                    print(f"\n✅ Evaluation completed! Metrics saved to: {metrics_path}")
            except Exception as e:
                print(f"\nERROR during evaluation: {str(e)}")
                import traceback
                traceback.print_exc()
        elif evaluate:
            print(f"\nERROR: Ground truth file not found at: {ground_truth_path}")
            print("Skipping evaluation.")
    
    # 9. Save classification metadata with CRS details
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
        'output_crs': 'EPSG:3979 - NAD83(CSRS) / Canada Atlas Lambert',
        'transform': [transform.a, transform.b, transform.c, 
                     transform.d, transform.e, transform.f],
        'dimensions': {'height': H, 'width': W, 'channels': C},
        'evaluation_performed': evaluate,
        'evaluation_ground_truth': ground_truth_path if evaluate else None,
        'processing_date': str(datetime.now()),
        'processing_time_seconds': pbar.format_dict['elapsed']
    }
    
    if not save_probabilities and rgb_path:
        metadata['rgb_visualization'] = str(rgb_path)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    
    # 10. Create symbolic links for easy access
    if not save_probabilities:
        scene_stem = Path(scene_path).stem
        
        # Link for label map
        link_path = experiment_dir / f"classified_{scene_stem}.tif"
        if not link_path.exists():
            try:
                os.symlink(output_path.relative_to(experiment_dir), link_path)
                print(f"✓ Label map symbolic link: {link_path}")
            except:
                pass
        
        # Link for RGB visualization
        if rgb_path:
            rgb_link_path = experiment_dir / f"classified_{scene_stem}_rgb.tif"
            if not rgb_link_path.exists():
                try:
                    os.symlink(rgb_path.relative_to(experiment_dir), rgb_link_path)
                    print(f"✓ RGB symbolic link: {rgb_link_path}")
                except:
                    pass
    
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
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--save_probabilities', action='store_true', default=False,
                       help='Save class probabilities instead of labels')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: "cuda", "cpu", or auto-detect')
    
    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='Evaluate classification against ground truth')
    parser.add_argument('--ground_truth_path', type=str, default=None,
                       help='Path to ground truth label map for evaluation')
    
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
                target_crs='EPSG:3979',
                evaluate=args.evaluate,
                ground_truth_path=args.ground_truth_path
            )
            
            print(f"\n✅ Scene classification completed for {experiment_dir.name}!")
            print(f"   Label map saved to: {output_path}")
            
        except Exception as e:
            print(f"ERROR processing {experiment_dir.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("All processing completed!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()