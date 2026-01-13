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
def save_classification_results_with_gdal(classified_scene, transform, output_dir, scene_name, timestamp, save_probabilities=False):
    """
    Save classification results using GDAL with exact EPSG:3979 CRS.
    
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
    # Based on your training data and the PDF
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
    
    # Note: The following Canada classes are NOT present in Alberta according to the PDF:
    # - Class 11: Sub-polar or polar shrubland-lichen-moss
    # - Class 13: Sub-polar or polar barren-lichen-moss
    # These are excluded from Canada's national dataset for Alberta
    
    # Create a list of ALL Canada classes that should appear in Alberta
    ALBERTA_CANADA_CLASSES = sorted(ALBERTA_TO_CANADA_MAPPING.values())
    
    # Remap the classified scene from Alberta classes (0-12) to Canada-wide classes
    print("Remapping Alberta classes (0-12) to Canada Land Cover classes...")
    remapped_scene = np.zeros_like(classified_scene, dtype=np.uint8)
    
    # First, set all pixels to 0 (background/unclassified)
    remapped_scene.fill(0)
    
    # Then map each Alberta class to its Canada-wide equivalent
    for alberta_class, canada_class in ALBERTA_TO_CANADA_MAPPING.items():
        mask = classified_scene == alberta_class
        if mask.any():
            remapped_scene[mask] = canada_class
            canada_name = CANADA_CLASS_DEFINITIONS[canada_class]['name']
            print(f"  Mapped Alberta class {alberta_class} -> Canada class {canada_class} ({canada_name})")
    
    # Count occurrences of each class
    unique_classes = np.unique(remapped_scene)
    print(f"\nFound {len(unique_classes)-1} Canada land cover classes in Alberta scene:")
    for class_id in sorted(unique_classes):
        if class_id == 0:
            continue  # Skip background class
        count = np.sum(remapped_scene == class_id)
        name = CANADA_CLASS_DEFINITIONS.get(class_id, {}).get('name', 'Unknown')
        print(f"  Class {class_id}: {name} ({count:,} pixels)")
    
    # Count background pixels
    background_count = np.sum(remapped_scene == 0)
    print(f"  Background/unclassified: {background_count:,} pixels")
    
    # EXACT WKT string for EPSG:3979
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
    
    # 1. Save the label map (single band) - This is what QGIS will show with class names
    label_path = output_dir / f"{scene_name}_labels_{timestamp}.tif"
    
    driver = gdal.GetDriverByName('GTiff')
    
    # For single-band label map: Use PHOTOMETRIC=PALETTE for color table
    ds = driver.Create(str(label_path), W, H, 1, gdal.GDT_Byte, 
                   options=['COMPRESS=LZW', 'PREDICTOR=1', 'TILED=YES',
                            'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
                            'PHOTOMETRIC=PALETTE'])
    
    ds.SetGeoTransform((transform.c, transform.a, transform.b, 
                        transform.f, transform.d, transform.e))
    
    srs = osr.SpatialReference()
    srs.ImportFromWkt(EPSG_3979_WKT)
    ds.SetProjection(srs.ExportToWkt())
    
    band = ds.GetRasterBand(1)
    band.WriteArray(remapped_scene)  # Use remapped scene instead of original
    band.SetNoDataValue(0)  # Set 0 as nodata (background/unclassified)
    band.SetDescription("2020 Canada Land Cover Classification")
    
    # Create and set colormap for Canada classes present in Alberta
    colors = gdal.ColorTable()
    
    # First, set class 0 (background/unclassified) to transparent black
    colors.SetColorEntry(0, (0, 0, 0, 0))  # Transparent black for background
    
    # Set colors for all Canada classes that appear in Alberta
    for class_id in ALBERTA_CANADA_CLASSES:
        r, g, b = CANADA_CLASS_DEFINITIONS[class_id]['color']
        colors.SetColorEntry(class_id, (r, g, b, 255))
    
    # Note: We don't set colors for classes 11 and 13 since they don't appear in Alberta
    # according to the PDF
    
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    
    # Create category names for QGIS - only for classes present in Alberta
    category_names = []
    
    # Add transparent entry for class 0 (background)
    category_names.append("0:0:0:0:0:No Data")  # Transparent black
    
    # Add entries for all Canada classes that appear in Alberta
    for class_id in ALBERTA_CANADA_CLASSES:
        r, g, b = CANADA_CLASS_DEFINITIONS[class_id]['color']
        name = CANADA_CLASS_DEFINITIONS[class_id]['name']
        # Format: value:red:green:blue:opacity:name
        category_names.append(f"{class_id}:{r}:{g}:{b}:255:{name}")
    
    # Set the categories on the band
    band.SetRasterCategoryNames(category_names)
    
    # Also set as metadata
    band.SetMetadata({
        'CLASS_0_NAME': 'No Data',
        'CLASS_0_COLOR': '0,0,0',
        'CLASS_0_OPACITY': '0'
    })
    
    for class_id in ALBERTA_CANADA_CLASSES:
        name = CANADA_CLASS_DEFINITIONS[class_id]['name']
        r, g, b = CANADA_CLASS_DEFINITIONS[class_id]['color']
        band.SetMetadata({
            f'CLASS_{class_id}_NAME': name,
            f'CLASS_{class_id}_COLOR': f'{r},{g},{b}'
        })
    
    # Set dataset metadata
    ds.SetMetadataItem('TIFFTAG_SOFTWARE', 'AlphaEarth Classification')
    ds.SetMetadataItem('TIFFTAG_IMAGEDESCRIPTION', '2020 Canada Land Cover Classification - Alberta')
    ds.SetMetadataItem('TIFFTAG_DATETIME', datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
    
    # Set additional metadata
    ds.SetMetadata({
        'CLASS_COUNT': str(len(ALBERTA_CANADA_CLASSES)),
        'CLASS_DEFINITION': '2020 Canada Land Cover',
        'SOURCE': 'AlphaEarth Alberta Classification',
        'DATA_SOURCE': 'Landsat 8, Sentinel-2, AlphaEarth',
        'PROCESSING_DATE': datetime.now().strftime('%Y-%m-%d'),
        'CLASS_MAPPING': 'Alberta 13-class -> Canada Land Cover classes',
        'NOTE': 'Classes 11 and 13 are excluded as per Canada national dataset for Alberta'
    }, '')
    
    ds.BuildOverviews("NEAREST", [2, 4, 8, 16])
    ds = None  # Close dataset
    
    print(f"✓ Label map saved: {label_path}")
    print(f"  Canada Land Cover classes with official names and colors")
    print(f"  Background (class 0) is transparent")
    print(f"  No 'Undefined' classes - only official Canada classes")
    
    # 2. Create and save RGB visualization (for visual inspection)
    rgb_path = output_dir / f"{scene_name}_rgb_{timestamp}.tif"
    
    rgb_array = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id in ALBERTA_CANADA_CLASSES:
        mask = remapped_scene == class_id
        if mask.any():
            rgb_array[mask] = CANADA_CLASS_DEFINITIONS[class_id]['color']
    
    # Background pixels (class 0) remain black
    
    # For RGB file: USE PHOTOMETRIC=RGB
    ds = driver.Create(str(rgb_path), W, H, 3, gdal.GDT_Byte,
                       options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES', 
                                'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
                                'PHOTOMETRIC=RGB'])
    
    ds.SetGeoTransform((transform.c, transform.a, transform.b,
                        transform.f, transform.d, transform.e))
    
    srs = osr.SpatialReference()
    srs.ImportFromWkt(EPSG_3979_WKT)
    ds.SetProjection(srs.ExportToWkt())
    
    for i in range(3):
        band = ds.GetRasterBand(i + 1)
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
    
    ds.SetMetadataItem('TIFFTAG_SOFTWARE', 'Classification')
    ds.SetMetadataItem('TIFFTAG_IMAGEDESCRIPTION', '2020 Canada Land Cover - RGB Visualization')
    
    ds.BuildOverviews("AVERAGE", [2, 4, 8, 16])
    ds = None
    
    print(f"✓ RGB visualization saved: {rgb_path}")
    print(f"  This is a true RGB image for visual inspection")
    
    # 3. Create a metadata text file
    metadata_path = output_dir / f"{scene_name}_metadata_{timestamp}.txt"
    with open(metadata_path, 'w') as f:
        f.write("2020 Canada Land Cover Classification - Alberta\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Classification Details:\n")
        f.write(f"  Scene: {scene_name}\n")
        f.write(f"  Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Output Files:\n")
        f.write(f"    - Label Map: {label_path.name}\n")
        f.write(f"    - RGB Visualization: {rgb_path.name}\n\n")
        
        f.write("Canada Land Cover Classes in Alberta:\n")
        f.write("-" * 40 + "\n")
        for class_id in ALBERTA_CANADA_CLASSES:
            name = CANADA_CLASS_DEFINITIONS[class_id]['name']
            r, g, b = CANADA_CLASS_DEFINITIONS[class_id]['color']
            count = np.sum(remapped_scene == class_id)
            f.write(f"  Class {class_id:2d}: {name}\n")
            f.write(f"       RGB: {r:3d}, {g:3d}, {b:3d}\n")
            f.write(f"       Pixels: {count:,}\n\n")
        
        f.write("Note:\n")
        f.write("  - Class 0: No Data (transparent background)\n")
        f.write("  - Classes 11 and 13 are excluded from Canada's national dataset for Alberta\n")
        f.write("  - Based on 2020 Canada Land Cover classification scheme\n")
    
    print(f"✓ Metadata file saved: {metadata_path}")
    
    # 4. Create a QGIS style file (.qml) for better visualization
    qml_path = label_path.with_suffix('.qml')
    create_qgis_style_file(qml_path, CANADA_CLASS_DEFINITIONS, ALBERTA_CANADA_CLASSES)
    
    return label_path, rgb_path


def create_qgis_style_file(qml_path, class_definitions, alberta_classes):
    """Create a QGIS style file (.qml) for the classified raster"""
    
    qml_content = '''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.0-Firenze" styleCategories="Symbology">
  <pipe>
    <rasterrenderer opacity="1" alphaBand="-1" classificationMax="19" classificationMin="0" type="paletted" band="1">
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
    
    # Add transparent entry for class 0
    qml_content += '        <paletteEntry value="0" color="#000000" label="No Data" alpha="0"/>\n'
    
    # Add palette entries for Alberta Canada classes
    for class_id in alberta_classes:
        color = class_definitions[class_id]['color']
        name = class_definitions[class_id]['name']
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
    print(f"  When you open the TIFF in QGIS, it will automatically load this style file")
    return qml_path

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
    target_crs='EPSG:3979'
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
    print(f"  Output directory: {output_dir}")
    
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
        'output_crs': 'EPSG:3979 - NAD83(CSRS) / Canada Atlas Lambert',
        'transform': [transform.a, transform.b, transform.c, 
                     transform.d, transform.e, transform.f],
        'dimensions': {'height': H, 'width': W, 'channels': C},
        'processing_date': str(datetime.now()),
        'processing_time_seconds': pbar.format_dict['elapsed']
    }
    
    if not save_probabilities and rgb_path:
        metadata['rgb_visualization'] = str(rgb_path)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    
    # 9. Create symbolic links for easy access
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
                target_crs='EPSG:3979'
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