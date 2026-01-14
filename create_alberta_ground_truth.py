"""
Create Alberta-remapped ground truth from Canada-wide land cover map.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import argparse
from osgeo import gdal, osr

# Mapping from Canada classes to Alberta classes (0-12)
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

# Alberta class names for reference
ALBERTA_CLASS_NAMES = [
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

def remap_ground_truth(input_path, output_path, nodata_value=-99):
    """
    Remap Canada-wide land cover classes to Alberta classes (0-12).
    
    Args:
        input_path: Path to original ground truth (Canada classes 1-19)
        output_path: Path to save remapped ground truth (Alberta classes 0-12)
        nodata_value: Value for invalid/background pixels (default: -99)
    """
    print(f"Remapping ground truth...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Load original ground truth
    with rasterio.open(input_path) as src:
        data = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        profile = src.profile
        
        print(f"Original shape: {data.shape}")
        print(f"Original CRS: {crs}")
        
        # Check original classes
        unique_orig = np.unique(data)
        print(f"\nOriginal classes in ground truth: {sorted(unique_orig)}")
        
        # Create remapped array
        remapped_data = np.full_like(data, nodata_value, dtype=np.int16)
        
        # Apply mapping
        mapping_counts = {}
        for canada_class, alberta_class in CANADA_TO_ALBERTA_MAPPING.items():
            mask = data == canada_class
            if mask.any():
                count = mask.sum()
                remapped_data[mask] = alberta_class
                mapping_counts[canada_class] = {
                    'alberta_class': alberta_class,
                    'pixels': count,
                    'name': ALBERTA_CLASS_NAMES[alberta_class] if alberta_class < len(ALBERTA_CLASS_NAMES) else 'Unknown'
                }
        
        # Handle class 0 (Unknown/NoData) - already set to nodata_value
        unknown_mask = data == 0
        if unknown_mask.any():
            remapped_data[unknown_mask] = nodata_value
            print(f"  Class 0 (Unknown) -> {nodata_value}: {unknown_mask.sum():,} pixels")
        
        # Check for unmapped Canada classes
        all_canada_classes = set(range(1, 20))
        mapped_classes = set(CANADA_TO_ALBERTA_MAPPING.keys())
        unmapped_classes = all_canada_classes - mapped_classes
        
        for canada_class in unmapped_classes:
            mask = data == canada_class
            if mask.any():
                count = mask.sum()
                remapped_data[mask] = nodata_value
                print(f"  WARNING: Unmapped Canada class {canada_class} -> {nodata_value}: {count:,} pixels")
        
        # Print mapping summary
        print(f"\nMapping Summary:")
        print("-" * 60)
        total_mapped = 0
        for canada_class, info in sorted(mapping_counts.items()):
            print(f"  Canada {canada_class:2d} -> Alberta {info['alberta_class']:2d}: "
                  f"{info['pixels']:12,d} pixels ({info['name']})")
            total_mapped += info['pixels']
        
        # Statistics
        total_pixels = data.size
        nodata_pixels = np.sum(remapped_data == nodata_value)
        valid_pixels = total_pixels - nodata_pixels
        
        print(f"\nStatistics:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Mapped pixels: {total_mapped:,} ({total_mapped/total_pixels*100:.1f}%)")
        print(f"  Invalid/NoData: {nodata_pixels:,} ({nodata_pixels/total_pixels*100:.1f}%)")
        
        # Check remapped classes
        unique_remapped = np.unique(remapped_data)
        alberta_classes = [c for c in unique_remapped if c != nodata_value]
        print(f"\nRemapped Alberta classes: {sorted(alberta_classes)}")
        
        # Update profile for output
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=nodata_value,
            compress='LZW'
        )
        
        # Save remapped ground truth
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(remapped_data, 1)
            
            # Add band description
            dst.set_band_description(1, 'Alberta Land Cover Classes (0-12)')
            
            # Add metadata
            dst.update_tags(
                title='Alberta Land Cover Ground Truth (Remapped)',
                source_file=str(input_path),
                mapping='Canada to Alberta classes',
                nodata_value=str(nodata_value),
                alberta_classes=str(sorted(alberta_classes))
            )
        
        print(f"\n✓ Remapped ground truth saved: {output_path}")
        
        # Also create a color-mapped version for visualization
        create_colormap_version(output_path, output_path.with_name(f"{output_path.stem}_colormap.tif"))
        
        return output_path

def create_colormap_version(input_path, output_path):
    """Create a version with color table for visualization"""
    try:
        from osgeo import gdal
        
        print(f"\nCreating color-mapped version...")
        
        # Define Alberta class colors
        ALBERTA_CLASS_COLORS = {
            0: (0, 61, 0),        # Temperate needleleaf forest
            1: (148, 156, 112),   # Sub-polar taiga forest
            2: (20, 140, 61),     # Temperate broadleaf forest
            3: (91, 117, 43),     # Mixed forest
            4: (179, 138, 51),    # Temperate shrubland
            5: (225, 207, 138),   # Temperate grassland
            6: (186, 212, 143),   # Polar grassland-lichen
            7: (107, 163, 138),   # Wetland
            8: (230, 174, 102),   # Cropland
            9: (168, 171, 174),   # Barren lands
            10: (220, 33, 38),    # Urban
            11: (76, 112, 163),   # Water
            12: (255, 250, 255)   # Snow/ice
        }
        
        # Open input
        ds = gdal.Open(str(input_path), gdal.GA_ReadOnly)
        if ds is None:
            print(f"  Could not open {input_path}")
            return
        
        # Create output with Byte type for color table
        driver = gdal.GetDriverByName('GTiff')
        ds_out = driver.Create(
            str(output_path),
            ds.RasterXSize,
            ds.RasterYSize,
            1,
            gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES']
        )
        
        # Copy geotransform and projection
        ds_out.SetGeoTransform(ds.GetGeoTransform())
        ds_out.SetProjection(ds.GetProjection())
        
        # Read data and convert to Byte
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        
        # Convert to Byte, remapping -99 to 255 for NoData
        byte_data = np.zeros_like(data, dtype=np.uint8)
        
        # Map Alberta classes (0-12) directly
        for alberta_class in range(13):
            mask = data == alberta_class
            if mask.any():
                byte_data[mask] = alberta_class
        
        # Map NoData (-99) to 255
        if nodata is not None:
            nodata_mask = data == nodata
            if nodata_mask.any():
                byte_data[nodata_mask] = 255
        
        # Write data
        ds_out.GetRasterBand(1).WriteArray(byte_data)
        ds_out.GetRasterBand(1).SetNoDataValue(255)
        
        # Create and set color table
        colors = gdal.ColorTable()
        
        # NoData (255) - black
        colors.SetColorEntry(255, (0, 0, 0, 255))
        
        # Alberta classes
        for alberta_class in range(13):
            if alberta_class in ALBERTA_CLASS_COLORS:
                r, g, b = ALBERTA_CLASS_COLORS[alberta_class]
                colors.SetColorEntry(alberta_class, (r, g, b, 255))
            else:
                colors.SetColorEntry(alberta_class, (128, 128, 128, 255))
        
        ds_out.GetRasterBand(1).SetRasterColorTable(colors)
        ds_out.GetRasterBand(1).SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        
        # Set category names
        category_names = []
        category_names.append("255:0:0:0:255:NoData/Outside Alberta")
        
        for alberta_class in range(13):
            if alberta_class < len(ALBERTA_CLASS_NAMES):
                name = ALBERTA_CLASS_NAMES[alberta_class]
                if alberta_class in ALBERTA_CLASS_COLORS:
                    r, g, b = ALBERTA_CLASS_COLORS[alberta_class]
                    category_names.append(f"{alberta_class}:{r}:{g}:{b}:255:{name}")
        
        ds_out.GetRasterBand(1).SetRasterCategoryNames(category_names)
        
        # Close datasets
        ds = None
        ds_out = None
        
        print(f"✓ Color-mapped version saved: {output_path}")
        
    except Exception as e:
        print(f"  Could not create color-mapped version: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Remap Canada land cover to Alberta classes')
    parser.add_argument('--input_gt', type=str, 
                        default= './GroundTruth_Landsat_Canada/landcover-2020-classification_CLIPPED.tif',
                       help='Input ground truth path (Canada classes)')
    parser.add_argument('--output_path', type=str,
                        default= './GroundTruth_Landsat_Canada/landcover-2020-classification_CLIPPED_ALBERTA_REMAPPED.tif',
                       help='Output path for remapped ground truth')
    parser.add_argument('--nodata', type=int, default=-99,
                       help='NoData value for invalid pixels')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output_path is None:
        input_path = Path(args.input_gt)
        output_path = input_path.with_name(f"{input_path.stem}_ALBERTA_REMAPPED.tif")
    else:
        output_path = Path(args.output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remap the ground truth
    remap_ground_truth(args.input_gt, output_path, args.nodata)

if __name__ == '__main__':
    main()