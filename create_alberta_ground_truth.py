"""
Create Alberta-remapped ground truth from Canada-wide land cover map.
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from pathlib import Path
import argparse
from osgeo import gdal, osr

# EPSG:3979 WKT definition
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
        print(f"Transform: {transform}")
        
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
    
    # ==================== SAVE WITH GDAL AND EPSG:3979 CRS ====================
    print(f"\nSaving remapped ground truth with EPSG:3979 CRS...")
    
    # Use GDAL to save with exact EPSG:3979 CRS
    driver = gdal.GetDriverByName('GTiff')
    
    # Create dataset
    ds = driver.Create(
        str(output_path),
        data.shape[1],  # width
        data.shape[0],  # height
        1,  # number of bands
        gdal.GDT_Int16,  # data type
        options=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES',
                 'BLOCKXSIZE=256', 'BLOCKYSIZE=256']
    )
    
    if ds is None:
        raise ValueError(f"Could not create output file: {output_path}")
    
    # Set geotransform
    ds.SetGeoTransform((
        transform.c,  # top left x
        transform.a,  # west-east pixel spacing
        transform.b,  # rotation (0 if north-up)
        transform.f,  # top left y
        transform.d,  # rotation (0 if north-up)
        transform.e   # north-south pixel spacing (negative for north-up)
    ))
    
    # Set CRS to EPSG:3979
    srs = osr.SpatialReference()
    srs.ImportFromWkt(EPSG_3979_WKT)
    ds.SetProjection(srs.ExportToWkt())
    
    # Write data
    band = ds.GetRasterBand(1)
    band.WriteArray(remapped_data)
    band.SetNoDataValue(nodata_value)
    
    # Set band description
    band.SetDescription('Alberta Land Cover Classes (0-12)')
    
    # Set metadata
    ds.SetMetadataItem('TIFFTAG_SOFTWARE', 'Alberta Ground Truth Remapping')
    ds.SetMetadataItem('TIFFTAG_IMAGEDESCRIPTION', '2020 Alberta Land Cover Ground Truth (Remapped from Canada classes)')
    ds.SetMetadataItem('TIFFTAG_DATETIME', datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
    
    # Calculate statistics
    band.ComputeStatistics(False)
    band.FlushCache()
    
    # Build overviews
    ds.BuildOverviews("NEAREST", [2, 4, 8, 16])
    
    # Close dataset
    ds = None
    
    print(f"\n✓ Remapped ground truth saved: {output_path}")
    print(f"  CRS: EPSG:3979 - NAD83(CSRS) / Canada Atlas Lambert")
    print(f"  NoData value: {nodata_value}")
    print(f"  Data type: Int16")
    
    # Verify the saved file
    verify_saved_file(output_path, nodata_value)
    
    return output_path

def verify_saved_file(file_path, expected_nodata=-99):
    """Verify the saved remapped ground truth file."""
    try:
        print(f"\nVerifying saved file...")
        
        with rasterio.open(file_path) as src:
            data = src.read(1)
            crs = src.crs
            transform = src.transform
            nodata = src.nodata
            
            print(f"  Shape: {data.shape}")
            print(f"  CRS: {crs}")
            print(f"  NoData: {nodata}")
            print(f"  Transform: {transform}")
            
            # Check classes
            unique_classes = np.unique(data)
            alberta_classes = [c for c in unique_classes if c != nodata]
            
            print(f"  Alberta classes present: {sorted(alberta_classes)}")
            
            # Count pixels per class
            for class_id in sorted(alberta_classes):
                count = np.sum(data == class_id)
                if class_id < len(ALBERTA_CLASS_NAMES):
                    name = ALBERTA_CLASS_NAMES[class_id]
                    print(f"    Class {class_id}: {name} - {count:,} pixels")
                else:
                    print(f"    Class {class_id}: Unknown - {count:,} pixels")
            
            nodata_pixels = np.sum(data == nodata)
            total_pixels = data.size
            print(f"  NoData pixels: {nodata_pixels:,} ({nodata_pixels/total_pixels*100:.1f}%)")
            
            # Verify CRS is EPSG:3979
            crs_str = str(crs).upper()
            is_epsg_3979 = ('EPSG:3979' in crs_str) or ('NAD83(CSRS) / CANADA ATLAS LAMBERT' in crs_str)
            
            if is_epsg_3979:
                print(f"  ✓ CRS verified as EPSG:3979")
            else:
                print(f"  ⚠️  CRS mismatch: {crs}")
    
    except Exception as e:
        print(f"  Error verifying file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Remap Canada land cover to Alberta classes')
    parser.add_argument('--input_gt', type=str, 
                       default='/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/GroundTruth_Landsat_Canada/landcover-2020-classification_CLIPPED.tif',
                       help='Input ground truth path (Canada classes)')
    parser.add_argument('--output_path', type=str,
                       default='/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/GroundTruth_Landsat_Canada/landcover-2020-classification_CLIPPED_ALBERTA_REMAPPED.tif',
                       help='Output path for remapped ground truth')
    parser.add_argument('--nodata', type=int, default=-99,
                       help='NoData value for invalid pixels')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remap the ground truth
    remap_ground_truth(args.input_gt, args.output_path, args.nodata)

if __name__ == '__main__':
    from datetime import datetime
    main()