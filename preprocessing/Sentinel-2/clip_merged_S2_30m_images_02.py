import os
import glob
from osgeo import gdal, ogr

gdal.UseExceptions()

# =====================================================
# PATHS - SENTINEL-2 SPECIFIC
# =====================================================
# Input directory with Sentinel-2 mosaics
mosaic_dir = r"D:\Hackathon15_AlphaEarth\Alberta_Sentinel2_2020\Alberta_2020_NAD83_StatsCan_Sentinel2_30m_Mosaics_EPSG_3979"

# Alberta boundary shapefile/geopackage
alberta_gpkg = r"D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta_EPSG_3979.gpkg"

# Output directory for clipped bands
output_dir = r"D:\Hackathon15_AlphaEarth\Alberta_Sentinel2_2020\Alberta_2020_NAD83_StatsCan_Sentinel2_30m_Mosaics_EPSG_3979_Clipped"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Sentinel-2 bands to process (10 bands)
sentinel_bands = [
    "B2",   # Blue
    "B3",   # Green
    "B4",   # Red
    "B5",   # Red Edge 1
    "B6",   # Red Edge 2
    "B7",   # Red Edge 3
    "B8",   # NIR
    "B8A",  # Red Edge 4
    "B11",  # SWIR 1
    "B12",  # SWIR 2
]

# =====================================================
# CLIP EACH SENTINEL-2 BAND SEPARATELY - IDENTICAL TO LANDSAT-8
# =====================================================
def clip_individual_bands():
    """Clip each Sentinel-2 band separately - IDENTICAL METHOD to Landsat-8"""
    print("=" * 70)
    print("CLIPPING INDIVIDUAL SENTINEL-2 BANDS - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    print(f"Input directory: {mosaic_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Clip boundary: {alberta_gpkg}")
    print("CRS: EPSG:3979 (NAD83 / Statistics Canada Lambert)")
    print("Resolution: 30m")
    print("=" * 70)
    
    clipped_bands = []
    failed_bands = []
    
    for band in sentinel_bands:
        # Sentinel-2 files use pattern: Alberta_2020_S2_B2_NAD83_StatsCan.tif
        input_file = os.path.join(mosaic_dir, f"Alberta_2020_S2_{band}_NAD83_StatsCan.tif")
        output_file = os.path.join(output_dir, f"Alberta_2020_S2_{band}_NAD83_StatsCan_CLIPPED.tif")
        
        print(f"\nProcessing Sentinel-2 band {band}...")
        print(f"  Input: {os.path.basename(input_file)}")
        
        if not os.path.exists(input_file):
            print(f"  ✗ ERROR: Input file not found!")
            failed_bands.append((band, "Input file not found"))
            continue
        
        try:
            # Clip the band - IDENTICAL SETTINGS to Landsat-8
            warp_options = gdal.WarpOptions(
                format="GTiff",
                cutlineDSName=alberta_gpkg,
                cropToCutline=True,          # Same as Landsat-8
                dstNodata=0,                 # Same as Landsat-8
                resampleAlg='near',          # Same as Landsat-8
                creationOptions=[
                    "COMPRESS=LZW",          # Same as Landsat-8
                    "PREDICTOR=2",           # Same as Landsat-8
                    "TILED=YES",             # Same as Landsat-8
                    "BLOCKXSIZE=256",        # Same as Landsat-8
                    "BLOCKYSIZE=256",        # Same as Landsat-8
                    "BIGTIFF=YES",           # Same as Landsat-8
                    "NUM_THREADS=ALL_CPUS"   # Same as Landsat-8
                ],
                # Preserve original resolution and CRS - IDENTICAL to Landsat-8
                xRes=30,  # 30m resolution - Same as Landsat-8
                yRes=30,  # 30m resolution - Same as Landsat-8
                targetAlignedPixels=False # Same as Landsat-8 - Allows pixel boundaries to shift
            )
            
            print(f"  Clipping to Alberta boundary...")
            ds = gdal.Warp(output_file, input_file, options=warp_options)
            ds = None
            
            # Verify the output - IDENTICAL to Landsat-8
            if os.path.exists(output_file):
                # Get file info
                ds = gdal.Open(output_file)
                width = ds.RasterXSize
                height = ds.RasterYSize
                gt = ds.GetGeoTransform()
                proj = ds.GetProjection()
                
                # Get band information
                raster_band = ds.GetRasterBand(1)
                data_type = gdal.GetDataTypeName(raster_band.DataType)
                no_data = raster_band.GetNoDataValue()
                
                ds = None
                
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                
                print(f"  ✓ SUCCESS: Clipped Sentinel-2 band {band}")
                print(f"    Output: {os.path.basename(output_file)}")
                print(f"    Size: {width} x {height} pixels")
                print(f"    Data type: {data_type}")
                print(f"    NoData value: {no_data}")
                print(f"    Resolution: {gt[1]:.2f} m")
                print(f"    File size: {file_size_mb:.1f} MB")
                print(f"    CRS preserved: {'3979' in proj}")
                
                clipped_bands.append((band, output_file))
            else:
                print(f"  ✗ ERROR: Output file was not created")
                failed_bands.append((band, "Output file creation failed"))
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed_bands.append((band, str(e)))
    
    # Summary - IDENTICAL to Landsat-8
    print("\n" + "=" * 70)
    print("CLIPPING SUMMARY - SENTINEL-2 BANDS")
    print("=" * 70)
    print(f"Total bands attempted: {len(sentinel_bands)}")
    print(f"Successfully clipped: {len(clipped_bands)}")
    print(f"Failed: {len(failed_bands)}")
    
    if clipped_bands:
        print("\nClipped Sentinel-2 bands:")
        for band, filepath in clipped_bands:
            print(f"  ✓ Band {band}: {os.path.basename(filepath)}")
    
    if failed_bands:
        print("\nFailed bands:")
        for band, reason in failed_bands:
            print(f"  ✗ Band {band}: {reason}")
    
    print(f"\nOutput directory: {output_dir}")
    return clipped_bands

# =====================================================
# OPTION 2: BATCH PROCESS ALL FILES - IDENTICAL TO LANDSAT-8
# =====================================================
def batch_clip_all_files():
    """Batch process all Sentinel-2 files automatically - IDENTICAL to Landsat-8"""
    print("=" * 70)
    print("BATCH CLIPPING ALL SENTINEL-2 FILES - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    
    # Find all Sentinel-2 mosaic files
    pattern = os.path.join(mosaic_dir, "Alberta_2020_S2_*.tif")
    mosaic_files = glob.glob(pattern)
    
    if not mosaic_files:
        print("No Sentinel-2 mosaic files found!")
        print(f"Checked pattern: {pattern}")
        return []
    
    # Filter only Sentinel-2 band files
    valid_files = []
    for filepath in mosaic_files:
        filename = os.path.basename(filepath)
        # Check if it matches our Sentinel-2 pattern
        if "Alberta_2020_S2_B" in filename and "NAD83_StatsCan.tif" in filename:
            valid_files.append(filepath)
    
    print(f"Found {len(valid_files)} Sentinel-2 mosaic files to clip")
    
    clipped_files = []
    failed_files = []
    
    for i, input_file in enumerate(valid_files, 1):
        filename = os.path.basename(input_file)
        
        # Create output filename (append _CLIPPED before .tif) - IDENTICAL to Landsat-8
        if filename.endswith(".tif"):
            output_filename = filename.replace(".tif", "_CLIPPED.tif")
        else:
            output_filename = f"{filename}_CLIPPED.tif"
        
        output_file = os.path.join(output_dir, output_filename)
        
        print(f"\n[{i}/{len(valid_files)}] Processing {filename}...")
        
        try:
            # Clip the file - IDENTICAL SETTINGS to Landsat-8 Option 2
            warp_options = gdal.WarpOptions(
                format="GTiff",
                cutlineDSName=alberta_gpkg,
                cropToCutline=True,          # Same as Landsat-8
                dstNodata=0,                 # Same as Landsat-8
                creationOptions=[
                    "COMPRESS=LZW",          # Same as Landsat-8
                    "PREDICTOR=2",           # Same as Landsat-8
                    "TILED=YES",             # Same as Landsat-8
                    "BLOCKXSIZE=256",        # Same as Landsat-8
                    "BLOCKYSIZE=256",        # Same as Landsat-8
                    "BIGTIFF=YES",           # Same as Landsat-8
                    "NUM_THREADS=ALL_CPUS"   # Same as Landsat-8
                ],
                xRes=30,                     # Same as Landsat-8
                yRes=30,                     # Same as Landsat-8
                targetAlignedPixels=False     # Same as Landsat-8 Option 2
            )
            
            ds = gdal.Warp(output_file, input_file, options=warp_options)
            ds = None
            
            if os.path.exists(output_file):
                # Verify the file - IDENTICAL to Landsat-8
                ds = gdal.Open(output_file)
                if ds:
                    width = ds.RasterXSize
                    height = ds.RasterYSize
                    gt = ds.GetGeoTransform()
                    ds = None
                    
                    print(f"  ✓ Clipped: {width} x {height} pixels, {gt[1]:.2f}m resolution")
                    clipped_files.append(output_file)
                else:
                    print(f"  ✗ ERROR: Could not verify output file")
                    failed_files.append((filename, "Verification failed"))
            else:
                print(f"  ✗ ERROR: Output file was not created")
                failed_files.append((filename, "File creation failed"))
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed_files.append((filename, str(e)))
    
    # Summary - IDENTICAL to Landsat-8
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(valid_files)}")
    print(f"Successfully clipped: {len(clipped_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if clipped_files:
        print(f"\nOutput directory: {output_dir}")
        print("Clipped files:")
        for filepath in clipped_files:
            print(f"  ✓ {os.path.basename(filepath)}")
    
    return clipped_files

# =====================================================
# MAIN EXECUTION - IDENTICAL STRUCTURE TO LANDSAT-8
# =====================================================
if __name__ == "__main__":
    print("SENTINEL-2 BAND CLIPPING TOOL - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    print(f"Input directory: {mosaic_dir}")
    print(f"Clip boundary: {alberta_gpkg}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Check if input directory exists - IDENTICAL to Landsat-8
    if not os.path.exists(mosaic_dir):
        print(f"ERROR: Input directory not found: {mosaic_dir}")
        exit(1)
    
    # Check if boundary file exists - IDENTICAL to Landsat-8
    if not os.path.exists(alberta_gpkg):
        print(f"ERROR: Boundary file not found: {alberta_gpkg}")
        exit(1)
    
    # Create output directory - IDENTICAL to Landsat-8
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSelect processing option - IDENTICAL to Landsat-8:")
    print("1. Clip known Sentinel-2 bands (B2 to B12)")
    print("2. Batch clip all Sentinel-2 files in directory (automatic detection)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        clipped_bands = clip_individual_bands()
        
    elif choice == "2":
        clipped_files = batch_clip_all_files()
        
    else:
        print("Invalid choice. Please run the script again.")
        exit(1)
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    print("✓ Each band clipped separately")
    print("✓ CRS preserved: EPSG:3979")
    print("✓ Resolution preserved: 30m")
    print("✓ Output saved to separate files")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print("\nNEXT STEPS FOR LULC COMPARISON:")
    print("✓ BOTH DATASETS CLIPPED WITH IDENTICAL SETTINGS:")
    print("   - Same cropToCutline=True")
    print("   - Same xRes=30, yRes=30")
    print("   - Same resampleAlg='near'")
    print("   - Same compression (LZW, PREDICTOR=2)")
    print("   - Same NoData value (0)")
    print("\nFor comparison, use equivalent bands:")
    print("   - Sentinel-2 B2 (Blue)      ↔ Landsat-8 SR_B2 (Blue)")
    print("   - Sentinel-2 B3 (Green)     ↔ Landsat-8 SR_B3 (Green)")
    print("   - Sentinel-2 B4 (Red)       ↔ Landsat-8 SR_B4 (Red)")
    print("   - Sentinel-2 B8 (NIR)       ↔ Landsat-8 SR_B5 (NIR)")
    print("   - Sentinel-2 B11 (SWIR1)    ↔ Landsat-8 SR_B6 (SWIR1)")
    print("   - Sentinel-2 B12 (SWIR2)    ↔ Landsat-8 SR_B7 (SWIR2)")
    print("\nEnsure both datasets have:")
    print("   - Same extent (Alberta boundary)")
    print("   - Same resolution (30m)")
    print("   - Same CRS (EPSG:3979)")
    print("   - Same data type (UInt8)")
    print("=" * 70)
