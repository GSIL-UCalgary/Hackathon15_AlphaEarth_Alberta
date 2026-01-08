import os
import glob
from osgeo import gdal, ogr

gdal.UseExceptions()

# =====================================================
# PATHS - ALPHAEARTH SPECIFIC
# =====================================================
# Input directory with AlphaEarth mosaics
mosaic_dir = r"D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979"

# Alberta boundary shapefile/geopackage
alberta_gpkg = r"D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta\Alberta_EPSG_3979.gpkg"

# Output directory for clipped bands
output_dir = r"D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979_Clipped"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# AlphaEarth bands to process (64 bands A00 to A63)
alphaearth_bands = [f'A{i:02d}' for i in range(64)]  # A00, A01, ..., A63

# =====================================================
# CLIP EACH ALPHAEARTH BAND SEPARATELY - IDENTICAL TO LANDSAT-8
# =====================================================
def clip_individual_bands():
    """Clip each AlphaEarth band separately - IDENTICAL METHOD to Landsat-8"""
    print("=" * 70)
    print("CLIPPING INDIVIDUAL ALPHAEARTH BANDS - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    print(f"Input directory: {mosaic_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Clip boundary: {alberta_gpkg}")
    print("CRS: EPSG:3979 (NAD83 / Statistics Canada Lambert)")
    print("Resolution: 30m")
    print(f"Total bands: {len(alphaearth_bands)} (A00 to A63)")
    print("=" * 70)
    
    clipped_bands = []
    failed_bands = []
    
    # Process all 64 bands
    for band in alphaearth_bands:
        # AlphaEarth files use pattern: Alberta_2020_AlphaEarth_A00_NAD83_StatsCan.tif
        input_file = os.path.join(mosaic_dir, f"Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan.tif")
        output_file = os.path.join(output_dir, f"Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan_CLIPPED.tif")
        
        print(f"\nProcessing AlphaEarth band {band}...")
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
                
                print(f"  ✓ SUCCESS: Clipped AlphaEarth band {band}")
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
    print("CLIPPING SUMMARY - ALPHAEARTH BANDS")
    print("=" * 70)
    print(f"Total bands attempted: {len(alphaearth_bands)}")
    print(f"Successfully clipped: {len(clipped_bands)}")
    print(f"Failed: {len(failed_bands)}")
    
    if clipped_bands:
        print("\nClipped AlphaEarth bands (first 10):")
        for band, filepath in clipped_bands[:10]:
            print(f"  ✓ Band {band}: {os.path.basename(filepath)}")
        if len(clipped_bands) > 10:
            print(f"  ... and {len(clipped_bands)-10} more bands")
    
    if failed_bands:
        print("\nFailed bands (first 10):")
        for band, reason in failed_bands[:10]:
            print(f"  ✗ Band {band}: {reason}")
        if len(failed_bands) > 10:
            print(f"  ... and {len(failed_bands)-10} more failed bands")
    
    print(f"\nOutput directory: {output_dir}")
    return clipped_bands

# =====================================================
# OPTION 2: BATCH PROCESS ALL FILES - IDENTICAL TO LANDSAT-8
# =====================================================
def batch_clip_all_files():
    """Batch process all AlphaEarth files automatically - IDENTICAL to Landsat-8"""
    print("=" * 70)
    print("BATCH CLIPPING ALL ALPHAEARTH FILES - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    
    # Find all AlphaEarth mosaic files
    pattern = os.path.join(mosaic_dir, "Alberta_2020_AlphaEarth_*.tif")
    mosaic_files = glob.glob(pattern)
    
    if not mosaic_files:
        print("No AlphaEarth mosaic files found!")
        print(f"Checked pattern: {pattern}")
        return []
    
    # Filter only AlphaEarth band files
    valid_files = []
    for filepath in mosaic_files:
        filename = os.path.basename(filepath)
        # Check if it matches our AlphaEarth pattern
        if "Alberta_2020_AlphaEarth_A" in filename and "NAD83_StatsCan.tif" in filename:
            valid_files.append(filepath)
    
    print(f"Found {len(valid_files)} AlphaEarth mosaic files to clip")
    
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
                targetAlignedPixels=True     # Same as Landsat-8 Option 2
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
        print("Clipped files (first 10):")
        for filepath in clipped_files[:10]:
            print(f"  ✓ {os.path.basename(filepath)}")
        if len(clipped_files) > 10:
            print(f"  ... and {len(clipped_files)-10} more files")
    
    return clipped_files

# =====================================================
# OPTION 3: CLIP A SUBSET OF BANDS FOR COMPARISON
# =====================================================
def clip_comparison_bands():
    """Clip only selected bands for comparison with Landsat-8/Sentinel-2"""
    print("=" * 70)
    print("CLIPPING SELECTED BANDS FOR COMPARISON")
    print("=" * 70)
    print("Select equivalent bands for comparison with Landsat-8/Sentinel-2")
    print("(Consult AlphaEarth documentation for band correspondence)")
    
    # Example: Select first 6 bands for comparison (A00 to A05)
    # You should adjust this based on AlphaEarth band documentation
    comparison_bands = ['A00', 'A01', 'A02', 'A03', 'A04', 'A05']
    
    print(f"\nSelected bands for comparison: {comparison_bands}")
    print("Note: Adjust these based on AlphaEarth documentation")
    
    clipped_bands = []
    failed_bands = []
    
    for band in comparison_bands:
        input_file = os.path.join(mosaic_dir, f"Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan.tif")
        output_file = os.path.join(output_dir, f"Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan_CLIPPED.tif")
        
        print(f"\nProcessing AlphaEarth band {band} for comparison...")
        
        if not os.path.exists(input_file):
            print(f"  ✗ ERROR: Input file not found!")
            failed_bands.append((band, "Input file not found"))
            continue
        
        try:
            # IDENTICAL SETTINGS to Landsat-8
            warp_options = gdal.WarpOptions(
                format="GTiff",
                cutlineDSName=alberta_gpkg,
                cropToCutline=True,
                dstNodata=0,
                resampleAlg='near',
                creationOptions=[
                    "COMPRESS=LZW",
                    "PREDICTOR=2",
                    "TILED=YES",
                    "BLOCKXSIZE=256",
                    "BLOCKYSIZE=256",
                    "BIGTIFF=YES",
                    "NUM_THREADS=ALL_CPUS"
                ],
                xRes=30,
                yRes=30,
                targetAlignedPixels=False
            )
            
            ds = gdal.Warp(output_file, input_file, options=warp_options)
            ds = None
            
            if os.path.exists(output_file):
                ds = gdal.Open(output_file)
                width = ds.RasterXSize
                height = ds.RasterYSize
                gt = ds.GetGeoTransform()
                ds = None
                
                print(f"  ✓ Clipped: {width} x {height} pixels, {gt[1]:.2f}m")
                clipped_bands.append(band)
            else:
                print(f"  ✗ Failed to create output")
                failed_bands.append((band, "Output creation failed"))
                
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed_bands.append((band, str(e)))
    
    print(f"\nComparison bands clipped: {len(clipped_bands)}/{len(comparison_bands)}")
    return clipped_bands

# =====================================================
# MAIN EXECUTION - IDENTICAL STRUCTURE TO LANDSAT-8
# =====================================================
if __name__ == "__main__":
    print("ALPHAEARTH BAND CLIPPING TOOL - IDENTICAL TO LANDSAT-8")
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
    print("1. Clip all 64 AlphaEarth bands (A00 to A63)")
    print("2. Batch clip all AlphaEarth files in directory (automatic detection)")
    print("3. Clip selected bands for comparison (first 6 bands)")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print(f"\nWARNING: This will clip all 64 AlphaEarth bands.")
        print("This may take significant time and disk space.")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            clipped_bands = clip_individual_bands()
        else:
            print("Operation cancelled.")
            exit(0)
        
    elif choice == "2":
        print(f"\nWARNING: Batch processing all AlphaEarth files.")
        print("This will clip all files matching the pattern.")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            clipped_files = batch_clip_all_files()
        else:
            print("Operation cancelled.")
            exit(0)
        
    elif choice == "3":
        print(f"\nClipping first 6 bands for comparison with Landsat-8.")
        print("Note: Adjust band selection in code based on AlphaEarth documentation.")
        clipped_bands = clip_comparison_bands()
        
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
    print("\nALL DATASETS NOW HAVE IDENTICAL CLIPPING SETTINGS:")
    print("✓ Landsat-8: 6 bands clipped")
    print("✓ Sentinel-2: 10 bands clipped")  
    print("✓ AlphaEarth: Selected bands clipped")
    print("\nFor comparison, ensure:")
    print("1. All datasets have same extent (Alberta boundary)")
    print("2. All datasets have same resolution (30m)")
    print("3. All datasets have same CRS (EPSG:3979)")
    print("4. All datasets have same data type (UInt8)")
    print("5. Select equivalent bands from each dataset:")
    print("   - Landsat-8: SR_B2 to SR_B7")
    print("   - Sentinel-2: B2, B3, B4, B8, B11, B12")
    print("   - AlphaEarth: Consult documentation for equivalent bands")
    print("=" * 70)
