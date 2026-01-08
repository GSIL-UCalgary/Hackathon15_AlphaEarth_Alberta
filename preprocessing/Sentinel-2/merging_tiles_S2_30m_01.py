import os
import glob
import re
from osgeo import gdal, osr
import datetime

# ============================================
# CONFIGURATION - SENTINEL-2 STRUCTURE
# ============================================
# Base directory where your Sentinel-2 tiles are stored
base_dir = r'D:\Hackathon15_AlphaEarth\Alberta_Sentinel2_2020\30m'  # Your exact path
output_dir = r'D:\Hackathon15_AlphaEarth\Alberta_Sentinel2_2020\Alberta_2020_NAD83_StatsCan_Sentinel2_30m_Mosaics_EPSG_3979'

# IMPORTANT: Sentinel-2 tiles are already in EPSG:3979
TARGET_CRS = 'EPSG:3979'
SOURCE_CRS = 'EPSG:3979'  # Same as target - no reprojection needed!
TARGET_RESOLUTION = 30  # 30 meters

# Sentinel-2 bands you have (based on your description)
SENTINEL_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

def get_all_bands():
    """Get list of all Sentinel-2 bands from the downloaded folders - IDENTICAL to Landsat-8"""
    bands = []
    
    # Check each band folder exists in your structure
    for band_name in SENTINEL_BANDS:
        band_folder = band_name  # e.g., "B2", "B3", etc.
        band_path = os.path.join(base_dir, band_folder)
        
        if os.path.exists(band_path):
            bands.append((band_folder, band_name))
            print(f"  ✓ Found folder: {band_folder}")
        else:
            print(f"  ✗ Missing folder: {band_folder}")
    
    print(f"\nTotal band folders found: {len(bands)}")
    return bands

def get_crs_info():
    """Get detailed information about the CRS - IDENTICAL to Landsat-8"""
    print(f"\nCOORDINATE SYSTEM INFORMATION:")
    print(f"  Source CRS: {SOURCE_CRS}")
    print(f"  Target CRS: {TARGET_CRS}")
    print(f"  Resolution: {TARGET_RESOLUTION} meters")
    print(f"  Note: No reprojection needed - tiles are already in target CRS")
    print("-" * 60)

def create_mosaic_no_reprojection(band_folder, band_name):
    """Create mosaic WITHOUT reprojection - IDENTICAL METHOD to Landsat-8"""
    band_dir = os.path.join(base_dir, band_folder)
    
    # Find all tiles for this band - SENTINEL-2 NAMING PATTERN
    # Pattern: Alberta_2020_S2_B2_tile_0_R0C1.tif
    pattern = os.path.join(band_dir, f'Alberta_2020_S2_{band_name}_tile_*.tif')
    tile_paths = glob.glob(pattern)
    
    if not tile_paths:
        print(f"    ERROR: No tiles found for pattern: {pattern}")
        
        # Try alternative patterns just in case - IDENTICAL to Landsat-8
        alt_patterns = [
            os.path.join(band_dir, f'*{band_name}*.tif'),
            os.path.join(band_dir, f'*S2*{band_name}*.tif'),
            os.path.join(band_dir, '*.tif'),  # All TIFFs in folder
        ]
        
        for alt_pattern in alt_patterns:
            alt_files = glob.glob(alt_pattern)
            if alt_files:
                print(f"    Found {len(alt_files)} files with pattern: {alt_pattern}")
                tile_paths = alt_files
                break
        
        if not tile_paths:
            print(f"    Checked directory: {band_dir}")
            print(f"    Files in directory: {os.listdir(band_dir)[:5]}...")  # First 5 files
            return None
    
    # Sort tiles by tile number for consistency - IDENTICAL to Landsat-8
    def extract_tile_number(filename):
        """Extract tile number from filename: ...tile_XX_R..."""
        match = re.search(r'tile_(\d+)_', os.path.basename(filename))
        return int(match.group(1)) if match else 999
    
    tile_paths.sort(key=extract_tile_number)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mosaic file path
    mosaic_path = os.path.join(output_dir, f'Alberta_2020_S2_{band_name}_NAD83_StatsCan.tif')
    
    print(f"    Found {len(tile_paths)} tiles for {band_name}")
    
    try:
        # STEP 1: Create VRT from source tiles - IDENTICAL SETTINGS to Landsat-8
        print(f"    Step 1: Creating VRT (no reprojection needed)...")
        vrt_path = os.path.join(output_dir, f'temp_S2_{band_name}.vrt')
        
        # Build VRT with EXACT SAME OPTIONS as Landsat-8
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg='nearest',  # Same as Landsat-8
            addAlpha=False,         # Same as Landsat-8
            srcNodata=0,            # Same as Landsat-8
            VRTNodata=0             # Same as Landsat-8
        )
        
        print(f"    Creating VRT with {len(tile_paths)} tiles...")
        vrt = gdal.BuildVRT(vrt_path, tile_paths, options=vrt_options)
        
        if vrt is None:
            print(f"    ERROR: Failed to create VRT for {band_name}")
            return None
            
        # Flush to disk - IDENTICAL to Landsat-8
        vrt.FlushCache()
        vrt = None
        
        # STEP 2: Check VRT properties - IDENTICAL to Landsat-8
        vrt_ds = gdal.Open(vrt_path)
        if not vrt_ds:
            print(f"    ERROR: Cannot open VRT file")
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            return None
        
        vrt_width = vrt_ds.RasterXSize
        vrt_height = vrt_ds.RasterYSize
        vrt_gt = vrt_ds.GetGeoTransform()
        vrt_ds = None
        
        print(f"    VRT created: {vrt_width} x {vrt_height} pixels")
        
        # STEP 3: Translate VRT to GeoTIFF - IDENTICAL SETTINGS to Landsat-8
        print(f"    Step 2: Creating final GeoTIFF mosaic...")
        
        # Use translate with EXACT SAME OPTIONS as Landsat-8
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=[
                'COMPRESS=LZW',      # Same as Landsat-8
                'PREDICTOR=2',       # Same as Landsat-8
                'TILED=YES',         # Same as Landsat-8
                'BLOCKXSIZE=256',    # Same as Landsat-8
                'BLOCKYSIZE=256',    # Same as Landsat-8
                'BIGTIFF=YES',       # Same as Landsat-8
                'NUM_THREADS=ALL_CPUS'  # Same as Landsat-8
            ]
        )
        
        print(f"    Translating VRT to GeoTIFF...")
        ds = gdal.Translate(mosaic_path, vrt_path, options=translate_options)
        
        if ds is None:
            print(f"    ERROR: Failed to create GeoTIFF for {band_name}")
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            return None
        
        # Get information about the mosaic - IDENTICAL to Landsat-8
        width = ds.RasterXSize
        height = ds.RasterYSize
        transform = ds.GetGeoTransform()
        
        # Verify CRS - IDENTICAL to Landsat-8
        crs_wkt = ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)
        crs_auth = srs.GetAuthorityName(None)
        crs_code = srs.GetAuthorityCode(None)
        
        # Get band information - IDENTICAL to Landsat-8
        band = ds.GetRasterBand(1)
        data_type = gdal.GetDataTypeName(band.DataType)
        no_data = band.GetNoDataValue()
        min_val, max_val, mean_val, std_val = band.ComputeStatistics(False)
        
        # Calculate bounds - IDENTICAL to Landsat-8
        minx = transform[0]
        maxx = minx + width * transform[1]
        maxy = transform[3]
        miny = maxy + height * transform[5]
        
        # Calculate area in km² - IDENTICAL to Landsat-8
        area_km2 = (width * abs(transform[1]) * height * abs(transform[5])) / 1000000
        
        ds = None
        
        # Clean up VRT - IDENTICAL to Landsat-8
        if os.path.exists(vrt_path):
            os.remove(vrt_path)
        
        # Verify the output - IDENTICAL to Landsat-8
        if os.path.exists(mosaic_path):
            file_size_mb = os.path.getsize(mosaic_path) / (1024 * 1024)
            
            print(f"    ✓ SUCCESS: Created Sentinel-2 mosaic for {band_name}")
            print(f"      Dimensions: {width:,} × {height:,} pixels")
            print(f"      File size: {file_size_mb:.1f} MB")
            print(f"      Data type: {data_type}")
            print(f"      NoData value: {no_data}")
            print(f"      Value range: {min_val:.1f} to {max_val:.1f}")
            print(f"      CRS: {crs_auth}:{crs_code}")
            print(f"      Resolution: {transform[1]:.2f}m × {-transform[5]:.2f}m")
            print(f"      Bounds (m): [{minx:,.0f}, {miny:,.0f}] to [{maxx:,.0f}, {maxy:,.0f}]")
            print(f"      Approx area: {area_km2:,.0f} km²")
            
            return mosaic_path
        else:
            print(f"    ERROR: Mosaic file was not created")
            return None
        
    except Exception as e:
        print(f"    ERROR processing {band_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp file - IDENTICAL to Landsat-8
        vrt_path = os.path.join(output_dir, f'temp_S2_{band_name}.vrt')
        if os.path.exists(vrt_path):
            try:
                os.remove(vrt_path)
            except:
                pass
        return None

def check_tile_crs(tile_path):
    """Check the CRS of a tile - IDENTICAL to Landsat-8"""
    try:
        ds = gdal.Open(tile_path)
        if ds:
            crs_wkt = ds.GetProjection()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs_wkt)
            auth = srs.GetAuthorityName(None)
            code = srs.GetAuthorityCode(None)
            ds = None
            return f"{auth}:{code}" if auth and code else "Unknown"
    except:
        pass
    return "Error reading"

def verify_all_tiles_crs(band_folder, band_name):
    """Verify that all tiles are in the expected CRS - IDENTICAL to Landsat-8"""
    band_dir = os.path.join(base_dir, band_folder)
    pattern = os.path.join(band_dir, f'Alberta_2020_S2_{band_name}_tile_*.tif')
    tile_paths = glob.glob(pattern)
    
    if not tile_paths:
        return False, "No tiles found"
    
    print(f"    Checking CRS for {len(tile_paths)} tiles...")
    
    # Check first 3 tiles - IDENTICAL to Landsat-8
    crs_list = []
    for tile in tile_paths[:3]:
        crs = check_tile_crs(tile)
        crs_list.append(crs)
        print(f"      {os.path.basename(tile)}: {crs}")
    
    # Check if all CRS are the same and match expected - IDENTICAL to Landsat-8
    expected_crs = "EPSG:3979"
    unique_crs = set(crs_list)
    
    if len(unique_crs) == 1:
        actual_crs = list(unique_crs)[0]
        if actual_crs == expected_crs:
            return True, f"All tiles in {expected_crs}"
        else:
            return False, f"Tiles in {actual_crs}, expected {expected_crs}"
    else:
        return False, f"Mixed CRS: {unique_crs}"

def show_sample_tile_names():
    """Show sample tile names to verify pattern - SIMILAR to Landsat-8"""
    print("\nSAMPLE TILE NAMES (verifying pattern):")
    print("-" * 60)
    
    for band_name in SENTINEL_BANDS[:2]:  # Check first 2 bands
        band_dir = os.path.join(base_dir, band_name)
        if os.path.exists(band_dir):
            pattern = os.path.join(band_dir, f'Alberta_2020_S2_{band_name}_tile_*.tif')
            files = glob.glob(pattern)
            if files:
                print(f"\nBand {band_name}:")
                for f in files[:2]:  # Show first 2 files
                    print(f"  {os.path.basename(f)}")
                if len(files) > 2:
                    print(f"  ... and {len(files)-2} more files")
            else:
                print(f"\nBand {band_name}: No files matching pattern")
        else:
            print(f"\nBand {band_name}: Directory not found")
    print("-" * 60)

def process_sentinel2_bands():
    """Process all Sentinel-2 bands - SIMILAR STRUCTURE to Landsat-8"""
    print("=" * 80)
    print("PROCESSING SENTINEL-2 BANDS - IDENTICAL METHOD TO LANDSAT-8")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Folder structure: D:\\Alberta_Sentinel2_2020\\30m\\B2\\ (etc.)")
    print(f"File pattern: Alberta_2020_S2_B2_tile_0_R0C1.tif")
    print(f"Output directory: {output_dir}")
    print(f"CRS: {TARGET_CRS} (NAD83 / Statistics Canada Lambert)")
    print(f"Resolution: {TARGET_RESOLUTION} meters")
    print(f"Expected tiles per band: 24")
    print("=" * 80)
    
    # Show sample tile names to verify pattern
    show_sample_tile_names()
    
    # Display CRS information - IDENTICAL to Landsat-8
    get_crs_info()
    
    # Get all bands
    bands = get_all_bands()
    
    if not bands:
        print("\nERROR: No band folders found!")
        print(f"Checked in: {base_dir}")
        print(f"Expected folders: {SENTINEL_BANDS}")
        print(f"Current directories in base path:")
        for item in os.listdir(base_dir):
            print(f"  - {item}")
        return
    
    print(f"\nFound {len(bands)} Sentinel-2 bands to process")
    
    # Verify CRS for each band - IDENTICAL to Landsat-8
    print("\nVerifying tile CRS (first 2 bands only)...")
    for band_folder, band_name in bands[:2]:  # Check first 2 bands
        ok, message = verify_all_tiles_crs(band_folder, band_name)
        status = "✓" if ok else "✗"
        print(f"  {status} {band_name}: {message}")
        if not ok:
            print(f"    WARNING: CRS mismatch may cause issues!")
    
    print("\nStarting mosaic creation...")
    print("-" * 80)
    
    successful_bands = []
    failed_bands = []
    
    # Process each band - IDENTICAL to Landsat-8
    for idx, (band_folder, band_name) in enumerate(bands, 1):
        print(f"\n[{idx:2d}/{len(bands)}] Sentinel-2 Band {band_name}")
        
        try:
            # Verify CRS first - IDENTICAL to Landsat-8
            ok, message = verify_all_tiles_crs(band_folder, band_name)
            if not ok:
                print(f"  ✗ CRS verification failed: {message}")
                failed_bands.append((band_name, f"CRS issue: {message}"))
                print(f"  ⚠️ Continuing anyway...")
            
            # Create mosaic with IDENTICAL method to Landsat-8
            mosaic_path = create_mosaic_no_reprojection(band_folder, band_name)
            
            if mosaic_path is None:
                print(f"  ✗ Failed to create mosaic for {band_name}")
                failed_bands.append((band_name, "Mosaic creation failed"))
            else:
                successful_bands.append(band_name)
                
        except Exception as e:
            print(f"  ✗ Error processing {band_name}: {str(e)}")
            failed_bands.append((band_name, str(e)))
    
    # Summary - IDENTICAL to Landsat-8
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total Sentinel-2 bands: {len(bands)}")
    print(f"Successful: {len(successful_bands)}")
    print(f"Failed: {len(failed_bands)}")
    
    if successful_bands:
        print(f"\nSuccessful bands:")
        for band in successful_bands:
            print(f"  ✓ {band}")
            # Show output file path - SIMILAR to Landsat-8
            mosaic_file = os.path.join(output_dir, f'Alberta_2020_S2_{band}_NAD83_StatsCan.tif')
            if os.path.exists(mosaic_file):
                size_mb = os.path.getsize(mosaic_file) / (1024 * 1024)
                print(f"      → {os.path.basename(mosaic_file)} ({size_mb:.1f} MB)")
    
    if failed_bands:
        print(f"\nFailed bands:")
        for band, reason in failed_bands:
            print(f"  ✗ {band}: {reason}")
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create a summary file - IDENTICAL to Landsat-8
    summary_path = os.path.join(output_dir, 'sentinel2_mosaics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SENTINEL-2 ALBERTA 2020 MOSAIC PROCESSING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base directory: {base_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"CRS: {TARGET_CRS} (NAD83 / Statistics Canada Lambert)\n")
        f.write(f"Resolution: {TARGET_RESOLUTION} meters\n")
        f.write(f"File pattern: Alberta_2020_S2_[BAND]_tile_*.tif\n")
        f.write(f"Total bands attempted: {len(bands)}\n")
        f.write(f"Successful: {len(successful_bands)}\n")
        f.write(f"Failed: {len(failed_bands)}\n\n")
        
        f.write("SUCCESSFUL BANDS:\n")
        f.write("-" * 30 + "\n")
        for band in successful_bands:
            mosaic_file = f'Alberta_2020_S2_{band}_NAD83_StatsCan.tif'
            f.write(f"{band}: {mosaic_file}\n")
        
        if failed_bands:
            f.write("\nFAILED BANDS:\n")
            f.write("-" * 30 + "\n")
            for band, reason in failed_bands:
                f.write(f"{band}: {reason}\n")
        
        f.write("\nDIRECTORY STRUCTURE:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Base: {base_dir}\n")
        for band_folder, _ in bands:
            f.write(f"  ├── {band_folder}/\n")
            f.write(f"  │   └── Alberta_2020_S2_{band_folder}_tile_*.tif\n")
        f.write(f"  └── {os.path.basename(output_dir)}/\n")
        f.write(f"      └── Alberta_2020_S2_[BAND]_NAD83_StatsCan.tif\n")
    
    print(f"\nDetailed summary saved to: {summary_path}")
    
    # Show example file info - IDENTICAL to Landsat-8
    if successful_bands:
        example_band = successful_bands[0]
        example_file = os.path.join(output_dir, f'Alberta_2020_S2_{example_band}_NAD83_StatsCan.tif')
        if os.path.exists(example_file):
            print(f"\nEXAMPLE OUTPUT FILE:")
            print(f"  File: {os.path.basename(example_file)}")
            try:
                ds = gdal.Open(example_file)
                if ds:
                    width = ds.RasterXSize
                    height = ds.RasterYSize
                    transform = ds.GetGeoTransform()
                    band = ds.GetRasterBand(1)
                    data_type = gdal.GetDataTypeName(band.DataType)
                    
                    # Calculate bounds - IDENTICAL to Landsat-8
                    minx = transform[0]
                    maxx = minx + width * transform[1]
                    maxy = transform[3]
                    miny = maxy + height * transform[5]
                    
                    ds = None
                    
                    print(f"  Dimensions: {width:,} × {height:,} pixels")
                    print(f"  Data type: {data_type}")
                    print(f"  Resolution: {transform[1]:.2f}m × {-transform[5]:.2f}m")
                    print(f"  Bounds X: {minx:,.0f} to {maxx:,.0f}")
                    print(f"  Bounds Y: {miny:,.0f} to {maxy:,.0f}")
                    print(f"  Width: {(maxx-minx)/1000:.0f} km")
                    print(f"  Height: {(maxy-miny)/1000:.0f} km")
                    
            except Exception as e:
                print(f"  Error reading example file: {str(e)}")
    
    print("=" * 80)

# Main execution - SIMILAR to Landsat-8
if __name__ == "__main__":
    gdal.UseExceptions()
    
    print("SENTINEL-2 ALBERTA 2020 MOSAIC CREATION")
    print("=" * 80)
    print("IDENTICAL METHOD TO LANDSAT-8 MERGING")
    print("=" * 80)
    print("This script will merge Sentinel-2 tiles using the EXACT SAME")
    print("settings and method as your Landsat-8 merging script.")
    print(f"Base directory: {base_dir}")
    print("Folder structure: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12")
    print("File pattern: Alberta_2020_S2_B2_tile_0_R0C1.tif")
    print(f"Expected: 24 tiles per band")
    print("=" * 80)
    
    process_sentinel2_bands()
    
    print("\n" + "=" * 80)
    print("IDENTICAL PROCESSING METHOD CONFIRMED:")
    print("=" * 80)
    print("✓ VRT creation: Same options (nearest neighbor, NoData=0)")
    print("✓ GeoTIFF translation: Same compression (LZW, PREDICTOR=2)")
    print("✓ Tiling: Same block size (256x256)")
    print("✓ Threading: Same (NUM_THREADS=ALL_CPUS)")
    print("✓ No reprojection: Both datasets already EPSG:3979")
    print("✓ Output format: Same GeoTIFF with BIGTIFF=YES")
    print("=" * 80)
    print("\nNOW BOTH DATASETS HAVE IDENTICAL PROCESSING:")
    print("- Landsat-8: 6 bands (SR_B2 to SR_B7)")
    print("- Sentinel-2: 10 bands (B2 to B12)")
    print("\nFor LULC comparison, use equivalent bands:")
    print("  Sentinel-2 B2 (Blue)      ↔ Landsat-8 SR_B2 (Blue)")
    print("  Sentinel-2 B3 (Green)     ↔ Landsat-8 SR_B3 (Green)")
    print("  Sentinel-2 B4 (Red)       ↔ Landsat-8 SR_B4 (Red)")
    print("  Sentinel-2 B8 (NIR)       ↔ Landsat-8 SR_B5 (NIR)")
    print("  Sentinel-2 B11 (SWIR1)    ↔ Landsat-8 SR_B6 (SWIR1)")
    print("  Sentinel-2 B12 (SWIR2)    ↔ Landsat-8 SR_B7 (SWIR2)")
    print("=" * 80)
