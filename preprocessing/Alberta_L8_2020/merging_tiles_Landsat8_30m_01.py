import os
import glob
import re
from osgeo import gdal, osr
import datetime

# ============================================
# CONFIGURATION - FOR YOUR EXACT STRUCTURE
# ============================================
# Base directory where your Landsat-8 tiles are stored
base_dir = r'D:\Hackathon15_AlphaEarth\Alberta_L8_2020'  # Your exact path
output_dir = r'D:\Hackathon15_AlphaEarth\Alberta_L8_2020\Alberta_2020_NAD83_StatsCan_L8_30m_Mosaics_EPSG_3979'

# IMPORTANT: Landsat-8 tiles are already in EPSG:3979 (from your GEE export)
TARGET_CRS = 'EPSG:3979'
SOURCE_CRS = 'EPSG:3979'  # Same as target - no reprojection needed!
TARGET_RESOLUTION = 30  # 30 meters

# Landsat-8 bands you downloaded (based on your description)
LANDSAT_BANDS = [
    'SR_B2',  # Blue
    'SR_B3',  # Green
    'SR_B4',  # Red
    'SR_B5',  # NIR
    'SR_B6',  # SWIR1
    'SR_B7',  # SWIR2
]

def get_all_bands():
    """Get list of all Landsat-8 bands from the downloaded folders"""
    bands = []
    
    # Check each band folder exists in your structure
    for band_name in LANDSAT_BANDS:
        band_folder = band_name  # e.g., "SR_B2", "SR_B3", etc.
        band_path = os.path.join(base_dir, band_folder)
        
        if os.path.exists(band_path):
            bands.append((band_folder, band_name))
            print(f"  ✓ Found folder: {band_folder}")
        else:
            print(f"  ✗ Missing folder: {band_folder}")
    
    print(f"\nTotal band folders found: {len(bands)}")
    return bands

def get_crs_info():
    """Get detailed information about the CRS"""
    print(f"\nCOORDINATE SYSTEM INFORMATION:")
    print(f"  Source CRS: {SOURCE_CRS}")
    print(f"  Target CRS: {TARGET_CRS}")
    print(f"  Resolution: {TARGET_RESOLUTION} meters")
    print(f"  Note: No reprojection needed - tiles are already in target CRS")
    print("-" * 60)

def create_mosaic_no_reprojection(band_folder, band_name):
    """Create mosaic WITHOUT reprojection for your exact file pattern"""
    band_dir = os.path.join(base_dir, band_folder)
    
    # Find all tiles for this band - YOUR EXACT NAMING PATTERN
    # Pattern: Alberta_2020_L8_SR_SR_B2_tile_0_R0C1.tif
    # Note the "SR_" duplication: "L8_SR_SR_B2"
    pattern = os.path.join(band_dir, f'Alberta_2020_L8_SR_{band_name}_tile_*.tif')
    tile_paths = glob.glob(pattern)
    
    if not tile_paths:
        print(f"    ERROR: No tiles found for pattern: {pattern}")
        
        # Try alternative patterns just in case
        alt_patterns = [
            os.path.join(band_dir, f'*{band_name}*.tif'),
            os.path.join(band_dir, f'*L8*{band_name}*.tif'),
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
    
    # Sort tiles by tile number for consistency
    def extract_tile_number(filename):
        """Extract tile number from filename: ...tile_XX_R..."""
        match = re.search(r'tile_(\d+)_', os.path.basename(filename))
        return int(match.group(1)) if match else 999
    
    tile_paths.sort(key=extract_tile_number)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mosaic file path
    mosaic_path = os.path.join(output_dir, f'Alberta_2020_L8_{band_name}_NAD83_StatsCan.tif')
    
    print(f"    Found {len(tile_paths)} tiles for {band_name}")
    
    try:
        # STEP 1: Create VRT from source tiles (direct merge)
        print(f"    Step 1: Creating VRT (no reprojection needed)...")
        vrt_path = os.path.join(output_dir, f'temp_L8_{band_name}.vrt')
        
        # Build VRT - since all tiles are same CRS, this works directly
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg='nearest',
            addAlpha=False,
            srcNodata=0,
            VRTNodata=0
        )
        
        print(f"    Creating VRT with {len(tile_paths)} tiles...")
        vrt = gdal.BuildVRT(vrt_path, tile_paths, options=vrt_options)
        
        if vrt is None:
            print(f"    ERROR: Failed to create VRT for {band_name}")
            return None
            
        # Flush to disk
        vrt.FlushCache()
        vrt = None
        
        # STEP 2: Check VRT properties
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
        
        # STEP 3: Translate VRT to GeoTIFF
        print(f"    Step 2: Creating final GeoTIFF mosaic...")
        
        # For Landsat-8 tiles from your GEE script, they should be aligned
        # Use translate for speed (no reprojection needed)
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=[
                'COMPRESS=LZW',
                'PREDICTOR=2',
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=YES',
                'NUM_THREADS=ALL_CPUS'
            ]
        )
        
        print(f"    Translating VRT to GeoTIFF...")
        ds = gdal.Translate(mosaic_path, vrt_path, options=translate_options)
        
        if ds is None:
            print(f"    ERROR: Failed to create GeoTIFF for {band_name}")
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            return None
        
        # Get information about the mosaic
        width = ds.RasterXSize
        height = ds.RasterYSize
        transform = ds.GetGeoTransform()
        
        # Verify CRS
        crs_wkt = ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)
        crs_auth = srs.GetAuthorityName(None)
        crs_code = srs.GetAuthorityCode(None)
        
        # Get band information
        band = ds.GetRasterBand(1)
        data_type = gdal.GetDataTypeName(band.DataType)
        no_data = band.GetNoDataValue()
        min_val, max_val, mean_val, std_val = band.ComputeStatistics(False)
        
        # Calculate bounds
        minx = transform[0]
        maxx = minx + width * transform[1]
        maxy = transform[3]
        miny = maxy + height * transform[5]
        
        # Calculate area in km²
        area_km2 = (width * abs(transform[1]) * height * abs(transform[5])) / 1000000
        
        ds = None
        
        # Clean up VRT
        if os.path.exists(vrt_path):
            os.remove(vrt_path)
        
        # Verify the output
        if os.path.exists(mosaic_path):
            file_size_mb = os.path.getsize(mosaic_path) / (1024 * 1024)
            
            print(f"    ✓ SUCCESS: Created Landsat-8 mosaic for {band_name}")
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
        
        # Clean up temp file
        vrt_path = os.path.join(output_dir, f'temp_L8_{band_name}.vrt')
        if os.path.exists(vrt_path):
            try:
                os.remove(vrt_path)
            except:
                pass
        return None

def check_tile_crs(tile_path):
    """Check the CRS of a tile"""
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
    """Verify that all tiles are in the expected CRS"""
    band_dir = os.path.join(base_dir, band_folder)
    pattern = os.path.join(band_dir, f'Alberta_2020_L8_SR_{band_name}_tile_*.tif')
    tile_paths = glob.glob(pattern)
    
    if not tile_paths:
        return False, "No tiles found"
    
    print(f"    Checking CRS for {len(tile_paths)} tiles...")
    
    # Check first 3 tiles
    crs_list = []
    for tile in tile_paths[:3]:
        crs = check_tile_crs(tile)
        crs_list.append(crs)
        print(f"      {os.path.basename(tile)}: {crs}")
    
    # Check if all CRS are the same and match expected
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
    """Show sample tile names to verify pattern"""
    print("\nSAMPLE TILE NAMES (verifying pattern):")
    print("-" * 60)
    
    for band_name in LANDSAT_BANDS[:2]:  # Check first 2 bands
        band_dir = os.path.join(base_dir, band_name)
        if os.path.exists(band_dir):
            pattern = os.path.join(band_dir, f'Alberta_2020_L8_SR_{band_name}_tile_*.tif')
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

def process_landsat8_bands():
    """Process all Landsat-8 bands"""
    print("=" * 80)
    print("PROCESSING LANDSAT-8 BANDS - YOUR EXACT STRUCTURE")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Folder structure: D:\\Alberta_L8_2020\\SR_B2\\ (etc.)")
    print(f"File pattern: Alberta_2020_L8_SR_SR_B2_tile_0_R0C1.tif")
    print(f"Output directory: {output_dir}")
    print(f"CRS: {TARGET_CRS} (NAD83 / Statistics Canada Lambert)")
    print(f"Resolution: {TARGET_RESOLUTION} meters")
    print("=" * 80)
    
    # Show sample tile names to verify pattern
    show_sample_tile_names()
    
    # Display CRS information
    get_crs_info()
    
    # Get all bands
    bands = get_all_bands()
    
    if not bands:
        print("\nERROR: No band folders found!")
        print(f"Checked in: {base_dir}")
        print(f"Expected folders: {LANDSAT_BANDS}")
        print(f"Current directories in base path:")
        for item in os.listdir(base_dir):
            print(f"  - {item}")
        return
    
    print(f"\nFound {len(bands)} Landsat-8 bands to process")
    
    # Verify CRS for each band
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
    
    # Process each band
    for idx, (band_folder, band_name) in enumerate(bands, 1):
        print(f"\n[{idx:2d}/{len(bands)}] Landsat-8 Band {band_name}")
        
        try:
            # Verify CRS first
            ok, message = verify_all_tiles_crs(band_folder, band_name)
            if not ok:
                print(f"  ✗ CRS verification failed: {message}")
                failed_bands.append((band_name, f"CRS issue: {message}"))
                # You can choose to continue or not
                print(f"  ⚠️ Continuing anyway...")
            
            # Create mosaic
            mosaic_path = create_mosaic_no_reprojection(band_folder, band_name)
            
            if mosaic_path is None:
                print(f"  ✗ Failed to create mosaic for {band_name}")
                failed_bands.append((band_name, "Mosaic creation failed"))
            else:
                successful_bands.append(band_name)
                
        except Exception as e:
            print(f"  ✗ Error processing {band_name}: {str(e)}")
            failed_bands.append((band_name, str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total Landsat-8 bands: {len(bands)}")
    print(f"Successful: {len(successful_bands)}")
    print(f"Failed: {len(failed_bands)}")
    
    if successful_bands:
        print(f"\nSuccessful bands:")
        for band in successful_bands:
            print(f"  ✓ {band}")
            # Show output file path
            mosaic_file = os.path.join(output_dir, f'Alberta_2020_L8_{band}_NAD83_StatsCan.tif')
            if os.path.exists(mosaic_file):
                size_mb = os.path.getsize(mosaic_file) / (1024 * 1024)
                print(f"      → {os.path.basename(mosaic_file)} ({size_mb:.1f} MB)")
    
    if failed_bands:
        print(f"\nFailed bands:")
        for band, reason in failed_bands:
            print(f"  ✗ {band}: {reason}")
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, 'landsat8_mosaics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("LANDSAT-8 ALBERTA 2020 MOSAIC PROCESSING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base directory: {base_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"CRS: {TARGET_CRS} (NAD83 / Statistics Canada Lambert)\n")
        f.write(f"Resolution: {TARGET_RESOLUTION} meters\n")
        f.write(f"File pattern: Alberta_2020_L8_SR_[BAND]_tile_*.tif\n")
        f.write(f"Total bands attempted: {len(bands)}\n")
        f.write(f"Successful: {len(successful_bands)}\n")
        f.write(f"Failed: {len(failed_bands)}\n\n")
        
        f.write("SUCCESSFUL BANDS:\n")
        f.write("-" * 30 + "\n")
        for band in successful_bands:
            mosaic_file = f'Alberta_2020_L8_{band}_NAD83_StatsCan.tif'
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
            f.write(f"  │   └── Alberta_2020_L8_SR_{band_folder}_tile_*.tif\n")
        f.write(f"  └── {os.path.basename(output_dir)}/\n")
        f.write(f"      └── Alberta_2020_L8_[BAND]_NAD83_StatsCan.tif\n")
    
    print(f"\nDetailed summary saved to: {summary_path}")
    
    # Show example file info
    if successful_bands:
        example_band = successful_bands[0]
        example_file = os.path.join(output_dir, f'Alberta_2020_L8_{example_band}_NAD83_StatsCan.tif')
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
                    
                    # Calculate bounds
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

# Main execution
if __name__ == "__main__":
    gdal.UseExceptions()
    
    print("LANDSAT-8 ALBERTA 2020 MOSAIC CREATION")
    print("=" * 80)
    print("This script will merge Landsat-8 tiles for your exact structure:")
    print(f"Base directory: {base_dir}")
    print("Folder structure: SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7")
    print("File pattern: Alberta_2020_L8_SR_SR_B2_tile_0_R0C1.tif")
    print("Note the 'SR_' duplication in filenames")
    print("=" * 80)

    process_landsat8_bands()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS FOR YOUR LULC COMPARISON:")
    print("=" * 80)
    print("1. Verify all 6 mosaics were created successfully")
    print("2. Check that file sizes are reasonable (100-500 MB each)")
    print("3. Open one mosaic in QGIS to verify:")
    print("   - Coverage of all Alberta")
    print("   - Correct CRS (EPSG:3979)")
    print("   - No gaps between tiles")
    print("4. Process Alpha Earth bands with similar workflow")
    print("5. Ensure both datasets:")
    print("   - Same CRS (EPSG:3979)")
    print("   - Same resolution (30m)")
    print("   - Same spatial extent")
    print("   - Same data type (uint8)")
    print("6. Stack bands for each dataset:")
    print("   Landsat-8: 6-band composite (B2, B3, B4, B5, B6, B7)")
    print("   Alpha Earth: Select equivalent bands")
    print("=" * 80)
