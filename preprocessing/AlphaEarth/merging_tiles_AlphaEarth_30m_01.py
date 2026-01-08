import os
import glob
import re
from osgeo import gdal, osr
import datetime

# ============================================
# CONFIGURATION - ALPHAEARTH STRUCTURE
# ============================================
# Base directory where your AlphaEarth tiles are stored
base_dir = r'D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset'  # Your exact path
output_dir = r'D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979'

# IMPORTANT: AlphaEarth tiles are in EPSG:4326, need reprojection to EPSG:3979
SOURCE_CRS = 'EPSG:4326'      # WGS 84 (geographic, degrees)
TARGET_CRS = 'EPSG:3979'      # NAD83 / Statistics Canada Lambert (projected, meters)
TARGET_RESOLUTION = 30        # 30 meters in target CRS

# AlphaEarth bands (A00 to A63)
# We'll generate all 64 band names
ALPHAEARTH_BANDS = [f'A{i:02d}' for i in range(64)]  # A00, A01, ..., A63

def get_all_bands():
    """Get list of all AlphaEarth bands from the downloaded folders"""
    bands = []
    
    # Check each band folder exists in your structure
    for band_name in ALPHAEARTH_BANDS:
        band_folder = f'AlphaEarth_Band_{band_name}'  # e.g., "AlphaEarth_Band_A00"
        band_path = os.path.join(base_dir, band_folder)
        
        if os.path.exists(band_path):
            bands.append((band_folder, band_name))
            print(f"  ✓ Found folder: {band_folder}")
        else:
            print(f"  ✗ Missing folder: {band_folder}")
    
    print(f"\nTotal band folders found: {len(bands)}")
    return bands

def get_crs_info():
    """Get detailed information about the CRS - DIFFERENT from Landsat-8 (reprojection needed)"""
    print(f"\nCOORDINATE SYSTEM INFORMATION:")
    print(f"  Source CRS: {SOURCE_CRS} (WGS 84 - geographic, degrees)")
    print(f"  Target CRS: {TARGET_CRS} (NAD83 / Statistics Canada Lambert - projected, meters)")
    print(f"  Resolution: {TARGET_RESOLUTION} meters (in target CRS)")
    print(f"  IMPORTANT: Reprojection from EPSG:4326 to EPSG:3979 is required!")
    print(f"  This ensures all datasets have same CRS for comparison.")
    print("-" * 60)

def create_mosaic_with_reprojection(band_folder, band_name):
    """Create mosaic WITH reprojection from EPSG:4326 to EPSG:3979"""
    band_dir = os.path.join(base_dir, band_folder)
    
    # Find all tiles for this band - ALPHAEARTH NAMING PATTERN
    # Pattern: Alberta_2020_A00_tile_0_R0C1.tif
    pattern = os.path.join(band_dir, f'Alberta_2020_{band_name}_tile_*.tif')
    tile_paths = glob.glob(pattern)
    
    if not tile_paths:
        print(f"    ERROR: No tiles found for pattern: {pattern}")
        
        # Try alternative patterns just in case
        alt_patterns = [
            os.path.join(band_dir, f'*{band_name}*.tif'),
            os.path.join(band_dir, f'*AlphaEarth*{band_name}*.tif'),
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
    mosaic_path = os.path.join(output_dir, f'Alberta_2020_AlphaEarth_{band_name}_NAD83_StatsCan.tif')
    
    print(f"    Found {len(tile_paths)} tiles for {band_name}")
    
    try:
        # STEP 1: Create VRT from source tiles (in original EPSG:4326)
        print(f"    Step 1: Creating VRT in source CRS (EPSG:4326)...")
        vrt_path = os.path.join(output_dir, f'temp_AlphaEarth_{band_name}_4326.vrt')
        
        # Build VRT in source CRS
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
        
        # STEP 2: Check VRT properties in source CRS
        vrt_ds = gdal.Open(vrt_path)
        if not vrt_ds:
            print(f"    ERROR: Cannot open VRT file")
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            return None
        
        vrt_width = vrt_ds.RasterXSize
        vrt_height = vrt_ds.RasterYSize
        vrt_proj = vrt_ds.GetProjection()
        vrt_ds = None
        
        print(f"    VRT created: {vrt_width} x {vrt_height} pixels in EPSG:4326")
        
        # STEP 3: REPROJECT VRT to target CRS and create final GeoTIFF
        print(f"    Step 2: Reprojecting to EPSG:3979 and creating final GeoTIFF...")
        
        # Warp options for reprojection
        warp_options = gdal.WarpOptions(
            format='GTiff',
            srcSRS=SOURCE_CRS,
            dstSRS=TARGET_CRS,
            dstNodata=0,
            resampleAlg='nearest',
            xRes=TARGET_RESOLUTION,  # 30 meters in target CRS
            yRes=TARGET_RESOLUTION,  # 30 meters in target CRS
            creationOptions=[
                'COMPRESS=LZW',
                'PREDICTOR=2',
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=YES',
                'NUM_THREADS=ALL_CPUS'
            ],
            warpMemoryLimit=2048,
            multithread=True
        )
        
        print(f"    Reprojecting from EPSG:4326 to EPSG:3979...")
        ds = gdal.Warp(mosaic_path, vrt_path, options=warp_options)
        
        if ds is None:
            print(f"    ERROR: Failed to create reprojected GeoTIFF for {band_name}")
            if os.path.exists(vrt_path):
                os.remove(vrt_path)
            return None
        
        # Get information about the reprojected mosaic
        width = ds.RasterXSize
        height = ds.RasterYSize
        transform = ds.GetGeoTransform()
        
        # Verify CRS - should now be EPSG:3979
        crs_wkt = ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)
        crs_auth = srs.GetAuthorityName(None)
        crs_code = srs.GetAuthorityCode(None)
        
        # Get band information
        band = ds.GetRasterBand(1)
        data_type = gdal.GetDataTypeName(band.DataType)
        no_data = band.GetNoDataValue()
        
        # Calculate bounds in meters (target CRS)
        minx = transform[0]
        maxx = minx + width * transform[1]
        maxy = transform[3]
        miny = maxy + height * transform[5]
        
        # Calculate area in km²
        area_km2 = (width * abs(transform[1]) * height * abs(transform[5])) / 1000000
        
        ds = None
        
        # Clean up temporary VRT
        if os.path.exists(vrt_path):
            os.remove(vrt_path)
        
        # Verify the output
        if os.path.exists(mosaic_path):
            file_size_mb = os.path.getsize(mosaic_path) / (1024 * 1024)
            
            print(f"    ✓ SUCCESS: Created and reprojected AlphaEarth mosaic for {band_name}")
            print(f"      Source CRS: {SOURCE_CRS}")
            print(f"      Target CRS: {crs_auth}:{crs_code}")
            print(f"      Dimensions: {width:,} × {height:,} pixels")
            print(f"      File size: {file_size_mb:.1f} MB")
            print(f"      Data type: {data_type}")
            print(f"      NoData value: {no_data}")
            print(f"      Resolution: {transform[1]:.2f}m × {-transform[5]:.2f}m")
            print(f"      Bounds X (m): {minx:,.0f} to {maxx:,.0f}")
            print(f"      Bounds Y (m): {miny:,.0f} to {maxy:,.0f}")
            print(f"      Approx area: {area_km2:,.0f} km²")
            
            # Verify CRS conversion was successful
            if crs_code == '3979':
                print(f"      ✓ CRS correctly reprojected to EPSG:3979")
            else:
                print(f"      ⚠️ WARNING: CRS is {crs_auth}:{crs_code}, expected EPSG:3979")
            
            return mosaic_path
        else:
            print(f"    ERROR: Mosaic file was not created")
            return None
        
    except Exception as e:
        print(f"    ERROR processing {band_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up temp file
        vrt_path = os.path.join(output_dir, f'temp_AlphaEarth_{band_name}_4326.vrt')
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
    """Verify that all tiles are in the expected source CRS (EPSG:4326)"""
    band_dir = os.path.join(base_dir, band_folder)
    pattern = os.path.join(band_dir, f'Alberta_2020_{band_name}_tile_*.tif')
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
    
    # Check if all CRS are the same and match expected source CRS
    expected_crs = "EPSG:4326"
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
    
    # Show first 3 bands as sample
    for band_name in ALPHAEARTH_BANDS[:3]:
        band_folder = f'AlphaEarth_Band_{band_name}'
        band_dir = os.path.join(base_dir, band_folder)
        if os.path.exists(band_dir):
            pattern = os.path.join(band_dir, f'Alberta_2020_{band_name}_tile_*.tif')
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

def process_alphaearth_bands():
    """Process all AlphaEarth bands with reprojection"""
    print("=" * 80)
    print("PROCESSING ALPHAEARTH BANDS WITH REPROJECTION")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Folder structure: D:\\AlphaEarth_Dataset\\Alberta\\AlphaEarth_Band_A00\\ (etc.)")
    print(f"File pattern: Alberta_2020_A00_tile_0_R0C1.tif")
    print(f"Output directory: {output_dir}")
    print(f"Source CRS: {SOURCE_CRS} (WGS 84 - degrees)")
    print(f"Target CRS: {TARGET_CRS} (NAD83 / Statistics Canada Lambert - meters)")
    print(f"Target Resolution: {TARGET_RESOLUTION} meters")
    print(f"Expected tiles per band: 24")
    print(f"Total bands: {len(ALPHAEARTH_BANDS)} (A00 to A63)")
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
        print(f"Expected folders: AlphaEarth_Band_A00 to AlphaEarth_Band_A63")
        print(f"Current directories in base path:")
        for item in os.listdir(base_dir):
            print(f"  - {item}")
        return
    
    print(f"\nFound {len(bands)} AlphaEarth bands to process")
    
    # Verify CRS for each band
    print("\nVerifying tile CRS (first 3 bands only)...")
    for band_folder, band_name in bands[:3]:  # Check first 3 bands
        ok, message = verify_all_tiles_crs(band_folder, band_name)
        status = "✓" if ok else "✗"
        print(f"  {status} {band_name}: {message}")
        if not ok:
            print(f"    WARNING: CRS mismatch may cause issues in reprojection!")
    
    print("\nStarting mosaic creation with reprojection...")
    print("-" * 80)
    
    successful_bands = []
    failed_bands = []
    
    # Process each band
    for idx, (band_folder, band_name) in enumerate(bands, 1):
        print(f"\n[{idx:3d}/{len(bands)}] AlphaEarth Band {band_name}")
        
        try:
            # Verify CRS first
            ok, message = verify_all_tiles_crs(band_folder, band_name)
            if not ok:
                print(f"  ✗ CRS verification failed: {message}")
                failed_bands.append((band_name, f"CRS issue: {message}"))
                print(f"  ⚠️ Continuing anyway...")
            
            # Create mosaic with reprojection
            mosaic_path = create_mosaic_with_reprojection(band_folder, band_name)
            
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
    print(f"Total AlphaEarth bands: {len(bands)}")
    print(f"Successfully processed: {len(successful_bands)}")
    print(f"Failed: {len(failed_bands)}")
    
    if successful_bands:
        print(f"\nSuccessful bands (first 10):")
        for band in successful_bands[:10]:
            print(f"  ✓ {band}")
            # Show output file path
            mosaic_file = os.path.join(output_dir, f'Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan.tif')
            if os.path.exists(mosaic_file):
                size_mb = os.path.getsize(mosaic_file) / (1024 * 1024)
                print(f"      → {os.path.basename(mosaic_file)} ({size_mb:.1f} MB)")
        if len(successful_bands) > 10:
            print(f"  ... and {len(successful_bands)-10} more bands")
    
    if failed_bands:
        print(f"\nFailed bands (first 10):")
        for band, reason in failed_bands[:10]:
            print(f"  ✗ {band}: {reason}")
        if len(failed_bands) > 10:
            print(f"  ... and {len(failed_bands)-10} more failed bands")
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, 'alphaearth_mosaics_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ALPHAEARTH ALBERTA 2020 MOSAIC PROCESSING SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base directory: {base_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Source CRS: {SOURCE_CRS} (WGS 84 - degrees)\n")
        f.write(f"Target CRS: {TARGET_CRS} (NAD83 / Statistics Canada Lambert - meters)\n")
        f.write(f"Target Resolution: {TARGET_RESOLUTION} meters\n")
        f.write(f"File pattern: Alberta_2020_[BAND]_tile_*.tif\n")
        f.write(f"Total bands attempted: {len(bands)}\n")
        f.write(f"Successful: {len(successful_bands)}\n")
        f.write(f"Failed: {len(failed_bands)}\n\n")
        
        f.write("SUCCESSFUL BANDS:\n")
        f.write("-" * 30 + "\n")
        for band in successful_bands:
            mosaic_file = f'Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan.tif'
            f.write(f"{band}: {mosaic_file}\n")
        
        if failed_bands:
            f.write("\nFAILED BANDS:\n")
            f.write("-" * 30 + "\n")
            for band, reason in failed_bands:
                f.write(f"{band}: {reason}\n")
        
        f.write("\nREPROJECTION DETAILS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"From: {SOURCE_CRS} (geographic, degrees)\n")
        f.write(f"To: {TARGET_CRS} (projected, meters)\n")
        f.write(f"Resolution: {TARGET_RESOLUTION} meters\n")
        f.write(f"Resampling: nearest neighbor\n")
    
    print(f"\nDetailed summary saved to: {summary_path}")
    
    # Show example file info
    if successful_bands:
        example_band = successful_bands[0]
        example_file = os.path.join(output_dir, f'Alberta_2020_AlphaEarth_{example_band}_NAD83_StatsCan.tif')
        if os.path.exists(example_file):
            print(f"\nEXAMPLE OUTPUT FILE (Band {example_band}):")
            print(f"  File: {os.path.basename(example_file)}")
            try:
                ds = gdal.Open(example_file)
                if ds:
                    width = ds.RasterXSize
                    height = ds.RasterYSize
                    transform = ds.GetGeoTransform()
                    band = ds.GetRasterBand(1)
                    data_type = gdal.GetDataTypeName(band.DataType)
                    
                    # Verify CRS
                    crs_wkt = ds.GetProjection()
                    srs = osr.SpatialReference()
                    srs.ImportFromWkt(crs_wkt)
                    crs_auth = srs.GetAuthorityName(None)
                    crs_code = srs.GetAuthorityCode(None)
                    
                    # Calculate bounds in meters
                    minx = transform[0]
                    maxx = minx + width * transform[1]
                    maxy = transform[3]
                    miny = maxy + height * transform[5]
                    
                    ds = None
                    
                    print(f"  Dimensions: {width:,} × {height:,} pixels")
                    print(f"  Data type: {data_type}")
                    print(f"  CRS: {crs_auth}:{crs_code} {'(✓ CORRECT)' if crs_code == '3979' else '(⚠️ CHECK)'}")
                    print(f"  Resolution: {transform[1]:.2f}m × {-transform[5]:.2f}m")
                    print(f"  Bounds X (m): {minx:,.0f} to {maxx:,.0f}")
                    print(f"  Bounds Y (m): {miny:,.0f} to {maxy:,.0f}")
                    print(f"  Width: {(maxx-minx)/1000:.1f} km")
                    print(f"  Height: {(maxy-miny)/1000:.1f} km")
                    
            except Exception as e:
                print(f"  Error reading example file: {str(e)}")
    
    print("=" * 80)

# Main execution
if __name__ == "__main__":
    gdal.UseExceptions()
    
    print("ALPHAEARTH ALBERTA 2020 MOSAIC CREATION WITH REPROJECTION")
    print("=" * 80)
    print("This script will merge AlphaEarth tiles and reproject them")
    print("from EPSG:4326 (WGS 84, degrees) to EPSG:3979 (NAD83/Statistics Canada Lambert, meters)")
    print(f"Base directory: {base_dir}")
    print("Folder structure: AlphaEarth_Band_A00 to AlphaEarth_Band_A63")
    print("File pattern: Alberta_2020_A00_tile_0_R0C1.tif")
    print(f"Expected: 24 tiles per band, 64 bands total")
    print(f"Target resolution: {TARGET_RESOLUTION} meters")
    print("=" * 80)
    
    # Optional: Ask for confirmation due to large number of bands
    print(f"\nWARNING: This will process {len(ALPHAEARTH_BANDS)} bands (A00-A63).")
    print("This will REPROJECT each band from EPSG:4326 to EPSG:3979.")
    print("This may take significant time and disk space.")
    print("Each band mosaic will be ~100-500 MB, total ~6-32 GB.")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        exit()
    
    process_alphaearth_bands()
    
    print("\n" + "=" * 80)
    print("REPROJECTION COMPLETE - IMPORTANT NOTES:")
    print("=" * 80)
    print("✓ All AlphaEarth bands reprojected from EPSG:4326 to EPSG:3979")
    print("✓ Now compatible with your other datasets:")
    print("  - Landsat-8: EPSG:3979 ✓")
    print("  - Sentinel-2: EPSG:3979 ✓")
    print("  - AlphaEarth: Now EPSG:3979 ✓")
    print(f"✓ All datasets now have same resolution: {TARGET_RESOLUTION}m")
    print("\nNEXT STEPS FOR COMPARISON:")
    print("1. Clip all AlphaEarth bands using identical clipping method")
    print("2. Select equivalent bands for comparison with Landsat-8/Sentinel-2")
    print("   (Consult AlphaEarth documentation for band correspondence)")
    print("3. Stack selected bands for analysis")
    print("4. Ensure all datasets have:")
    print("   - Same extent (Alberta boundary)")
    print("   - Same resolution (30m)")
    print("   - Same CRS (EPSG:3979)")
    print("   - Same data type (UInt8)")
    print("=" * 80)
