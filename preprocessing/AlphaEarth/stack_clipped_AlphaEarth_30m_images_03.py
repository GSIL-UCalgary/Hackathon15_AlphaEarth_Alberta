import os
from osgeo import gdal

gdal.UseExceptions()

# =====================================================
# PATHS - ALPHAEARTH SPECIFIC
# =====================================================
# Input directory with clipped AlphaEarth bands
input_dir = r"D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979_Clipped"

# Output directory for stacked image
output_dir = r"D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979_Clipped_Stack"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Output stacked file
output_file = os.path.join(output_dir, "Alberta_2020_AlphaEarth_Stacked_64Bands.tif")

# AlphaEarth bands in order (A00 to A63)
alphaearth_bands = [f'A{i:02d}' for i in range(64)]  # A00, A01, ..., A63

# =====================================================
# CREATE STACKED IMAGE - IDENTICAL TO LANDSAT-8
# =====================================================
def stack_alphaearth_bands():
    """Stack all AlphaEarth bands into a single multiband TIFF - IDENTICAL to Landsat-8"""
    print("=" * 70)
    print("STACKING ALPHAEARTH BANDS")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Bands to stack: {len(alphaearth_bands)}")
    print("Band order: A00 to A63")
    print("=" * 70)
    
    # Collect all band file paths
    band_files = []
    missing_bands = []
    
    for band in alphaearth_bands:
        # Pattern for clipped AlphaEarth files
        input_file = os.path.join(input_dir, f"Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan_CLIPPED.tif")
        
        if os.path.exists(input_file):
            band_files.append(input_file)
            print(f"✓ Found band {band}: {os.path.basename(input_file)}")
        else:
            missing_bands.append(band)
            print(f"✗ Missing band {band}: {os.path.basename(input_file)}")
    
    if missing_bands:
        print(f"\nERROR: Missing {len(missing_bands)} band(s): {missing_bands}")
        print("Please make sure all bands are clipped before stacking.")
        return False
    
    if len(band_files) != 64:
        print(f"\nERROR: Expected 64 bands, found {len(band_files)}")
        return False
    
    print(f"\nAll {len(band_files)} AlphaEarth bands found. Starting stacking...")
    
    try:
        # Method 1: Build VRT first, then translate to TIFF (recommended)
        print("\nStep 1: Creating VRT (virtual mosaic)...")
        vrt_file = os.path.join(output_dir, "temp_stack.vrt")
        
        vrt_options = gdal.BuildVRTOptions(
            separate=True,        # each input becomes one band
            srcNodata=0,
            VRTNodata=0
        )
        
        vrt = gdal.BuildVRT(vrt_file, band_files, options=vrt_options)
        if vrt is None:
            print("ERROR: Failed to create VRT")
            return False
        vrt = None  # Close VRT
        
        print("Step 2: Converting VRT to stacked TIFF...")
        
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
        
        ds = gdal.Translate(output_file, vrt_file, options=translate_options)
        if ds is None:
            print("ERROR: Failed to create stacked TIFF")
            return False
        
        # Clean up temporary VRT
        if os.path.exists(vrt_file):
            os.remove(vrt_file)
        
        # Verify the stacked image
        ds = gdal.Open(output_file)
        if ds is None:
            print("ERROR: Cannot open created stacked file")
            return False
        
        # Get image information
        bands = ds.RasterCount
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        
        # Get band information
        band_info = []
        for i in range(1, bands + 1):
            band = ds.GetRasterBand(i)
            data_type = gdal.GetDataTypeName(band.DataType)
            no_data = band.GetNoDataValue()
            band_info.append((i, data_type, no_data))
        
        ds = None
        
        # Calculate bounds
        minx = gt[0]
        maxx = minx + width * gt[1]
        maxy = gt[3]
        miny = maxy + height * gt[5]
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"\n✓ SUCCESS: AlphaEarth bands stacked!")
        print("=" * 70)
        print("STACKED IMAGE INFORMATION:")
        print("=" * 70)
        print(f"Output file: {os.path.basename(output_file)}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Bands: {bands}")
        print(f"Dimensions: {width} x {height} pixels")
        print(f"Resolution: {gt[1]:.2f} m")
        print(f"CRS: EPSG:3979 (NAD83/Statistics Canada Lambert)")
        print(f"Bounds (m):")
        print(f"  X: {minx:.0f} to {maxx:.0f}")
        print(f"  Y: {miny:.0f} to {maxy:.0f}")
        print(f"  Width: {(maxx-minx)/1000:.1f} km")
        print(f"  Height: {(maxy-miny)/1000:.1f} km")
        
        print("\nBAND INFORMATION (first 10 bands):")
        print("-" * 40)
        for i, (band_num, dtype, nodata) in enumerate(band_info[:10]):
            band_name = alphaearth_bands[i]
            print(f"Band {band_num}: {band_name}")
            print(f"  Data type: {dtype}")
            print(f"  NoData value: {nodata}")
        
        if len(band_info) > 10:
            print(f"\n... and {len(band_info)-10} more bands")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR during stacking: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# =====================================================
# VERIFY BAND ALIGNMENT
# =====================================================
def verify_band_alignment():
    """Verify that all bands have same dimensions and alignment"""
    print("\n" + "=" * 70)
    print("VERIFYING BAND ALIGNMENT")
    print("=" * 70)
    
    band_info = {}
    
    for band in alphaearth_bands:
        input_file = os.path.join(input_dir, f"Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan_CLIPPED.tif")
        
        if not os.path.exists(input_file):
            print(f"✗ Missing: {band}")
            continue
            
        ds = gdal.Open(input_file)
        if ds:
            width = ds.RasterXSize
            height = ds.RasterYSize
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            ds = None
            
            band_info[band] = {
                'width': width,
                'height': height,
                'geotransform': gt,
                'projection': proj
            }
            
            print(f"✓ {band}: {width}x{height}, Resolution: {gt[1]:.2f}m")
    
    # Check consistency
    if band_info:
        first_band = list(band_info.keys())[0]
        first_info = band_info[first_band]
        
        consistent = True
        issues = []
        
        for band, info in band_info.items():
            if band == first_band:
                continue
                
            if info['width'] != first_info['width']:
                issues.append(f"{band}: Width mismatch ({info['width']} vs {first_info['width']})")
                consistent = False
                
            if info['height'] != first_info['height']:
                issues.append(f"{band}: Height mismatch ({info['height']} vs {first_info['height']})")
                consistent = False
        
        if consistent:
            print("\n✓ All bands have consistent dimensions")
            print(f"  All bands: {first_info['width']} x {first_info['height']} pixels")
            print(f"  Resolution: {first_info['geotransform'][1]:.2f} m")
        else:
            print("\n✗ WARNING: Band dimension mismatches detected!")
            for issue in issues:
                print(f"  {issue}")
            
        return consistent
    else:
        print("No bands found to verify")
        return False

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    print("ALPHAEARTH BAND STACKING TOOL")
    print("=" * 70)
    print(f"Input clipped bands: {input_dir}")
    print(f"Output stacked file: {output_file}")
    print("=" * 70)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # First verify band alignment
    print("\nVerifying band alignment before stacking...")
    alignment_ok = verify_band_alignment()
    
    if not alignment_ok:
        print("\nWARNING: Band alignment issues detected!")
        print("Stacking may still work, but results may be misaligned.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Operation cancelled.")
            exit(0)
    
    print("\n" + "=" * 70)
    print("STARTING BAND STACKING")
    print("=" * 70)
    
    # Stack the bands
    success = stack_alphaearth_bands()
    
    if success:
        print("\n" + "=" * 70)
        print("STACKING COMPLETE - NEXT STEPS")
        print("=" * 70)
        print("1. Verify the stacked file opens in QGIS/ArcGIS")
        print("2. Check that all 64 bands are present and in correct order:")
        print("   Band 1: A00")
        print("   Band 2: A01")
        print("   Band 3: A02")
        print("   ...")
        print("   Band 64: A63")
        print("3. Compare with other stacked images:")
        print("   - Landsat-8: 6-band stack")
        print("   - Sentinel-2: 10-band stack")
        print("   - AlphaEarth: 64-band stack")
        print("4. Ensure all datasets have:")
        print("   - Same extent (Alberta boundary)")
        print("   - Same resolution (30m)")
        print("   - Same CRS (EPSG:3979)")
        print("   - Same data type (UInt8)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("STACKING FAILED")
        print("=" * 70)
        print("Possible issues:")
        print("1. Not all 64 bands are clipped")
        print("2. Band files have different dimensions")
        print("3. Band files have different CRS")
        print("4. Insufficient disk space")
        print("\nPlease check the input directory contains all 64 clipped bands:")
        for i in range(0, 64, 10):  # Show in groups of 10
            print(f"  Bands {i:02d}-{i+9:02d}: ", end="")
            bands = [f'A{j:02d}' for j in range(i, min(i+10, 64))]
            for band in bands:
                expected = f"Alberta_2020_AlphaEarth_{band}_NAD83_StatsCan_CLIPPED.tif"
                if os.path.exists(os.path.join(input_dir, expected)):
                    print(f"{band}✓ ", end="")
                else:
                    print(f"{band}✗ ", end="")
            print()
