import os
from osgeo import gdal

gdal.UseExceptions()

# =====================================================
# PATHS
# =====================================================
# Input directory with clipped Landsat-8 bands
input_dir = r"D:\Hackathon15_AlphaEarth\Alberta_L8_2020\Alberta_2020_NAD83_StatsCan_L8_30m_Mosaics_EPSG_3979_Clipped"

# Output directory for stacked image
output_dir = r"D:\Hackathon15_AlphaEarth\Alberta_L8_2020\Alberta_2020_NAD83_StatsCan_L8_30m_Mosaics_EPSG_3979_Clipped_Stack"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Output stacked file
output_file = os.path.join(output_dir, "Alberta_2020_L8_Stacked_6Bands.tif")

# Landsat-8 bands in order (Blue to SWIR2)
landsat_bands = [
    "SR_B2",  # Blue
    "SR_B3",  # Green
    "SR_B4",  # Red
    "SR_B5",  # NIR
    "SR_B6",  # SWIR1
    "SR_B7",  # SWIR2
]

# =====================================================
# CREATE STACKED IMAGE
# =====================================================
def stack_landsat_bands():
    """Stack all Landsat-8 bands into a single multiband TIFF"""
    print("=" * 70)
    print("STACKING LANDSAT-8 BANDS")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Bands to stack: {len(landsat_bands)}")
    print("Band order: Blue, Green, Red, NIR, SWIR1, SWIR2")
    print("=" * 70)
    
    # Collect all band file paths
    band_files = []
    missing_bands = []
    
    for band in landsat_bands:
        # Pattern for clipped Landsat-8 files
        input_file = os.path.join(input_dir, f"Alberta_2020_L8_{band}_NAD83_StatsCan_CLIPPED.tif")
        
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
    
    if len(band_files) != 6:
        print(f"\nERROR: Expected 6 bands, found {len(band_files)}")
        return False
    
    print(f"\nAll {len(band_files)} Landsat-8 bands found. Starting stacking...")
    
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
        
        print(f"\n✓ SUCCESS: Landsat-8 bands stacked!")
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
        
        print("\nBAND INFORMATION:")
        print("-" * 40)
        for i, (band_num, dtype, nodata) in enumerate(band_info):
            band_name = landsat_bands[i]
            band_desc = {
                "SR_B2": "Blue",
                "SR_B3": "Green", 
                "SR_B4": "Red",
                "SR_B5": "NIR",
                "SR_B6": "SWIR1",
                "SR_B7": "SWIR2"
            }.get(band_name, band_name)
            
            print(f"Band {band_num}: {band_name} ({band_desc})")
            print(f"  Data type: {dtype}")
            print(f"  NoData value: {nodata}")
        
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
    
    for band in landsat_bands:
        input_file = os.path.join(input_dir, f"Alberta_2020_L8_{band}_NAD83_StatsCan_CLIPPED.tif")
        
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
    print("LANDSAT-8 BAND STACKING TOOL")
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
        print("\n WARNING: Band alignment issues detected!")
        print("Stacking may still work, but results may be misaligned.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Operation cancelled.")
            exit(0)
    
    print("\n" + "=" * 70)
    print("STARTING BAND STACKING")
    print("=" * 70)
    
    # Stack the bands
    success = stack_landsat_bands()
    
    if success:
        print("\n" + "=" * 70)
        print("STACKING COMPLETE - NEXT STEPS")
        print("=" * 70)
        print("1. Verify the stacked file opens in QGIS/ArcGIS")
        print("2. Check that all 6 bands are present and in correct order:")
        print("   Band 1: SR_B2 (Blue)")
        print("   Band 2: SR_B3 (Green)")
        print("   Band 3: SR_B4 (Red)")
        print("   Band 4: SR_B5 (NIR)")
        print("   Band 5: SR_B6 (SWIR1)")
        print("   Band 6: SR_B7 (SWIR2)")
        print("3. For false color composites, use:")
        print("   - Natural color: Bands 4,3,2 (RGB)")
        print("   - False color (vegetation): Bands 5,4,3 (RGB)")
        print("   - SWIR composite: Bands 7,6,4 (RGB)")
        print("4. Compare with Sentinel-2 stacked image")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("STACKING FAILED")
        print("=" * 70)
        print("Possible issues:")
        print("1. Not all 6 bands are clipped")
        print("2. Band files have different dimensions")
        print("3. Band files have different CRS")
        print("4. Insufficient disk space")
        print("\nPlease check the input directory contains all 6 clipped bands:")
        for band in landsat_bands:
            expected_file = f"Alberta_2020_L8_{band}_NAD83_StatsCan_CLIPPED.tif"
            print(f"  - {expected_file}")
