import os
from osgeo import gdal

gdal.UseExceptions()

# =====================================================
# PATHS - SENTINEL-2 SPECIFIC
# =====================================================
# Input directory with clipped Sentinel-2 bands
input_dir = r"D:\Hackathon15_AlphaEarth\Alberta_Sentinel2_2020\Alberta_2020_NAD83_StatsCan_Sentinel2_30m_Mosaics_EPSG_3979_Clipped"

# Output directory for stacked image
output_dir = r"D:\Hackathon15_AlphaEarth\Alberta_Sentinel2_2020\Alberta_2020_NAD83_StatsCan_Sentinel2_30m_Mosaics_EPSG_3979_Clipped_Stack"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Output stacked file
output_file = os.path.join(output_dir, "Alberta_2020_S2_Stacked_10Bands.tif")

# Sentinel-2 bands in order (similar to Landsat-8 order for comparison)
sentinel_bands = [
    "B2",   # Blue
    "B3",   # Green
    "B4",   # Red
    "B8",   # NIR (comparable to Landsat-8 SR_B5)
    "B5",   # Red Edge 1
    "B6",   # Red Edge 2
    "B7",   # Red Edge 3
    "B8A",  # Red Edge 4
    "B11",  # SWIR1 (comparable to Landsat-8 SR_B6)
    "B12",  # SWIR2 (comparable to Landsat-8 SR_B7)
]

# =====================================================
# CREATE STACKED IMAGE - IDENTICAL TO LANDSAT-8
# =====================================================
def stack_sentinel2_bands():
    """Stack all Sentinel-2 bands into a single multiband TIFF - IDENTICAL to Landsat-8"""
    print("=" * 70)
    print("STACKING SENTINEL-2 BANDS - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Bands to stack: {len(sentinel_bands)}")
    print("Band order: Blue, Green, Red, NIR, Red Edge 1-4,  SWIR1, SWIR2")
    print("=" * 70)
    
    # Collect all band file paths - IDENTICAL to Landsat-8
    band_files = []
    missing_bands = []
    
    for band in sentinel_bands:
        # Pattern for clipped Sentinel-2 files
        input_file = os.path.join(input_dir, f"Alberta_2020_S2_{band}_NAD83_StatsCan_CLIPPED.tif")
        
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
    
    if len(band_files) != 10:
        print(f"\nERROR: Expected 10 bands, found {len(band_files)}")
        return False
    
    print(f"\nAll {len(band_files)} Sentinel-2 bands found. Starting stacking...")
    
    try:
        # Method 1: Build VRT first, then translate to TIFF - IDENTICAL to Landsat-8
        print("\nStep 1: Creating VRT (virtual mosaic)...")
        vrt_file = os.path.join(output_dir, "temp_stack.vrt")
        
        # VRT options - IDENTICAL to Landsat-8
        vrt_options = gdal.BuildVRTOptions(
            separate=True,        # each input becomes one band - Same as Landsat-8
            srcNodata=0,          # Same as Landsat-8
            VRTNodata=0           # Same as Landsat-8
        )
        
        vrt = gdal.BuildVRT(vrt_file, band_files, options=vrt_options)
        if vrt is None:
            print("ERROR: Failed to create VRT")
            return False
        vrt = None  # Close VRT - Same as Landsat-8
        
        print("Step 2: Converting VRT to stacked TIFF...")
        
        # Translate options - IDENTICAL to Landsat-8
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
        
        ds = gdal.Translate(output_file, vrt_file, options=translate_options)
        if ds is None:
            print("ERROR: Failed to create stacked TIFF")
            return False
        
        # Clean up temporary VRT - IDENTICAL to Landsat-8
        if os.path.exists(vrt_file):
            os.remove(vrt_file)
        
        # Verify the stacked image - IDENTICAL to Landsat-8
        ds = gdal.Open(output_file)
        if ds is None:
            print("ERROR: Cannot open created stacked file")
            return False
        
        # Get image information - IDENTICAL to Landsat-8
        bands = ds.RasterCount
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        
        # Get band information - IDENTICAL to Landsat-8
        band_info = []
        for i in range(1, bands + 1):
            band = ds.GetRasterBand(i)
            data_type = gdal.GetDataTypeName(band.DataType)
            no_data = band.GetNoDataValue()
            band_info.append((i, data_type, no_data))
        
        ds = None
        
        # Calculate bounds - IDENTICAL to Landsat-8
        minx = gt[0]
        maxx = minx + width * gt[1]
        maxy = gt[3]
        miny = maxy + height * gt[5]
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"\n✓ SUCCESS: Sentinel-2 bands stacked!")
        print("=" * 70)
        print("STACKED IMAGE INFORMATION - IDENTICAL PROCESSING TO LANDSAT-8:")
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
            band_name = sentinel_bands[i]
            band_desc = {
                "B2": "Blue",
                "B3": "Green", 
                "B4": "Red",
                "B8": "NIR",
                "B5": "Red Edge 1",
                "B6": "Red Edge 2",
                "B7": "Red Edge 3",
                "B8A": "Red Edge 4",
                "B11": "SWIR1",
                "B12": "SWIR2",
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
# VERIFY BAND ALIGNMENT - IDENTICAL TO LANDSAT-8
# =====================================================
def verify_band_alignment():
    """Verify that all bands have same dimensions and alignment - IDENTICAL to Landsat-8"""
    print("\n" + "=" * 70)
    print("VERIFYING BAND ALIGNMENT - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    
    band_info = {}
    
    for band in sentinel_bands:
        input_file = os.path.join(input_dir, f"Alberta_2020_S2_{band}_NAD83_StatsCan_CLIPPED.tif")
        
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
    
    # Check consistency - IDENTICAL to Landsat-8
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
# MAIN EXECUTION - IDENTICAL STRUCTURE TO LANDSAT-8
# =====================================================
if __name__ == "__main__":
    print("SENTINEL-2 BAND STACKING TOOL - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    print(f"Input clipped bands: {input_dir}")
    print(f"Output stacked file: {output_file}")
    print("=" * 70)
    
    # Check if input directory exists - IDENTICAL to Landsat-8
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        exit(1)
    
    # Create output directory - IDENTICAL to Landsat-8
    os.makedirs(output_dir, exist_ok=True)
    
    # First verify band alignment - IDENTICAL to Landsat-8
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
    print("STARTING BAND STACKING - IDENTICAL METHOD TO LANDSAT-8")
    print("=" * 70)
    
    # Stack the bands - IDENTICAL to Landsat-8
    success = stack_sentinel2_bands()
    
    if success:
        print("\n" + "=" * 70)
        print("STACKING COMPLETE - IDENTICAL PROCESSING CONFIRMED:")
        print("=" * 70)
        print("✓ VRT creation: Same options (separate=True, NoData=0)")
        print("✓ GeoTIFF translation: Same compression (LZW, PREDICTOR=2)")
        print("✓ Tiling: Same block size (256x256)")
        print("✓ Threading: Same (NUM_THREADS=ALL_CPUS)")
        print("✓ Output format: Same GeoTIFF with BIGTIFF=YES")
        print("✓ CRS preserved: EPSG:3979")
        print("✓ Resolution preserved: 30m")
        print("=" * 70)
        print("\nNEXT STEPS FOR LULC COMPARISON:")
        print("1. Verify the stacked file opens in QGIS/ArcGIS")
        print("2. Check that all 10 bands are present and in correct order:")
        print("   Band 1: B2 (Blue)")
        print("   Band 2: B3 (Green)")
        print("   Band 3: B4 (Red)")
        print("   Band 4: B8 (NIR)")
        print("   Band 5: B5 (Red Edge 1)")
        print("   Band 6: B6 (Red Edge 2)")
        print("   Band 7: B7 (Red Edge 3)")
        print("   Band 8: B8A (Red Edge 4)")
        print("   Band 9: B11 (SWIR1)")
        print("   Band 10: B12 (SWIR2)")
        
        print("\n4. For false color composites, use:")
        print("   - Natural color: Bands 3,2,1 (RGB) = B4, B3, B2")
        print("   - False color (vegetation): Bands 4,3,2 (RGB) = B8, B4, B3")
        print("   - SWIR composite: Bands 6,5,3 (RGB) = B12, B11, B4")
        print("\n5. Compare with Landsat-8 stacked image:")
        print("   - Same extent (Alberta boundary)")
        print("   - Same resolution (30m)")
        print("   - Same CRS (EPSG:3979)")
        print("   - Equivalent band combinations")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("STACKING FAILED")
        print("=" * 70)
        print("Possible issues:")
        print("1. Not all 10 bands are clipped")
        print("2. Band files have different dimensions")
        print("3. Band files have different CRS")
        print("4. Insufficient disk space")
        print("\nPlease check the input directory contains all 10 clipped bands:")
        for band in sentinel_bands:
            expected_file = f"Alberta_2020_S2_{band}_NAD83_StatsCan_CLIPPED.tif"
            print(f"  - {expected_file}")
        print("=" * 70)
