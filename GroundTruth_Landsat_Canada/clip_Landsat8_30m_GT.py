import os
from osgeo import gdal, ogr

gdal.UseExceptions()

# =====================================================
# PATHS - GROUND TRUTH SPECIFIC
# =====================================================
# Input ground truth label map
input_file = r"D:\Hackathon15_AlphaEarth\GroundTruth_Landsat_Canada\landcover-2020-classification.tif"

# Alberta boundary shapefile/geopackage
alberta_gpkg = r"D:\Hackathon15_AlphaEarth\AlphaEarth_Dataset\Alberta_EPSG_3979.gpkg"

# Output directory for clipped ground truth
output_dir = r"D:\Hackathon15_AlphaEarth\GroundTruth_Landsat_Canada"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Output clipped file
output_file = os.path.join(output_dir, "landcover-2020-classification_CLIPPED.tif")

# =====================================================
# CLIP GROUND TRUTH LABEL MAP - IDENTICAL TO LANDSAT-8
# =====================================================
def clip_ground_truth():
    """Clip ground truth label map - IDENTICAL METHOD to Landsat-8"""
    print("=" * 70)
    print("CLIPPING GROUND TRUTH LABEL MAP - IDENTICAL TO LANDSAT-8")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Clip boundary: {alberta_gpkg}")
    print("CRS: EPSG:3979 (NAD83 / Statistics Canada Lambert)")
    print("Resolution: 30m")
    print("Data type: Byte (UInt8)")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        print(f"✗ ERROR: Input file not found!")
        print(f"Checked: {input_file}")
        return False
    
    try:
        # First, check the input file properties
        print("Checking input file properties...")
        ds = gdal.Open(input_file)
        if ds:
            width = ds.RasterXSize
            height = ds.RasterYSize
            gt = ds.GetGeoTransform()
            proj = ds.GetProjection()
            band = ds.GetRasterBand(1)
            data_type = gdal.GetDataTypeName(band.DataType)
            no_data = band.GetNoDataValue()
            
            # Calculate bounds
            minx = gt[0]
            maxx = minx + width * gt[1]
            maxy = gt[3]
            miny = maxy + height * gt[5]
            
            ds = None
            
            print(f"✓ Input file properties:")
            print(f"  Dimensions: {width} x {height} pixels")
            print(f"  Data type: {data_type}")
            print(f"  NoData value: {no_data}")
            print(f"  Resolution: {gt[1]:.2f} m")
            print(f"  Bounds X: {minx:.0f} to {maxx:.0f}")
            print(f"  Bounds Y: {miny:.0f} to {maxy:.0f}")
            print(f"  CRS: EPSG:3979 {'✓' if '3979' in proj else '✗'}")
        else:
            print("✗ ERROR: Cannot open input file")
            return False
        
        # Clip the ground truth - IDENTICAL SETTINGS to Landsat-8
        print(f"\nClipping to Alberta boundary...")
        
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
            
            # Calculate bounds
            minx = gt[0]
            maxx = minx + width * gt[1]
            maxy = gt[3]
            miny = maxy + height * gt[5]
            
            ds = None
            
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            
            print(f"✓ SUCCESS: Ground truth label map clipped!")
            print(f"  Output: {os.path.basename(output_file)}")
            print(f"  Size: {width} x {height} pixels")
            print(f"  Data type: {data_type}")
            print(f"  NoData value: {no_data}")
            print(f"  Resolution: {gt[1]:.2f} m")
            print(f"  File size: {file_size_mb:.1f} MB")
            print(f"  CRS preserved: {'3979' in proj}")
            print(f"  Bounds X (m): {minx:.0f} to {maxx:.0f}")
            print(f"  Bounds Y (m): {miny:.0f} to {maxy:.0f}")
            print(f"  Width: {(maxx-minx)/1000:.1f} km")
            print(f"  Height: {(maxy-miny)/1000:.1f} km")
            
            return True
        else:
            print(f"✗ ERROR: Output file was not created")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# =====================================================
# VERIFY COMPATIBILITY WITH OTHER DATASETS
# =====================================================
def verify_compatibility():
    """Verify that clipped ground truth is compatible with other datasets"""
    print("\n" + "=" * 70)
    print("VERIFYING COMPATIBILITY WITH OTHER DATASETS")
    print("=" * 70)
    
    if not os.path.exists(output_file):
        print("✗ Clipped ground truth not found. Please clip first.")
        return False
    
    # Load a sample Landsat-8 clipped band for comparison
    sample_landsat = r"D:\Alberta_L8_2020\Alberta_2020_NAD83_StatsCan_L8_30m_Mosaics_EPSG_3979_Clipped\Alberta_2020_L8_SR_B2_NAD83_StatsCan_CLIPPED.tif"
    
    if not os.path.exists(sample_landsat):
        print("⚠️  Cannot find Landsat-8 sample for comparison")
        print("   Please make sure Landsat-8 bands are clipped first.")
        return True  # Continue anyway
    
    try:
        # Open ground truth
        gt_ds = gdal.Open(output_file)
        gt_width = gt_ds.RasterXSize
        gt_height = gt_ds.RasterYSize
        gt_gt = gt_ds.GetGeoTransform()
        gt_proj = gt_ds.GetProjection()
        gt_ds = None
        
        # Open Landsat sample
        ls_ds = gdal.Open(sample_landsat)
        ls_width = ls_ds.RasterXSize
        ls_height = ls_ds.RasterYSize
        ls_gt = ls_ds.GetGeoTransform()
        ls_proj = ls_ds.GetProjection()
        ls_ds = None
        
        print("Comparing ground truth with Landsat-8 band (SR_B2):")
        print(f"  Ground Truth: {gt_width} x {gt_height} pixels")
        print(f"  Landsat-8:    {ls_width} x {ls_height} pixels")
        print(f"  Ground Truth resolution: {gt_gt[1]:.2f}m")
        print(f"  Landsat-8 resolution:    {ls_gt[1]:.2f}m")
        
        compatible = True
        issues = []
        
        # Check dimensions
        if gt_width != ls_width:
            issues.append(f"Width mismatch: {gt_width} vs {ls_width}")
            compatible = False
        
        if gt_height != ls_height:
            issues.append(f"Height mismatch: {gt_height} vs {ls_height}")
            compatible = False
        
        # Check resolution
        if abs(gt_gt[1] - ls_gt[1]) > 0.001:
            issues.append(f"Resolution mismatch: {gt_gt[1]:.2f}m vs {ls_gt[1]:.2f}m")
            compatible = False
        
        # Check CRS
        if '3979' not in gt_proj:
            issues.append("Ground truth CRS not EPSG:3979")
            compatible = False
        
        if '3979' not in ls_proj:
            issues.append("Landsat-8 CRS not EPSG:3979")
            compatible = False
        
        if compatible:
            print("✓ Ground truth is compatible with Landsat-8 data")
            print("  ✓ Same dimensions")
            print("  ✓ Same resolution")
            print("  ✓ Same CRS (EPSG:3979)")
        else:
            print("✗ WARNING: Compatibility issues detected!")
            for issue in issues:
                print(f"  - {issue}")
        
        return compatible
        
    except Exception as e:
        print(f"Error during compatibility check: {str(e)}")
        return False

# =====================================================
# MAIN EXECUTION - SIMPLIFIED (SINGLE FILE)
# =====================================================
if __name__ == "__main__":
    print("GROUND TRUTH LABEL MAP CLIPPING TOOL")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"Clip boundary: {alberta_gpkg}")
    print(f"Output file: {output_file}")
    print("=" * 70)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        exit(1)
    
    # Check if boundary file exists
    if not os.path.exists(alberta_gpkg):
        print(f"ERROR: Boundary file not found: {alberta_gpkg}")
        exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStarting ground truth clipping...")
    
    # Clip the ground truth
    success = clip_ground_truth()
    
    if success:
        print("\n" + "=" * 70)
        print("CLIPPING COMPLETE - IDENTICAL TO LANDSAT-8")
        print("=" * 70)
        print("✓ Ground truth clipped with identical settings:")
        print("  ✓ cropToCutline=True")
        print("  ✓ xRes=30, yRes=30")
        print("  ✓ resampleAlg='near'")
        print("  ✓ COMPRESS=LZW, PREDICTOR=2")
        print("  ✓ targetAlignedPixels=False")
        print("  ✓ NoData value: 0")
        
        # Verify compatibility
        print("\nVerifying compatibility with other datasets...")
        verify_compatibility()
        
        print("\n" + "=" * 70)
        print("NEXT STEPS FOR LULC ANALYSIS:")
        print("=" * 70)
        print("1. Ground truth is now clipped to Alberta boundary")
        print("2. All datasets have identical clipping settings:")
        print("   - Landsat-8: 6 bands clipped")
        print("   - Sentinel-2: 10 bands clipped")
        print("   - AlphaEarth: 64 bands clipped")
        print("   - Ground Truth: 1 band clipped")
        print("\n3. For model training/validation:")
        print("   - Use clipped ground truth as labels")
        print("   - Use clipped imagery as features")
        print("   - Ensure all datasets align perfectly")
        print("\n4. Check dataset alignment:")
        print("   - Same extent (Alberta boundary)")
        print("   - Same resolution (30m)")
        print("   - Same CRS (EPSG:3979)")
        print("   - Same data type (UInt8)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("CLIPPING FAILED")
        print("=" * 70)
        print("Possible issues:")
        print("1. Input file not found or inaccessible")
        print("2. Shapefile boundary not found")
        print("3. CRS mismatch between input and shapefile")
        print("4. Insufficient disk space")
        print("5. File permissions")
        print("\nPlease check:")
        print(f"  - Input file exists: {os.path.exists(input_file)}")
        print(f"  - Boundary file exists: {os.path.exists(alberta_gpkg)}")
        print(f"  - Output directory writable: {output_dir}")
        print("=" * 70)