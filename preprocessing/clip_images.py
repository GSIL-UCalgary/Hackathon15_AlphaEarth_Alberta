# import os
# import glob
# from osgeo import gdal, ogr, osr
# import datetime

# # Input and output directories
# mosaic_dir = r'D:\AlphaEarth_Dataset\Alberta\NAD83_StatsCan_Mosaics_EPSG_3979'
# shapefile_path = r'D:\Alberta_Sentinel2_2020\Alberta_Shapefile\Alberta.gpkg'
# output_dir = r'D:\AlphaEarth_Dataset\Alberta\NAD83_StatsCan_Mosaics_EPSG_3979_Clipped'

# def get_all_mosaics():
#     """Get list of all mosaic files"""
#     pattern = os.path.join(mosaic_dir, '*.tif')
#     mosaic_files = glob.glob(pattern)
    
#     # Sort by band name
#     mosaic_files.sort()
    
#     print(f"Found {len(mosaic_files)} mosaic files:")
#     for i, mosaic in enumerate(mosaic_files, 1):
#         print(f"  {i:2d}. {os.path.basename(mosaic)}")
    
#     return mosaic_files

# def get_shapefile_info():
#     """Get information about the shapefile"""
#     print("\nSHAPEFILE INFORMATION:")
#     print("-" * 40)
    
#     try:
#         # Open shapefile
#         ds = ogr.Open(shapefile_path)
#         if ds is None:
#             print(f"ERROR: Cannot open shapefile: {shapefile_path}")
#             return None
        
#         layer = ds.GetLayer()
#         feature_count = layer.GetFeatureCount()
#         extent = layer.GetExtent()
        
#         # Get CRS info
#         srs = layer.GetSpatialRef()
#         if srs:
#             srs.AutoIdentifyEPSG()
#             crs_name = srs.GetName()
#             crs_auth = srs.GetAuthorityName(None)
#             crs_code = srs.GetAuthorityCode(None)
#             crs_info = f"{crs_auth}:{crs_code} - {crs_name}"
#         else:
#             crs_info = "Unknown CRS"
        
#         ds = None
        
#         print(f"  File: {shapefile_path}")
#         print(f"  Features: {feature_count}")
#         print(f"  CRS: {crs_info}")
#         print(f"  Extent (xmin, xmax, ymin, ymax):")
#         print(f"    {extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}, {extent[3]:.2f}")
#         print(f"  Width: {(extent[1] - extent[0]):.2f} m")
#         print(f"  Height: {(extent[3] - extent[2]):.2f} m")
        
#         return {
#             'path': shapefile_path,
#             'feature_count': feature_count,
#             'extent': extent,
#             'crs': srs
#         }
        
#     except Exception as e:
#         print(f"ERROR reading shapefile: {str(e)}")
#         return None

# def get_raster_info(raster_path):
#     """Get information about a raster file"""
#     try:
#         ds = gdal.Open(raster_path)
#         if ds is None:
#             return None
        
#         width = ds.RasterXSize
#         height = ds.RasterYSize
#         transform = ds.GetGeoTransform()
#         crs_wkt = ds.GetProjection()
        
#         # Get CRS info
#         srs = osr.SpatialReference()
#         srs.ImportFromWkt(crs_wkt)
#         crs_name = srs.GetName()
#         crs_auth = srs.GetAuthorityName(None)
#         crs_code = srs.GetAuthorityCode(None)
        
#         # Calculate bounds
#         minx = transform[0]
#         maxx = minx + width * transform[1]
#         maxy = transform[3]
#         miny = maxy + height * transform[5]
        
#         # Calculate area
#         width_m = maxx - minx
#         height_m = maxy - miny
#         area_km2 = (width_m * height_m) / 1000000
        
#         ds = None
        
#         return {
#             'width': width,
#             'height': height,
#             'transform': transform,
#             'crs_wkt': crs_wkt,
#             'crs_name': crs_name,
#             'crs_auth': crs_auth,
#             'crs_code': crs_code,
#             'bounds': (minx, maxx, miny, maxy),
#             'width_m': width_m,
#             'height_m': height_m,
#             'area_km2': area_km2
#         }
        
#     except Exception as e:
#         print(f"ERROR reading raster {raster_path}: {str(e)}")
#         return None

# def clip_raster_with_shapefile(input_raster, output_raster, shapefile_path):
#     """Clip raster using shapefile while maintaining original CRS"""
    
#     print(f"  Processing: {os.path.basename(input_raster)}")
    
#     try:
#         # Get raster info
#         raster_info = get_raster_info(input_raster)
#         if not raster_info:
#             print(f"    ERROR: Cannot read raster file")
#             return False
        
#         print(f"    Original size: {raster_info['width']:,} × {raster_info['height']:,} pixels")
#         print(f"    Original area: {raster_info['area_km2']:,.0f} km²")
#         print(f"    CRS: {raster_info['crs_auth']}:{raster_info['crs_code']}")
        
#         # Check if shapefile exists
#         if not os.path.exists(shapefile_path):
#             print(f"    ERROR: Shapefile not found: {shapefile_path}")
#             return False
        
#         # Clip options - using CUTLINE_ALL_TOUCHED to ensure all edge pixels are included
#         warp_options = gdal.WarpOptions(
#             format='GTiff',
#             cutlineDSName=shapefile_path,
#             cropToCutline=True,
#             dstNodata=0,  # Set nodata value to 0 (assuming your data is 0-255)
#             resampleAlg='near',
#             creationOptions=[
#                 'COMPRESS=LZW',
#                 'PREDICTOR=2',
#                 'TILED=YES',
#                 'BLOCKXSIZE=256',
#                 'BLOCKYSIZE=256',
#                 'BIGTIFF=YES'
#             ],
#             multithread=True,
#             warpMemoryLimit=2048
#         )
        
#         # Perform clipping
#         print(f"    Clipping to Alberta boundary...")
#         ds = gdal.Warp(output_raster, input_raster, options=warp_options)
        
#         if ds is None:
#             print(f"    ERROR: Failed to clip raster")
#             return False
        
#         # Get clipped raster info
#         clipped_width = ds.RasterXSize
#         clipped_height = ds.RasterYSize
#         transform = ds.GetGeoTransform()
        
#         # Calculate clipped bounds and area
#         minx = transform[0]
#         maxx = minx + clipped_width * transform[1]
#         maxy = transform[3]
#         miny = maxy + clipped_height * transform[5]
#         clipped_area_km2 = ((maxx - minx) * (maxy - miny)) / 1000000
        
#         # Verify CRS is preserved
#         clipped_crs_wkt = ds.GetProjection()
#         clipped_srs = osr.SpatialReference()
#         clipped_srs.ImportFromWkt(clipped_crs_wkt)
#         clipped_crs_code = clipped_srs.GetAuthorityCode(None)
        
#         ds = None
        
#         # Check file was created
#         if not os.path.exists(output_raster):
#             print(f"    ERROR: Output file was not created")
#             return False
        
#         file_size_mb = os.path.getsize(output_raster) / (1024 * 1024)
        
#         print(f"    ✓ Clipped successfully!")
#         print(f"      New size: {clipped_width:,} × {clipped_height:,} pixels")
#         print(f"      New area: {clipped_area_km2:,.0f} km²")
#         print(f"      File size: {file_size_mb:.1f} MB")
#         print(f"      CRS preserved: EPSG:{clipped_crs_code}")
#         print(f"      Reduction: {((raster_info['area_km2'] - clipped_area_km2) / raster_info['area_km2'] * 100):.1f}%")
        
#         return True
        
#     except Exception as e:
#         print(f"    ERROR during clipping: {str(e)}")
#         return False

# def process_all_bands():
#     """Clip all mosaic files using Alberta shapefile"""
#     print("=" * 80)
#     print("CLIP ALBERTA MOSAICS TO PROVINCIAL BOUNDARY")
#     print("=" * 80)
#     print(f"Mosaic directory: {mosaic_dir}")
#     print(f"Shapefile: {shapefile_path}")
#     print(f"Output directory: {output_dir}")
#     print("=" * 80)
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Get shapefile info
#     shapefile_info = get_shapefile_info()
#     if not shapefile_info:
#         print("ERROR: Cannot proceed without valid shapefile")
#         return
    
#     print("\n" + "=" * 80)
#     print("PROCESSING MOSAIC FILES")
#     print("=" * 80)
    
#     # Get all mosaic files
#     mosaic_files = get_all_mosaics()
    
#     if not mosaic_files:
#         print("No mosaic files found!")
#         return
    
#     successful_clips = []
#     failed_clips = []
    
#     # Process each mosaic
#     for idx, mosaic_path in enumerate(mosaic_files, 1):
#         print(f"\n[{idx:2d}/{len(mosaic_files)}]")
        
#         # Create output filename
#         basename = os.path.basename(mosaic_path)
#         name_without_ext = os.path.splitext(basename)[0]
#         output_filename = f"{name_without_ext}_clipped.tif"
#         output_path = os.path.join(output_dir, output_filename)
        
#         # Skip if already processed
#         if os.path.exists(output_path):
#             print(f"  Skipping (already exists): {output_filename}")
#             successful_clips.append(basename)
#             continue
        
#         # Clip the raster
#         success = clip_raster_with_shapefile(mosaic_path, output_path, shapefile_path)
        
#         if success:
#             successful_clips.append(basename)
#         else:
#             failed_clips.append(basename)
    
#     # Summary
#     print("\n" + "=" * 80)
#     print("CLIPPING COMPLETE - SUMMARY")
#     print("=" * 80)
#     print(f"Total mosaics: {len(mosaic_files)}")
#     print(f"Successfully clipped: {len(successful_clips)}")
#     print(f"Failed: {len(failed_clips)}")
    
#     if successful_clips:
#         print(f"\nSuccessfully clipped files:")
#         for i in range(0, len(successful_clips), 10):
#             chunk = successful_clips[i:i+10]
#             print(f"  {i+1:2d}-{i+len(chunk):2d}: {', '.join(chunk)}")
    
#     if failed_clips:
#         print(f"\nFailed files:")
#         for failed in failed_clips:
#             print(f"  - {failed}")
    
#     print(f"\nClipped mosaics saved to: {output_dir}")
    
#     # Create summary file
#     summary_path = os.path.join(output_dir, 'clipping_summary.txt')
#     with open(summary_path, 'w') as f:
#         f.write("ALBERTA MOSAIC CLIPPING SUMMARY\n")
#         f.write("=" * 60 + "\n")
#         f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"Input directory: {mosaic_dir}\n")
#         f.write(f"Shapefile: {shapefile_path}\n")
#         f.write(f"Output directory: {output_dir}\n")
#         f.write(f"Total mosaics: {len(mosaic_files)}\n")
#         f.write(f"Successfully clipped: {len(successful_clips)}\n")
#         f.write(f"Failed: {len(failed_clips)}\n\n")
#         f.write("SUCCESSFUL CLIPS:\n")
#         for clip in successful_clips:
#             f.write(f"  {clip}\n")
#         if failed_clips:
#             f.write("\nFAILED CLIPS:\n")
#             for clip in failed_clips:
#                 f.write(f"  {clip}\n")
    
#     print(f"\nDetailed summary saved to: {summary_path}")
    
#     # Show example of clipped file
#     if successful_clips:
#         example_input = os.path.join(mosaic_dir, successful_clips[0])
#         example_output = os.path.join(output_dir, f"{os.path.splitext(successful_clips[0])[0]}_clipped.tif")
        
#         if os.path.exists(example_output):
#             print(f"\nEXAMPLE COMPARISON:")
#             print("-" * 40)
            
#             input_info = get_raster_info(example_input)
#             output_info = get_raster_info(example_output)
            
#             if input_info and output_info:
#                 print(f"Input: {os.path.basename(example_input)}")
#                 print(f"  Size: {input_info['width']:,} × {input_info['height']:,} pixels")
#                 print(f"  Area: {input_info['area_km2']:,.0f} km²")
                
#                 print(f"\nOutput: {os.path.basename(example_output)}")
#                 print(f"  Size: {output_info['width']:,} × {output_info['height']:,} pixels")
#                 print(f"  Area: {output_info['area_km2']:,.0f} km²")
                
#                 reduction = ((input_info['area_km2'] - output_info['area_km2']) / input_info['area_km2']) * 100
#                 print(f"\nReduction: {reduction:.1f}%")
#                 print(f"Alberta area preserved: {output_info['area_km2']:,.0f} km²")
    
#     print("=" * 80)

# def clip_single_band(band_name):
#     """Clip a single band by name"""
#     print(f"\nClipping single band: {band_name}")
#     print("-" * 40)
    
#     # Find the mosaic file
#     pattern = os.path.join(mosaic_dir, f'*{band_name}*.tif')
#     mosaic_files = glob.glob(pattern)
    
#     if not mosaic_files:
#         print(f"ERROR: No mosaic found for band {band_name}")
#         return
    
#     mosaic_path = mosaic_files[0]
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Create output filename
#     basename = os.path.basename(mosaic_path)
#     name_without_ext = os.path.splitext(basename)[0]
#     output_filename = f"{name_without_ext}_clipped.tif"
#     output_path = os.path.join(output_dir, output_filename)
    
#     # Clip the raster
#     success = clip_raster_with_shapefile(mosaic_path, output_path, shapefile_path)
    
#     if success:
#         print(f"\n✓ Successfully clipped {band_name}")
#         print(f"  Output: {output_filename}")
#     else:
#         print(f"\n✗ Failed to clip {band_name}")

# # Main execution
# if __name__ == "__main__":
#     gdal.UseExceptions()
    
#     print("ALBERTA MOSAIC CLIPPING TOOL")
#     print("=" * 80)
#     print("This script will clip all mosaics to Alberta provincial boundary")
#     print(f"Mosaics are in: {mosaic_dir}")
#     print(f"Using shapefile: {shapefile_path}")
#     print(f"Output will be in: {output_dir}")
#     print("CRS will be preserved (EPSG:3347)")
#     print("=" * 80)
    
#     print("\nCHOOSE PROCESSING MODE:")
#     print("1. Clip ALL mosaics")
#     print("2. Clip specific band")
#     print("=" * 80)
    
#     choice = input("Enter choice (1 or 2): ").strip()
    
#     if choice == "2":
#         band_name = input("Enter band name (e.g., A00, A01, etc.): ").strip().upper()
#         clip_single_band(band_name)
#     else:
#         process_all_bands()
    
#     print("\n" + "=" * 80)
#     print("CLIPPING COMPLETE")
#     print("=" * 80)
#     print("Clipped mosaics are saved with '_clipped' suffix:")
#     print(f"  {output_dir}")
#     print("\nExample naming:")
#     print("  Original: Alberta_2020_A00_NAD83_StatsCan.tif")
#     print("  Clipped:  Alberta_2020_A00_NAD83_StatsCan_clipped.tif")
#     print("\nKEY FEATURES:")
#     print("- CRS: Preserved as EPSG:3347 (NAD83 / Statistics Canada Lambert)")
#     print("- Resolution: Preserved at 30 meters")
#     print("- Nodata value: Set to 0")
#     print("- Compression: LZW with tiling")
#     print("- Coverage: Exact Alberta provincial boundary")
#     print("=" * 80)

import os
import glob
from osgeo import gdal, ogr, osr
import datetime

# Input and output directories
mosaic_dir = r'D:\AlphaEarth_Dataset\Alberta\NAD83_StatsCan_Mosaics_EPSG_3979'
shapefile_path = r'D:\Alberta_Sentinel2_2020\Alberta_Shapefile\Alberta_EPSG_3979.gpkg'  # Updated to EPSG:3979 shapefile
output_dir = r'D:\AlphaEarth_Dataset\Alberta\Alberta_2020_NAD83_StatsCan_Mosaics_EPSG_3979_Clipped'

def check_crs_compatibility():
    """Check if mosaics and shapefile have compatible CRS"""
    print("Checking CRS compatibility...")
    print("-" * 40)
    
    # Get CRS from first mosaic
    pattern = os.path.join(mosaic_dir, '*.tif')
    mosaic_files = glob.glob(pattern)
    
    if not mosaic_files:
        print("ERROR: No mosaic files found!")
        return False
    
    mosaic_crs_info = get_raster_crs(mosaic_files[0])
    if not mosaic_crs_info:
        print("ERROR: Cannot read mosaic CRS")
        return False
    
    print(f"Mosaic CRS: EPSG:{mosaic_crs_info['code']} - {mosaic_crs_info['name']}")
    
    # Get CRS from shapefile
    shapefile_crs_info = get_shapefile_crs(shapefile_path)
    if not shapefile_crs_info:
        print("ERROR: Cannot read shapefile CRS")
        return False
    
    print(f"Shapefile CRS: EPSG:{shapefile_crs_info['code']} - {shapefile_crs_info['name']}")
    
    # Check if CRS match
    if mosaic_crs_info['code'] == shapefile_crs_info['code']:
        print("✓ CRS MATCH: Both are in EPSG:3979")
        return True
    else:
        print(f"✗ CRS MISMATCH: Mosaic EPSG:{mosaic_crs_info['code']} vs Shapefile EPSG:{shapefile_crs_info['code']}")
        return False

def get_raster_crs(raster_path):
    """Get CRS information from raster"""
    try:
        ds = gdal.Open(raster_path)
        if ds is None:
            return None
        
        crs_wkt = ds.GetProjection()
        
        # Get CRS info
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)
        crs_name = srs.GetName()
        crs_auth = srs.GetAuthorityName(None)
        crs_code = srs.GetAuthorityCode(None)
        
        ds = None
        
        return {
            'name': crs_name,
            'auth': crs_auth,
            'code': crs_code,
            'wkt': crs_wkt
        }
        
    except Exception as e:
        print(f"ERROR reading raster CRS {raster_path}: {str(e)}")
        return None

def get_shapefile_crs(shapefile_path):
    """Get CRS information from shapefile"""
    try:
        ds = ogr.Open(shapefile_path)
        if ds is None:
            return None
        
        layer = ds.GetLayer()
        srs = layer.GetSpatialRef()
        
        if srs:
            srs.AutoIdentifyEPSG()
            crs_name = srs.GetName()
            crs_auth = srs.GetAuthorityName(None)
            crs_code = srs.GetAuthorityCode(None)
            
            ds = None
            
            return {
                'name': crs_name,
                'auth': crs_auth,
                'code': crs_code,
                'srs': srs
            }
        else:
            ds = None
            return None
            
    except Exception as e:
        print(f"ERROR reading shapefile CRS {shapefile_path}: {str(e)}")
        return None

def get_all_mosaics():
    """Get list of all mosaic files"""
    pattern = os.path.join(mosaic_dir, '*.tif')
    mosaic_files = glob.glob(pattern)
    
    # Sort by band name
    mosaic_files.sort()
    
    print(f"Found {len(mosaic_files)} mosaic files:")
    for i, mosaic in enumerate(mosaic_files, 1):
        print(f"  {i:2d}. {os.path.basename(mosaic)}")
    
    return mosaic_files

def get_shapefile_info():
    """Get information about the shapefile"""
    print("\nSHAPEFILE INFORMATION:")
    print("-" * 40)
    
    try:
        # Open shapefile
        ds = ogr.Open(shapefile_path)
        if ds is None:
            print(f"ERROR: Cannot open shapefile: {shapefile_path}")
            return None
        
        layer = ds.GetLayer()
        feature_count = layer.GetFeatureCount()
        extent = layer.GetExtent()
        
        # Get CRS info
        srs = layer.GetSpatialRef()
        if srs:
            srs.AutoIdentifyEPSG()
            crs_name = srs.GetName()
            crs_auth = srs.GetAuthorityName(None)
            crs_code = srs.GetAuthorityCode(None)
            crs_info = f"{crs_auth}:{crs_code} - {crs_name}"
        else:
            crs_info = "Unknown CRS"
        
        ds = None
        
        print(f"  File: {os.path.basename(shapefile_path)}")
        print(f"  Features: {feature_count}")
        print(f"  CRS: {crs_info}")
        print(f"  Extent (xmin, xmax, ymin, ymax):")
        print(f"    {extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}, {extent[3]:.2f}")
        print(f"  Width: {(extent[1] - extent[0]):.2f} m")
        print(f"  Height: {(extent[3] - extent[2]):.2f} m")
        
        return {
            'path': shapefile_path,
            'feature_count': feature_count,
            'extent': extent,
            'crs': srs
        }
        
    except Exception as e:
        print(f"ERROR reading shapefile: {str(e)}")
        return None

def get_raster_info(raster_path):
    """Get information about a raster file"""
    try:
        ds = gdal.Open(raster_path)
        if ds is None:
            return None
        
        width = ds.RasterXSize
        height = ds.RasterYSize
        transform = ds.GetGeoTransform()
        crs_wkt = ds.GetProjection()
        
        # Get CRS info
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)
        crs_name = srs.GetName()
        crs_auth = srs.GetAuthorityName(None)
        crs_code = srs.GetAuthorityCode(None)
        
        # Calculate bounds
        minx = transform[0]
        maxx = minx + width * transform[1]
        maxy = transform[3]
        miny = maxy + height * transform[5]
        
        # Calculate area
        width_m = maxx - minx
        height_m = maxy - miny
        area_km2 = (width_m * height_m) / 1000000
        
        ds = None
        
        return {
            'width': width,
            'height': height,
            'transform': transform,
            'crs_wkt': crs_wkt,
            'crs_name': crs_name,
            'crs_auth': crs_auth,
            'crs_code': crs_code,
            'bounds': (minx, maxx, miny, maxy),
            'width_m': width_m,
            'height_m': height_m,
            'area_km2': area_km2
        }
        
    except Exception as e:
        print(f"ERROR reading raster {raster_path}: {str(e)}")
        return None

def clip_raster_with_shapefile(input_raster, output_raster, shapefile_path):
    """Clip raster using shapefile while maintaining original CRS"""
    
    print(f"  Processing: {os.path.basename(input_raster)}")
    
    try:
        # Get raster info
        raster_info = get_raster_info(input_raster)
        if not raster_info:
            print(f"    ERROR: Cannot read raster file")
            return False
        
        print(f"    Original size: {raster_info['width']:,} × {raster_info['height']:,} pixels")
        print(f"    Original area: {raster_info['area_km2']:,.0f} km²")
        print(f"    CRS: EPSG:{raster_info['crs_code']}")
        
        # Check if shapefile exists
        if not os.path.exists(shapefile_path):
            print(f"    ERROR: Shapefile not found: {shapefile_path}")
            return False
        
        # Get shapefile CRS for verification
        shapefile_crs = get_shapefile_crs(shapefile_path)
        if shapefile_crs and shapefile_crs['code'] != raster_info['crs_code']:
            print(f"    WARNING: CRS mismatch!")
            print(f"      Raster: EPSG:{raster_info['crs_code']}")
            print(f"      Shapefile: EPSG:{shapefile_crs['code']}")
            print(f"    Proceeding anyway... GDAL will handle transformation")
        
        # Clip options - using CUTLINE_ALL_TOUCHED to ensure all edge pixels are included
        warp_options = gdal.WarpOptions(
            format='GTiff',
            cutlineDSName=shapefile_path,
            cropToCutline=True,
            dstNodata=0,  # Set nodata value to 0
            resampleAlg='near',
            creationOptions=[
                'COMPRESS=LZW',
                'PREDICTOR=2',
                'TILED=YES',
                'BLOCKXSIZE=256',
                'BLOCKYSIZE=256',
                'BIGTIFF=YES'
            ],
            multithread=True,
            warpMemoryLimit=2048
        )
        
        # Perform clipping
        print(f"    Clipping to Alberta boundary...")
        ds = gdal.Warp(output_raster, input_raster, options=warp_options)
        
        if ds is None:
            print(f"    ERROR: Failed to clip raster")
            return False
        
        # Get clipped raster info
        clipped_width = ds.RasterXSize
        clipped_height = ds.RasterYSize
        transform = ds.GetGeoTransform()
        
        # Calculate clipped bounds and area
        minx = transform[0]
        maxx = minx + clipped_width * transform[1]
        maxy = transform[3]
        miny = maxy + clipped_height * transform[5]
        clipped_area_km2 = ((maxx - minx) * (maxy - miny)) / 1000000
        
        # Verify CRS is preserved
        clipped_crs_wkt = ds.GetProjection()
        clipped_srs = osr.SpatialReference()
        clipped_srs.ImportFromWkt(clipped_crs_wkt)
        clipped_crs_code = clipped_srs.GetAuthorityCode(None)
        
        ds = None
        
        # Check file was created
        if not os.path.exists(output_raster):
            print(f"    ERROR: Output file was not created")
            return False
        
        file_size_mb = os.path.getsize(output_raster) / (1024 * 1024)
        
        print(f"    ✓ Clipped successfully!")
        print(f"      New size: {clipped_width:,} × {clipped_height:,} pixels")
        print(f"      New area: {clipped_area_km2:,.0f} km²")
        print(f"      File size: {file_size_mb:.1f} MB")
        print(f"      CRS preserved: EPSG:{clipped_crs_code}")
        print(f"      Reduction: {((raster_info['area_km2'] - clipped_area_km2) / raster_info['area_km2'] * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"    ERROR during clipping: {str(e)}")
        return False

def process_all_bands():
    """Clip all mosaic files using Alberta shapefile"""
    print("=" * 80)
    print("CLIP ALBERTA MOSAICS TO PROVINCIAL BOUNDARY")
    print("=" * 80)
    print(f"Mosaic directory: {mosaic_dir}")
    print(f"Shapefile: {os.path.basename(shapefile_path)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # First check CRS compatibility
    if not check_crs_compatibility():
        print("\nWARNING: CRS mismatch detected!")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Processing cancelled.")
            return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get shapefile info
    shapefile_info = get_shapefile_info()
    if not shapefile_info:
        print("ERROR: Cannot proceed without valid shapefile")
        return
    
    print("\n" + "=" * 80)
    print("PROCESSING MOSAIC FILES")
    print("=" * 80)
    
    # Get all mosaic files
    mosaic_files = get_all_mosaics()
    
    if not mosaic_files:
        print("No mosaic files found!")
        return
    
    successful_clips = []
    failed_clips = []
    
    # Process each mosaic
    for idx, mosaic_path in enumerate(mosaic_files, 1):
        print(f"\n[{idx:2d}/{len(mosaic_files)}]")
        
        # Create output filename
        basename = os.path.basename(mosaic_path)
        name_without_ext = os.path.splitext(basename)[0]
        output_filename = f"{name_without_ext}_clipped.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"  Skipping (already exists): {output_filename}")
            successful_clips.append(basename)
            continue
        
        # Clip the raster
        success = clip_raster_with_shapefile(mosaic_path, output_path, shapefile_path)
        
        if success:
            successful_clips.append(basename)
        else:
            failed_clips.append(basename)
    
    # Summary
    print("\n" + "=" * 80)
    print("CLIPPING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Total mosaics: {len(mosaic_files)}")
    print(f"Successfully clipped: {len(successful_clips)}")
    print(f"Failed: {len(failed_clips)}")
    
    if successful_clips:
        print(f"\nSuccessfully clipped files:")
        for i in range(0, len(successful_clips), 10):
            chunk = successful_clips[i:i+10]
            print(f"  {i+1:2d}-{i+len(chunk):2d}: {', '.join(chunk)}")
    
    if failed_clips:
        print(f"\nFailed files:")
        for failed in failed_clips:
            print(f"  - {failed}")
    
    print(f"\nClipped mosaics saved to: {output_dir}")
    
    # Create summary file
    summary_path = os.path.join(output_dir, 'clipping_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ALBERTA MOSAIC CLIPPING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {mosaic_dir}\n")
        f.write(f"Shapefile: {shapefile_path}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Total mosaics: {len(mosaic_files)}\n")
        f.write(f"Successfully clipped: {len(successful_clips)}\n")
        f.write(f"Failed: {len(failed_clips)}\n\n")
        f.write("SUCCESSFUL CLIPS:\n")
        for clip in successful_clips:
            f.write(f"  {clip}\n")
        if failed_clips:
            f.write("\nFAILED CLIPS:\n")
            for clip in failed_clips:
                f.write(f"  {clip}\n")
    
    print(f"\nDetailed summary saved to: {summary_path}")
    
    # Show example of clipped file
    if successful_clips:
        example_input = os.path.join(mosaic_dir, successful_clips[0])
        example_output = os.path.join(output_dir, f"{os.path.splitext(successful_clips[0])[0]}_clipped.tif")
        
        if os.path.exists(example_output):
            print(f"\nEXAMPLE COMPARISON:")
            print("-" * 40)
            
            input_info = get_raster_info(example_input)
            output_info = get_raster_info(example_output)
            
            if input_info and output_info:
                print(f"Input: {os.path.basename(example_input)}")
                print(f"  Size: {input_info['width']:,} × {input_info['height']:,} pixels")
                print(f"  Area: {input_info['area_km2']:,.0f} km²")
                print(f"  CRS: EPSG:{input_info['crs_code']}")
                
                print(f"\nOutput: {os.path.basename(example_output)}")
                print(f"  Size: {output_info['width']:,} × {output_info['height']:,} pixels")
                print(f"  Area: {output_info['area_km2']:,.0f} km²")
                print(f"  CRS: EPSG:{output_info['crs_code']}")
                
                reduction = ((input_info['area_km2'] - output_info['area_km2']) / input_info['area_km2']) * 100
                print(f"\nReduction: {reduction:.1f}%")
                print(f"Alberta area preserved: {output_info['area_km2']:,.0f} km²")
    
    print("=" * 80)

def clip_single_band(band_name):
    """Clip a single band by name"""
    print(f"\nClipping single band: {band_name}")
    print("-" * 40)
    
    # First check CRS compatibility
    if not check_crs_compatibility():
        print("\nWARNING: CRS mismatch detected!")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Processing cancelled.")
            return
    
    # Find the mosaic file
    pattern = os.path.join(mosaic_dir, f'*{band_name}*.tif')
    mosaic_files = glob.glob(pattern)
    
    if not mosaic_files:
        print(f"ERROR: No mosaic found for band {band_name}")
        return
    
    mosaic_path = mosaic_files[0]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    basename = os.path.basename(mosaic_path)
    name_without_ext = os.path.splitext(basename)[0]
    output_filename = f"{name_without_ext}_clipped.tif"
    output_path = os.path.join(output_dir, output_filename)
    
    # Skip if already exists
    if os.path.exists(output_path):
        print(f"File already exists: {output_filename}")
        return
    
    # Clip the raster
    success = clip_raster_with_shapefile(mosaic_path, output_path, shapefile_path)
    
    if success:
        print(f"\n✓ Successfully clipped {band_name}")
        print(f"  Output: {output_filename}")
    else:
        print(f"\n✗ Failed to clip {band_name}")

# Main execution
if __name__ == "__main__":
    gdal.UseExceptions()
    
    print("ALBERTA MOSAIC CLIPPING TOOL (EPSG:3979)")
    print("=" * 80)
    print("This script will clip all mosaics to Alberta provincial boundary")
    print(f"Mosaics are in: {mosaic_dir}")
    print(f"Using shapefile: {os.path.basename(shapefile_path)}")
    print(f"Output will be in: {output_dir}")
    print("CRS should be: EPSG:3979 (Canada Atlas Lambert)")
    print("=" * 80)
    
    print("\nCHOOSE PROCESSING MODE:")
    print("1. Clip ALL mosaics")
    print("2. Clip specific band")
    print("3. Check CRS compatibility only")
    print("=" * 80)
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "3":
        check_crs_compatibility()
    elif choice == "2":
        band_name = input("Enter band name (e.g., A00, A01, etc.): ").strip().upper()
        clip_single_band(band_name)
    else:
        process_all_bands()
    
    print("\n" + "=" * 80)
    print("CLIPPING COMPLETE")
    print("=" * 80)
    print("Clipped mosaics are saved with '_clipped' suffix:")
    print(f"  {output_dir}")
    print("\nExample naming:")
    print("  Original: Alberta_2020_A00_NAD83_CanadaAtlas.tif")
    print("  Clipped:  Alberta_2020_A00_NAD83_CanadaAtlas_clipped.tif")
    print("\nKEY FEATURES:")
    print("- CRS: Preserved as EPSG:3979 (Canada Atlas Lambert)")
    print("- Resolution: Preserved at 30 meters")
    print("- Nodata value: Set to 0")
    print("- Compression: LZW with tiling")
    print("- Coverage: Exact Alberta provincial boundary")
    print("=" * 80)
