#!/usr/bin/env python3
"""
Extract PCA components from a 64-band GeoTIFF
and save them as a compressed GeoTIFF while preserving
geospatial information and making it QGIS-friendly.

This version uses IncrementalPCA and block-wise reading
on all pixels, ignoring valid/no-data masking.
"""

import numpy as np
from osgeo import gdal
from sklearn.decomposition import IncrementalPCA
import argparse
from tqdm import tqdm

# ----------------------------
# Load raster info
# ----------------------------
def get_raster_info(tif_path):
    ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open {tif_path}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    bands = ds.RasterCount
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    print(f"nodata value: {nodata}")
    ds = None
    return width, height, bands, geotransform, projection, nodata

# ----------------------------
# Read raster block
# ----------------------------
def read_block(ds, xoff, yoff, xsize, ysize):
    """
    Read a block of the raster and return shape (ysize, xsize, bands)
    """
    data = ds.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32)
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)  # (bands, y, x) -> (y, x, bands)
    else:
        data = data[:, :, np.newaxis]
    return data

# ----------------------------
# Compute IncrementalPCA on all pixels
# ----------------------------
def compute_ipca_all_pixels(input_file, n_components=3, block_height=512):
    width, height, bands, gt, proj, nodata = get_raster_info(input_file)

    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    ipca = IncrementalPCA(n_components=n_components)

    print("Fitting IncrementalPCA on raster (all pixels)...")
    for y in tqdm(range(0, height, block_height)):
        rows = min(block_height, height - y)
        block = read_block(ds, 0, y, width, rows)

        # Normalize 0-255 → -1 to 1
        pixels = (block / 127.5) - 1.0  # shape: (rows, width, bands)
        pixels_2d = pixels.reshape(-1, bands)  # (rows*width, bands)

        ipca.partial_fit(pixels_2d)

    ds = None
    return ipca, gt, proj, nodata, width, height, bands

# ----------------------------
# Transform and save PCA raster
# ----------------------------
def transform_and_save_all_pixels(input_file, output_file, ipca, gt, proj, nodata,
                                  width, height, block_height=512):
    n_components = ipca.n_components_
    driver = gdal.GetDriverByName("GTiff")

    # Create temporary in-memory dataset
    temp_ds = driver.Create("/vsimem/temp.tif", width, height, n_components, gdal.GDT_Float32)
    temp_ds.SetGeoTransform(gt)
    temp_ds.SetProjection(proj)

    ds = gdal.Open(input_file, gdal.GA_ReadOnly)

    print("Transforming raster and writing PCA bands...")
    for y in tqdm(range(0, height, block_height)):
        rows = min(block_height, height - y)
        block = read_block(ds, 0, y, width, rows)

        # Normalize
        pixels = (block / 127.5) - 1.0
        pixels_2d = pixels.reshape(-1, block.shape[2])
        reduced = ipca.transform(pixels_2d)
        reduced_block = reduced.reshape(rows, width, n_components)

        # Write each PCA component band
        for b in range(n_components):
            band = temp_ds.GetRasterBand(b + 1)
            band.WriteArray(reduced_block[:, :, b], 0, y)
            if nodata is not None:
                band.SetNoDataValue(nodata)

    ds = None

    # Translate to compressed, tiled GeoTIFF (QGIS-friendly)
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
    gdal.Translate(output_file, temp_ds, options=translate_options)
    temp_ds = None
    gdal.Unlink("/vsimem/temp.tif")
    print(f"Saved PCA GeoTIFF → {output_file}")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  help="Input 64-band GeoTIFF")
    parser.add_argument("--output", help="Output PCA GeoTIFF")
    parser.add_argument("--components", type=int, default=3, help="Number of PCA components")
    parser.add_argument("--block", type=int, default=512, help="Block height for processing")
    args = parser.parse_args()

    # Hardcoded input/output for testing
    if args.input is None:
        args.input = './preprocessing/AlphaEarth_Dataset/Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979_Clipped_Stack/Alberta_2020_AlphaEarth_Stacked_64Bands.tif'
    if args.output is None:
        args.output = './preprocessing/AlphaEarth_Dataset/Alberta_2020_NAD83_StatsCan_AlphaEarth_30m_Mosaics_EPSG_3979_Clipped_Stack/Alberta_2020_AlphaEarth_PCA.tif'

    ipca, gt, proj, nodata, width, height, bands = compute_ipca_all_pixels(
        args.input, n_components=args.components, block_height=args.block
    )

    transform_and_save_all_pixels(
        args.input, args.output, ipca, gt, proj, nodata, width, height, block_height=args.block
    )

# ----------------------------
if __name__ == "__main__":
    main()
