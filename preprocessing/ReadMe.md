## Workflow

Follow the steps below to generate the datasets for Alberta (2020):

1. **Export tiled data**
   - Run `.js` code of each dataset in the **Google Earth Engine Code Editor**.
   - This script downloads **tiled GeoTIFFs** for each band of datasets over Alberta.

2. **Merge tiles**
   - Since the region of interest is divided into multiple tiles, the tiles must be merged.
   - Run `merging_tiles_..._01.py` to stack and merge all tiles into a single dataset covering Alberta.

3. **Clip to Alberta boundary**
   - After merging all tiles, run `clip_merged_...._02.py` to remove data outside the **Alberta boundary**.
4. **Stack bands**
   - Finaly, run `stack_clipped_...._03.py` to stack all abnds into one image
### Notes
- Ensure all export tasks in Google Earth Engine have completed before running the Python scripts.
- All tiles should have the same CRS, resolution, and naming convention.
- Put all the bands in the dataset folder.
- Save the tiles of each band in a folder named after that band such as B2 for Sentinel-2 Blue band and SR_B2 for Landsat-8 Blue band.
  
