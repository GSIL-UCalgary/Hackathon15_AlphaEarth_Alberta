## Workflow

Follow the steps below to generate the AlphaEarth dataset for Alberta (2020):

1. **Export tiled AlphaEarth data**
   - Run `AlphaEarth_Alberta_2020_Uint8.js` in the **Google Earth Engine Code Editor**.
   - This script downloads **tiled GeoTIFFs** for each AlphaEarth embedding band over Alberta.

2. **Merge tiles**
   - Since the region of interest is divided into multiple tiles, the tiles must be merged.
   - Run `merging_tiles.py` to stack and merge all tiles into a single dataset covering Alberta.

3. **Clip to Alberta boundary**
   - After merging all tiles, run `clip_images.py` to remove data outside the **Alberta boundary**.

### Notes
- Ensure all export tasks in Google Earth Engine have completed before running the Python scripts.
- All tiles should have the same CRS, resolution, and naming convention.
