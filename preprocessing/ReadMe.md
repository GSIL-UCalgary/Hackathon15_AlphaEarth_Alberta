1- Run AlphaEarth_Alberta_2020_Uint8.js code in Google Earth Engine Code Platfrom to downlaod the tiled data of each band.
2- Since the region of interest was divided into several tiles, they must be stack together to generate a union dataset of Alberta by running merging_tiles.py.
3- After merging all tiles, clip_images.py must be applied to remove data out of the ALberta region.
