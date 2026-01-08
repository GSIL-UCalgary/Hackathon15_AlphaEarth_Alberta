// ====================================================================
// TILED EXPORT FOR SENTINEL-2 SR (2020, 30m, EPSG:3979)
// ====================================================================

// 1. Define Alberta boundary
var canada = ee.FeatureCollection("FAO/GAUL/2015/level1");
var alberta = canada.filter(ee.Filter.eq('ADM1_NAME', 'Alberta')).geometry();

// 2. Define year 2020
var targetYear = 2020;
var startDate = ee.Date.fromYMD(targetYear, 1, 1);
var endDate   = ee.Date.fromYMD(targetYear + 1, 1, 1);

// 3. Sentinel-2 bands to export (common ML-ready set)
var S2_BANDS = [
  // 'B2',  // Blue
  // 'B3',  // Green
  // 'B4',  // Red
  // 'B5',  // RedEdge1
  // 'B6',  // RedEdge2
  // 'B7',  // RedEdge3
  // 'B8',  // NIR
  // 'B8A', // RedEdge4
  // 'B11', // SWIR1
  'B12'  // SWIR2
];

// 4. Cloud masking function (S2 SR)
function maskS2sr(image) {
  var scl = image.select('SCL');

  // Valid pixels
  var mask = scl
    .neq(3)  // cloud shadow
    .and(scl.neq(7))  // unclassified
    .and(scl.neq(8))  // cloud medium
    .and(scl.neq(9))  // cloud high
    .and(scl.neq(10)) // cirrus
    .and(scl.neq(11)); // snow

  return image.updateMask(mask);
}

// 5. Function to create a grid of tiles over Alberta
function createTiles(geometry, numTiles) {
  var bounds = geometry.bounds();
  var coords = ee.List(bounds.coordinates().get(0));
  
  var xCoords = coords.map(function(point) {
    return ee.Number(ee.List(point).get(0));
  });
  
  var yCoords = coords.map(function(point) {
    return ee.Number(ee.List(point).get(1));
  });
  
  var xMin = xCoords.reduce(ee.Reducer.min());
  var xMax = xCoords.reduce(ee.Reducer.max());
  var yMin = yCoords.reduce(ee.Reducer.min());
  var yMax = yCoords.reduce(ee.Reducer.max());
  
  var xMinValue = xMin.getInfo();
  var xMaxValue = xMax.getInfo();
  var yMinValue = yMin.getInfo();
  var yMaxValue = yMax.getInfo();
  
  var gridSize = Math.sqrt(numTiles);
  var xStep = (xMaxValue - xMinValue) / gridSize;
  var yStep = (yMaxValue - yMinValue) / gridSize;
  
  var tiles = [];
  
  for (var i = 0; i < gridSize; i++) {
    for (var j = 0; j < gridSize; j++) {
      var x1 = xMinValue + xStep * i;
      var x2 = xMinValue + xStep * (i + 1);
      var y1 = yMinValue + yStep * j;
      var y2 = yMinValue + yStep * (j + 1);
      
      var tileBounds = ee.Geometry.Rectangle([x1, y1, x2, y2], null, false);
      var tile = tileBounds.intersection(geometry, 100);
      
      // Skip empty tiles
      var area = tile.area().getInfo();
      if (area > 0) {
        tiles.push({
          bounds: tileBounds,
          tile: tile,
          id: i * gridSize + j,
          row: i,
          col: j
        });
      }
    }
  }
  
  print('Created ' + tiles.length + ' non-empty tiles out of ' + numTiles + ' total');
  return tiles;
}

// 6. Create tiles (adjust number based on your needs)
var numTiles = 25; // 5x5 grid
var tiles = createTiles(alberta, numTiles);

// 7. Process and export one tile
function processTile(tileInfo, tileNumber) {
  
  // Assume S2_BANDS has only ONE band uncommented
  var bandName = S2_BANDS[0]; // take the first (and only) band

  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(tileInfo.tile)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2sr)
    .select([bandName]);

  // Annual composite for this band
  var annual = collection.median()
    .clip(tileInfo.tile)
    .rename(bandName); // rename band in the image

  // Normalization: Divide by 10000 to get surface reflectance (0-1)
  var normalized = annual.divide(10000);

  // Clip to 0-1 range (in case of values outside expected range)
  var clipped = normalized.clamp(0, 1);

  // Convert to Uint8 (0-255)
  var uint8Image = clipped.multiply(255).uint8();

  Export.image.toDrive({
    image: uint8Image,
    description: 'Alberta_2020_S2_' + bandName + '_tile_' + tileNumber,
    fileNamePrefix: 'Alberta_2020_S2_' + bandName + '_tile_' +
                    tileNumber + '_R' + tileInfo.row + 'C' + tileInfo.col,
    region: tileInfo.bounds,
    scale: 30,
    crs: 'EPSG:3979',
    maxPixels: 1e13,
    folder: 'Sentinel2_2020_3979',
    fileFormat: 'GeoTIFF',
    formatOptions: {
      cloudOptimized: true
    }
  });
}

// 8. Export all tiles
for (var i = 0; i < tiles.length; i++) {
  processTile(tiles[i], i);
}

// 9. Visualization
Map.centerObject(alberta, 6);
Map.addLayer(alberta, {color: 'red'}, 'Alberta');
