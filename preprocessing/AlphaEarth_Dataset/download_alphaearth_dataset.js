// ====================================================================
// TILED EXPORT FOR ALPHAEARTH EMBEDDINGS - SPECIFIC BAND
// ====================================================================

// 1. Define Alberta boundary
var canada = ee.FeatureCollection("FAO/GAUL/2015/level1");
var alberta = canada.filter(ee.Filter.eq('ADM1_NAME', 'Alberta')).geometry();

// 2. Define year 2020
var targetYear = '2020';
var startDate = targetYear + '-01-01';
var endDate = (parseInt(targetYear) + 1) + '-01-01';

// 3. SPECIFY THE BAND NAME HERE (e.g., 'A00', 'A15', 'A63')
var targetBandName = 'A61'; // CHANGE THIS TO YOUR DESIRED BAND

// 4. UINT8 Normalization function (-1 to +1 range → 0-255)
function normalizeToUint8(img) {
  // Transform to 0-1 range: (value + 1) / 2
  var normalized = img.add(1).divide(2);
  // Scale to 0-255 and convert to uint8
  return normalized.multiply(255).uint8();
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

// 7. Function to process and download a single tile for SPECIFIED BAND
function processTileForBand(tileInfo, tileNumber, bandName) {
  // Load the embedding collection
  var embeddingCollection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterBounds(tileInfo.tile)
    .filterDate(startDate, endDate);
  
  // Mosaic the collection for the year
  var annualEmbedding = embeddingCollection.mosaic();
  
  // Select only the target band
  var singleBand = annualEmbedding.select(bandName);
  
  // Clip to the tile area
  var clipped = singleBand.clip(tileInfo.tile);
  
  // Normalize to UINT8 (0-255)
  var uint8Image = normalizeToUint8(clipped);
  
  // Rename with year information
  var finalImage = uint8Image.rename(bandName + '_2020');
  
  // Export the image
  Export.image.toDrive({
    image: finalImage,
    description: 'Alberta_2020_' + bandName + '_tile_' + tileNumber,
    fileNamePrefix: 'Alberta_2020_' + bandName + '_tile_' + tileNumber + '_R' + tileInfo.row + 'C' + tileInfo.col,
    region: tileInfo.bounds,
    scale: 30,
    crs: 'EPSG:4326', // Use WGS84 for easier merging
    maxPixels: 1e13,
    folder: 'AlphaEarth_Band_' + bandName,
    fileFormat: 'GeoTIFF',
    formatOptions: {
      cloudOptimized: true
    }
  });
}

// 8. Export all tiles for the specified band
print('Starting export for band: ' + targetBandName);
for (var i = 0; i < tiles.length; i++) {
  print('Processing ' + targetBandName + ' - Tile ' + (i + 1) + ' of ' + tiles.length);
  processTileForBand(tiles[i], i, targetBandName);
}

// 9. Visualization
Map.centerObject(alberta, 6);
Map.addLayer(alberta, {color: 'red'}, 'Alberta Boundary');

// Add tile grid visualization
var gridFeatures = [];
for (var k = 0; k < tiles.length; k++) {
  gridFeatures.push(ee.Feature(tiles[k].bounds, {
    tileId: tiles[k].id,
    row: tiles[k].row,
    col: tiles[k].col
  }));
}
var gridFeatureCollection = ee.FeatureCollection(gridFeatures);
Map.addLayer(gridFeatureCollection, {color: 'blue'}, 'Tile Grid');

// 10. Print completion message
print('Export tasks for band ' + targetBandName + ' have been created.');
print('Check the "Tasks" tab to run them.');
print('');
print('To export a different band, change the "targetBandName" variable and run the script again.');
