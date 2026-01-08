// ====================================================================
// LANDSAT-8 SURFACE REFLECTANCE DOWNLOAD - AUTO FOLDER CREATION
// ====================================================================

// 1. USE YOUR UPLOADED SHAPEFILE
var canada = ee.FeatureCollection("FAO/GAUL/2015/level1");
var alberta = canada.filter(ee.Filter.eq('ADM1_NAME', 'Alberta')).geometry();

// 2. Define year 2020
var targetYear = '2020';
var startDate = targetYear + '-01-01';
var endDate = (parseInt(targetYear) + 1) + '-01-01';

// 3. SPECIFY THE BAND NAME HERE
var targetBandName = 'SR_B7'; // CHANGE AS NEEDED

// 4. UINT8 Normalization for T1_L2 Surface Reflectance
function normalizeSRToUint8(img) {
  var scaled = img.multiply(0.0000275).add(-0.2);
  var clipped = scaled.clamp(0, 1);
  return clipped.multiply(255).uint8();
}

// 5. Function to create a grid of tiles over Alberta
function createTiles(geometry, numTiles) {
  var bounds = geometry.bounds({maxError: 100});
  var boundsInfo = bounds.getInfo();
  var coords = boundsInfo.coordinates[0];
  
  var xCoords = [];
  var yCoords = [];
  
  for (var c = 0; c < coords.length; c++) {
    xCoords.push(coords[c][0]);
    yCoords.push(coords[c][1]);
  }
  
  var xMinValue = Math.min.apply(null, xCoords);
  var xMaxValue = Math.max.apply(null, xCoords);
  var yMinValue = Math.min.apply(null, yCoords);
  var yMaxValue = Math.max.apply(null, yCoords);
  
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

// 6. Create tiles (4x4 grid to avoid memory issues)
var numTiles = 25; 
var tiles = createTiles(alberta, numTiles);

// Add this after creating the tiles (after line 67):
var tileFeatures = tiles.map(function(tile) {
  return ee.Feature(tile.tile, {id: tile.id, row: tile.row, col: tile.col});
});

var tileFeatureCollection = ee.FeatureCollection(tileFeatures);

// Add tile boundaries to the map
var tileOutlines = tiles.map(function(tile) {
  return ee.Feature(tile.bounds, {id: tile.id, row: tile.row, col: tile.col});
});

var tileFeatureCollection = ee.FeatureCollection(tileOutlines);

// Add tile boundaries to the map
Map.addLayer(tileFeatureCollection, 
  {color: 'yellow', fillColor: '00000000', width: 2}, 
  'Tile Grid (25 tiles)');
  
  
// 7. Function to mask clouds using QA_PIXEL band
function maskCloudsL8SR(image) {
  var qa = image.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0);  // Cloud (bit 3)
  var shadowMask = qa.bitwiseAnd(1 << 4).eq(0); // Cloud shadow (bit 4)
  var snowMask = qa.bitwiseAnd(1 << 5).eq(0);   // Snow (bit 5)
  
  return image.updateMask(cloudMask)
               .updateMask(shadowMask)
               .updateMask(snowMask);
}

// 8. Function to process and download a single tile
function processTileForBand(tileInfo, tileNumber, bandName) {
  // Load Landsat-8 Surface Reflectance collection
  var landsatCollection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(tileInfo.tile)
    .filterDate(startDate, endDate);
  
  // Apply cloud masking
  var maskedCollection = landsatCollection.map(maskCloudsL8SR);
  var medianComposite = maskedCollection.median();
  var singleBand = medianComposite.select(bandName);
  var clipped = singleBand.clip(tileInfo.tile);
  var uint8Image = normalizeSRToUint8(clipped);
  var finalImage = uint8Image.rename(bandName + '_2020_SR');
  
  // Reproject to NAD83 Statistics Canada (EPSG:3979)
  var reprojectedImage = finalImage.reproject({
    crs: 'EPSG:3979',
    scale: 30
  });
  
  // ====================================================
  // AUTO-FOLDER CREATION HAPPENS HERE:
  // ====================================================
  Export.image.toDrive({
    image: reprojectedImage,
    description: 'Alberta_L8_SR_2020_' + bandName + '_tile_' + tileNumber,
    fileNamePrefix: 'Alberta_2020_L8_SR_' + bandName + '_tile_' + tileNumber + '_R' + tileInfo.row + 'C' + tileInfo.col,
    region: tileInfo.bounds,
    scale: 30,
    crs: 'EPSG:3979',
    maxPixels: 1e13,
    // FOLDER WILL BE AUTO-CREATED WHEN EXPORT RUNS
    folder: 'Alberta/2020/Landsat8_SR/Band_' + bandName.replace('SR_', ''),
    fileFormat: 'GeoTIFF',
    formatOptions: {
      cloudOptimized: true
    }
  });
}

// 9. Export all tiles for the specified band
print('Starting export for Landsat-8 Surface Reflectance band: ' + targetBandName);
print('Folder will be auto-created: Alberta/2020/Landsat8_SR/Band_' + targetBandName.replace('SR_', ''));
for (var i = 0; i < tiles.length; i++) {
  print('Processing ' + targetBandName + ' - Tile ' + (i + 1) + ' of ' + tiles.length);
  processTileForBand(tiles[i], i, targetBandName);
}

// 10. Visualization and completion message
Map.centerObject(alberta, 6);
Map.addLayer(alberta, {color: 'red'}, 'Alberta Boundary');

print('===========================================');
print('EXPORT READY - AUTO FOLDER CREATION');
print('===========================================');
print('Dataset: Landsat-8 Surface Reflectance (T1_L2)');
print('Band: ' + targetBandName);
print('Year: 2020');
print('Tiles: ' + tiles.length + ' files');
print('CRS: EPSG:3979 (NAD83 Statistics Canada)');
print('Output location:');
print('  Google Drive → Alberta/2020/Landsat8_SR/Band_' + targetBandName.replace('SR_', ''));
print('');
print('FOLDER CREATION PROCESS:');
print('1. Run export tasks from "Tasks" tab');
print('2. GEE automatically creates: Alberta/2020/Landsat8_SR/');
print('3. Then creates subfolder: Band_' + targetBandName.replace('SR_', ''));
print('4. Saves all GeoTIFF files inside');
print('');
print('NOTE: If folder exists, files will be added to it');
print('===========================================');
