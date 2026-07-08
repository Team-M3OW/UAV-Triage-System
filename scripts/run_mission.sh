#!/bin/bash
set -e

DEM_PATH="${1:?Usage: $0 <dem_path> <slope_weight> <slope_change_weight>}"
SLOPE_WEIGHT="${2:?Usage: $0 <dem_path> <slope_weight> <slope_change_weight>}"
SLOPE_CHANGE_WEIGHT="${3:?Usage: $0 <dem_path> <slope_weight> <slope_change_weight>}"

echo "=== UAV Triage System - DEM Pathfinding ==="
echo "DEM: ${DEM_PATH}"
echo "Slope Weight: ${SLOPE_WEIGHT}"
echo "Slope Change Weight: ${SLOPE_CHANGE_WEIGHT}"

python3 src/pathfinding/select_points.py "${DEM_PATH}"

cat > /tmp/gen_cost.py << 'EOF'
import sys
import numpy as np
from osgeo import gdal

dem_path = sys.argv[1]
slope_w = float(sys.argv[2])
slope_change_w = float(sys.argv[3])

ds = gdal.Open(dem_path)
band = ds.GetRasterBand(1)
dem = band.ReadAsArray().astype(float)

dy, dx = np.gradient(dem)
slope = np.sqrt(dy**2 + dx**2)
slope_change = np.sqrt(np.gradient(slope, axis=0)**2 + np.gradient(slope, axis=1)**2)

cost = 1.0 + slope_w * slope + slope_change_w * slope_change

driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create('cost.tif', ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
out_ds.SetGeoTransform(ds.GetGeoTransform())
out_ds.SetProjection(ds.GetProjection())
out_ds.GetRasterBand(1).WriteArray(cost)
out_ds = None
print("cost.tif created")
EOF

python3 /tmp/gen_cost.py "${DEM_PATH}" "${SLOPE_WEIGHT}" "${SLOPE_CHANGE_WEIGHT}"

python3 src/pathfinding/raster_planner.py --dem "${DEM_PATH}"

echo "=== Done ==="
