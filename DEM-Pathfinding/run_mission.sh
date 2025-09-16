#!/bin/bash

# --- Configuration ---
# Path to your Python script that generates the cost file.
# Make sure it's executable or call it with python3.
COST_GENERATOR_SCRIPT="create_cost_file.py" 

# --- Script Start ---
set -e # Exit immediately if a command exits with a non-zero status.

# Check for required arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_to_original_dem.tif> <slope_weight> <slope_change_weight>"
    echo "Example: ./run_mission.sh /data/dem.tif 5.0 10.0"
    exit 1
fi

DEM_FILE="$1"
SLOPE_WEIGHT="$2"
SLOPE_CHANGE_WEIGHT="$3"

# Define output file names
COST_FILE="cost.tif"
BASE_FILE="base.csv"
CASUALTIES_FILE="casualties.csv"
OUTPUT_MAP="mission_map.png"

echo "--- Step 1: Interactive Point Selection ---"
python3 select_points.py "$DEM_FILE"

# Check if points were actually selected
if [ ! -f "$BASE_FILE" ] || [ ! -f "$CASUALTIES_FILE" ]; then
    echo "Point selection was cancelled or no points were saved. Exiting."
    exit 1
fi

echo ""
echo "--- Step 2: Generating Cost File ($COST_FILE) ---"
echo "Using Slope Weight: $SLOPE_WEIGHT and Slope Change Weight: $SLOPE_CHANGE_WEIGHT"
# This calls your cost generator. It assumes the script is modified
# to take DEM path, output path, and weights as arguments.
# We will create a temporary version of your script for this.

# Create a temporary, parameterized cost generator
cat << EOF > temp_cost_generator.py
import rasterio
import numpy as np
import sys

dem_path = sys.argv[1]
output_path = sys.argv[2]
slope_weight = float(sys.argv[3])
slope_change_weight = float(sys.argv[4])

with rasterio.open(dem_path) as src:
    dem = src.read(1)
    profile = src.profile
    # Handle NoData values
    nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan

res = profile['transform'][0]
dy, dx = np.gradient(dem, res, res)
slope = np.sqrt(dx**2 + dy**2)

slope_dy, slope_dx = np.gradient(slope, res, res)
slope_change = np.sqrt(slope_dx**2 + slope_dy**2)

base_cost = np.ones_like(dem)

cost = base_cost + slope_weight * slope + slope_change_weight * slope_change

# Set nodata areas in cost to infinity (impassable)
if nodata is not None:
    cost[np.isnan(cost)] = np.inf

with rasterio.open(output_path, "w", **profile) as dst:
    dst.write(cost.astype(rasterio.float32), 1)
EOF

python3 temp_cost_generator.py "$DEM_FILE" "$COST_FILE" "$SLOPE_WEIGHT" "$SLOPE_CHANGE_WEIGHT"
rm temp_cost_generator.py # Clean up the temporary script
echo "Cost file generated successfully."

echo ""
echo "--- Step 3: Running Path Planner ---"
python3 planners/rasterplanner.py \
    --region "$COST_FILE" \
    --dem "$DEM_FILE" \
    --casualties "$CASUALTIES_FILE" \
    --map "$OUTPUT_MAP"

echo ""
echo "--- Mission Planning Complete ---"
echo "Output map saved to: $OUTPUT_MAP"
echo "Waypoint files saved for each casualty."
