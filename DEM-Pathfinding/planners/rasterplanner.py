import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from osgeo import gdal
import csv
import time
import rasterio
from rasterio.plot import show

import gridplanner

def getGridExtent(data):
    cols, rows = data.RasterXSize, data.RasterYSize
    transform = data.GetGeoTransform()
    minx, maxy = transform[0], transform[3]
    maxx = minx + transform[1] * cols
    miny = maxy + transform[5] * rows
    return {'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy, 'rows': rows, 'cols': cols}

def world2grid(y, x, transform):
    row = int((y - transform[3]) / transform[5])
    col = int((x - transform[0]) / transform[1])
    return (row, col)

# --- FIX: The missing grid2world function has been added back ---
def grid2world(row, col, transform):
    """Converts pixel/grid coordinates back to world coordinates."""
    x = transform[1] * col + transform[2] * row + transform[0]
    y = transform[4] * col + transform[5] * row + transform[3]
    return (y, x) # Returns (Northing, Easting)

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return None
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None: # Path is broken
             return None
    path.append(start)
    path.reverse()
    return path

parser = OptionParser()
parser.add_option("-r", "--region", help="Path to the pre-computed COST raster (e.g., cost.tif).", default="cost.tif")
parser.add_option("-m", "--map", help="Path to save the final solution map.", default="multi_path_map.png")
parser.add_option("-p", "--path", help="Path prefix to save solution paths (e.g., casualty_path).", default="casualty_path")
parser.add_option("--casualties", help="Path to a CSV file with casualty coordinates (Northing,Easting).", default="casualties.csv")
parser.add_option("--dem", help="Path to the original DEM for visualization.")
parser.add_option("-n", "--nhood_type", help="Neighborhood type (4, 8, or 16).", default=8)
parser.add_option("--trace", help="Path to save map of solver's history.", default=None)

(options, args) = parser.parse_args()

if not options.dem:
    parser.error("--dem option is required. The shell script should provide it automatically.")


# --- Base coordinates are now read from a file ---
try:
    with open("base.csv", 'r') as f:
        reader = csv.reader(f)
        base_proj = tuple(map(float, next(reader)))
except FileNotFoundError:
    parser.error("base.csv not found. Please run select_points.py first.")

costRasterFile = options.region
casualties_proj = []
try:
    with open(options.casualties, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            casualties_proj.append((float(row[0]), float(row[1])))
except FileNotFoundError:
    parser.error(f"{options.casualties} not found. Please run select_points.py first.")

costData = gdal.Open(costRasterFile)
regionTransform = costData.GetGeoTransform()
cost_grid = costData.GetRasterBand(1).ReadAsArray()
rows, cols = cost_grid.shape

print(f"Using {rows}x{cols} cost grid from {costRasterFile}")
print(f"NDRF Base (Projected): {base_proj}")
print(f"Found {len(casualties_proj)} casualties.")

base_pixel = world2grid(base_proj[0], base_proj[1], regionTransform)
casualties_pixels = [world2grid(y, x, regionTransform) for y, x in casualties_proj]

print("Running full Dijkstra search from base...")
t0 = time.time()
came_from, cost_so_far, traceGrid = gridplanner.solve(cost_grid, base_pixel, ntype=options.nhood_type, trace=bool(options.trace))
t1 = time.time()
print(f"Full search complete in {t1 - t0:.2f} seconds.")

all_paths = {}
for i, casualty_pixel in enumerate(casualties_pixels):
    print(f"\n--- Reconstructing path for Casualty #{i+1} at pixel {casualty_pixel} ---")
    path = reconstruct_path(came_from, base_pixel, casualty_pixel)
    if path:
        all_paths[i] = path
        total_cost = cost_so_far.get(casualty_pixel, -1)
        print(f"Path found with total cost: {total_cost:.2f}")

        path_outfile = f"{options.path}_{i+1}.txt"
        np.savetxt(path_outfile, path, delimiter=",", fmt="%d")
        print(f"Path saved to {path_outfile}")
    else:
        print("No path found to this casualty.")

# --- Visualization now uses the original DEM ---
if options.map:
    print("\nGenerating output map...")
    with rasterio.open(options.dem) as src:
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Use a robust percentile for color scaling to handle outliers
        dem_data = src.read(1)
        vmin, vmax = np.nanpercentile(dem_data[dem_data > -9999], [2, 98])
        
        # Plot the original DEM
        show(src, ax=ax, cmap='terrain', vmin=vmin, vmax=vmax)
        
        cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label('Elevation (m)', fontsize=14)

        # Re-calculate world coordinates for plotting on the DEM
        base_world_y, base_world_x = base_proj
        cas_world_y = [p[0] for p in casualties_proj]
        cas_world_x = [p[1] for p in casualties_proj]

        # Plot each path
        for i, path in all_paths.items():
            # Convert pixel path back to world coordinates for plotting
            path_world_coords = [grid2world(p[0], p[1], regionTransform) for p in path]
            path_y = [c[0] for c in path_world_coords]
            path_x = [c[1] for c in path_world_coords]
            ax.plot(path_x, path_y, linewidth=2.5, label=f'Path to Casualty {i+1}')

        # Plot the base station
        ax.scatter([base_world_x], [base_world_y], c='white', edgecolor='black', s=250, zorder=5, marker='H', label='NDRF Base')

        # Plot the casualty locations
        ax.scatter(cas_world_x, cas_world_y, c='red', edgecolor='white', s=200, zorder=5, marker='X', label='Casualties')

        ax.legend()
        ax.set_title('NDRF Optimal Paths on DEM', fontsize=16)
        plt.savefig(options.map)
        plt.close()
        print(f"Map saved to {options.map}")


