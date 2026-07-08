import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from osgeo import gdal
import csv
import time
import rasterio
from rasterio.plot import show

from src.pathfinding.grid_planner import solve


def get_grid_extent(data):
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


def grid2world(row, col, transform):
    x = transform[1] * col + transform[2] * row + transform[0]
    y = transform[4] * col + transform[5] * row + transform[3]
    return (y, x)


def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return None
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return None
    path.append(start)
    path.reverse()
    return path


def run_planner(cost_raster, casualties_csv, base_csv, dem_path, output_map, output_prefix, nhood_type=8, trace=None):
    try:
        with open(base_csv, 'r') as f:
            reader = csv.reader(f)
            base_proj = tuple(map(float, next(reader)))
    except FileNotFoundError:
        print(f"Error: {base_csv} not found. Run select_points.py first.")
        return

    casualties_proj = []
    try:
        with open(casualties_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                casualties_proj.append((float(row[0]), float(row[1])))
    except FileNotFoundError:
        print(f"Error: {casualties_csv} not found. Run select_points.py first.")
        return

    cost_data = gdal.Open(cost_raster)
    region_transform = cost_data.GetGeoTransform()
    cost_grid = cost_data.GetRasterBand(1).ReadAsArray()
    rows, cols = cost_grid.shape

    print(f"Using {rows}x{cols} cost grid from {cost_raster}")
    print(f"NDRF Base (Projected): {base_proj}")
    print(f"Found {len(casualties_proj)} casualties.")

    base_pixel = world2grid(base_proj[0], base_proj[1], region_transform)
    casualties_pixels = [world2grid(y, x, region_transform) for y, x in casualties_proj]

    print("Running full Dijkstra search from base...")
    t0 = time.time()
    came_from, cost_so_far, trace_grid = solve(cost_grid, base_pixel, ntype=nhood_type, trace=bool(trace))
    t1 = time.time()
    print(f"Full search complete in {t1 - t0:.2f} seconds.")

    all_paths = {}
    for i, casualty_pixel in enumerate(casualties_pixels):
        print(f"\n--- Reconstructing path for Casualty #{i + 1} at pixel {casualty_pixel} ---")
        path = reconstruct_path(came_from, base_pixel, casualty_pixel)
        if path:
            all_paths[i] = path
            total_cost = cost_so_far.get(casualty_pixel, -1)
            print(f"Path found with total cost: {total_cost:.2f}")
            path_outfile = f"{output_prefix}_{i + 1}.txt"
            np.savetxt(path_outfile, path, delimiter=",", fmt="%d")
            print(f"Path saved to {path_outfile}")
        else:
            print("No path found to this casualty.")

    if output_map:
        print("\nGenerating output map...")
        with rasterio.open(dem_path) as src:
            fig, ax = plt.subplots(figsize=(12, 12))
            dem_data = src.read(1)
            vmin, vmax = np.nanpercentile(dem_data[dem_data > -9999], [2, 98])
            show(src, ax=ax, cmap='terrain', vmin=vmin, vmax=vmax)

            cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.75, pad=0.02)
            cbar.set_label('Elevation (m)', fontsize=14)

            base_world_y, base_world_x = base_proj
            cas_world_y = [p[0] for p in casualties_proj]
            cas_world_x = [p[1] for p in casualties_proj]

            for i, path in all_paths.items():
                path_world_coords = [grid2world(p[0], p[1], region_transform) for p in path]
                path_y = [c[0] for c in path_world_coords]
                path_x = [c[1] for c in path_world_coords]
                ax.plot(path_x, path_y, linewidth=2.5, label=f'Path to Casualty {i + 1}')

            ax.scatter([base_world_x], [base_world_y], c='white', edgecolor='black', s=250, zorder=5, marker='H', label='NDRF Base')
            ax.scatter(cas_world_x, cas_world_y, c='red', edgecolor='white', s=200, zorder=5, marker='X', label='Casualties')
            ax.legend()
            ax.set_title('NDRF Optimal Paths on DEM', fontsize=16)
            plt.savefig(output_map)
            plt.close()
            print(f"Map saved to {output_map}")


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-r", "--region", help="Path to cost raster", default="cost.tif")
    parser.add_option("-m", "--map", help="Path to save output map", default="multi_path_map.png")
    parser.add_option("-p", "--path", help="Path prefix for solution paths", default="casualty_path")
    parser.add_option("--casualties", help="Path to casualties CSV", default="casualties.csv")
    parser.add_option("--base", help="Path to base CSV", default="base.csv")
    parser.add_option("--dem", help="Path to original DEM")
    parser.add_option("-n", "--nhood_type", help="Neighborhood type (4, 8, or 16)", default=8)
    parser.add_option("--trace", help="Path to save solver trace map", default=None)
    (options, args) = parser.parse_args()

    if not options.dem:
        parser.error("--dem is required")

    run_planner(
        cost_raster=options.region,
        casualties_csv=options.casualties,
        base_csv=options.base,
        dem_path=options.dem,
        output_map=options.map,
        output_prefix=options.path,
        nhood_type=options.nhood_type,
        trace=options.trace,
    )
