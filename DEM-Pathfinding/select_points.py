import rasterio
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
from rasterio.plot import show
from rasterio.errors import CRSError

def check_crs(dem_path):
    """Checks if the DEM's CRS is projected or geographic."""
    with rasterio.open(dem_path) as src:
        try:
            if src.crs.is_geographic:
                print("\n" + "="*60)
                print("WARNING: Your DEM is in a Geographic Coordinate System (like WGS84).")
                print("         This can lead to inaccurate distance and slope calculations.")
                print("         It is highly recommended to reproject your DEM to a Projected")
                print("         Coordinate System (like UTM or Albers) before proceeding.")
                print("="*60 + "\n")
                return False
            elif src.crs.is_projected:
                print("\nCRS Check: OK. DEM is in a Projected Coordinate System.")
                return True
            else:
                print("\nWarning: Could not determine if CRS is projected or geographic.")
                return True # Allow proceeding but with a warning
        except CRSError:
            print("\n" + "="*60)
            print("ERROR: The DEM does not have a recognizable CRS (Coordinate Reference System).")
            print("       Cannot proceed. Please ensure your GeoTIFF is properly georeferenced.")
            print("="*60 + "\n")
            sys.exit(1)


class PointSelector:
    def __init__(self, dem_path):
        self.dem_path = dem_path
        self.base = None
        self.casualties = []
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.connection = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.mode = 'base' # Start by selecting the base

        with rasterio.open(self.dem_path) as src:
            self.transform = src.transform
            self.dem = src.read(1)
            # Use a robust percentile for color scaling to handle outliers
            vmin, vmax = np.nanpercentile(self.dem[self.dem > -9999], [2, 98])
            show(self.dem, ax=self.ax, transform=self.transform, cmap='terrain', vmin=vmin, vmax=vmax)
        
        self.update_title()
        plt.show()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        
        if self.mode == 'base':
            self.base = (y, x)
            self.ax.scatter(x, y, c='white', edgecolor='black', s=200, zorder=5, marker='H', label='NDRF Base')
            print(f"NDRF Base selected at (Northing, Easting): {self.base[0]:.2f}, {self.base[1]:.2f}")
            self.mode = 'casualties'
        
        elif self.mode == 'casualties':
            self.casualties.append((y, x))
            self.ax.scatter(x, y, c='red', edgecolor='white', s=150, zorder=5, marker='X')
            print(f"Casualty #{len(self.casualties)} selected at (Northing, Easting): {self.casualties[-1][0]:.2f}, {self.casualties[-1][1]:.2f}")

        self.update_title()
        self.fig.canvas.draw()

    def update_title(self):
        if self.mode == 'base':
            self.ax.set_title("Click to select the NDRF Base location. Close window when finished.", fontsize=14)
        else:
            self.ax.set_title(f"Click to select Casualty locations ({len(self.casualties)} selected). Close window when finished.", fontsize=14)

    def save_points(self):
        if self.base:
            with open("base.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.base[0], self.base[1]])
            print("\nBase location saved to base.csv")
        
        if self.casualties:
            with open("casualties.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.casualties)
            print("Casualty locations saved to casualties.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 select_points.py <path_to_dem.tif>")
        sys.exit(1)
    
    dem_file = sys.argv[1]
    
    # First, check the CRS of the DEM
    if not check_crs(dem_file):
        # Ask user if they want to continue despite the warning
        choice = input("Do you want to continue anyway? (y/n): ").lower()
        if choice != 'y':
            print("Exiting.")
            sys.exit(0)
            
    print("--- Interactive Point Selection ---")
    selector = PointSelector(dem_file)
    selector.save_points()
    print("\nPoint selection complete.")
