
# DEM Pathfinding for Rescue Missions
This tool calculates the smoothest, lowest-cost paths from a single base to multiple casualty locations on a Digital Elevation Model (DEM). It uses a custom cost function that penalizes both steep slopes and rough, uneven terrain to find optimal routes for rescue teams.

# Dependencies
Before running, ensure you have the required Python libraries installed:

```pip install rasterio numpy matplotlib```

# How to Use
The entire workflow is automated with a single script.
### Run the Mission Script
Execute the run_mission.sh script from your terminal. You need to provide three arguments:

```./run_mission.sh [path/to/your/dem.tif] [slope_weight] [slope_change_weight]```

### Example
```chmod +x run_mission.sh ./run_mission.sh /data/dems/gore_range.tif 5.0 10.0 ```

