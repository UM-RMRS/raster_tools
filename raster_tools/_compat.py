import os

# Force the use of shapely 2 instead of pygeos in geopandas
os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd  # noqa: E402

# Check if geopandas has already been imported before raster_tools and turn off
# use of pygeos if it is turned on. shapely is required as the geopandas
# backend for line_stats.
if gpd.options.use_pygeos:
    gpd.options.use_pygeos = False
