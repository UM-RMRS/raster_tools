import os

# Force the use of shapely 2 instead of pygeos in geopandas
os.environ["USE_PYGEOS"] = "0"  # noqa: E402

import geopandas as gpd  # noqa: E402

# Check if geopandas has already been imported before raster_tools and set the
# use_pygeos option accordingly
if gpd.options.use_pygeos:
    gpd.options.use_pygeos = False
