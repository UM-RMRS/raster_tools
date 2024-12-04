import os

import numpy as np

from raster_tools.utils import version_to_tuple

# Force the use of shapely 2 instead of pygeos in geopandas
os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd  # noqa: E402

# Check if geopandas has already been imported before raster_tools and turn off
# use of pygeos if it is turned on. shapely is required as the geopandas
# backend for line_stats.
if gpd.options.use_pygeos:
    gpd.options.use_pygeos = False


# Numpy 2.0 made several changes to type promotion rules.
NUMPY_GE_2 = version_to_tuple(np.__version__) >= (2, 0, 0)
