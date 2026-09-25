import os
import re
import sys

import dask
import numpy as np

from raster_tools.utils import version_to_tuple

MIN_DASK_VERSION = (2025, 1, 0)


def _release_tuple(version_str):
    """Parse the leading numeric release segments of a version string.

    Tolerates dev and local suffixes such as ``2025.1.0+3.gabcdef``.
    """
    parts = []
    for piece in version_str.strip().split(".")[:3]:
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def check_dask_version(version_str=dask.__version__):
    """Raise a clear error if the installed dask is below the floor.

    Old dask releases fail deep inside ``import dask.dataframe`` on current
    Python versions, which hides the real problem. Checking up front turns
    that into an actionable message.
    """
    if _release_tuple(version_str) < MIN_DASK_VERSION:
        floor = ".".join(map(str, MIN_DASK_VERSION))
        raise ImportError(
            f"raster_tools requires dask>={floor} but found dask "
            f"{version_str}. Upgrade dask, e.g. "
            f"'conda install -c conda-forge \"dask>={floor}\"' or "
            f"'pip install \"dask>={floor}\"'."
        )


check_dask_version()

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
# Numpy 2.2 added two new matrix/vector ufuncs that don't work with rasters
NUMPY_GE_2_2 = version_to_tuple(np.__version__) >= (2, 2, 0)

PY_VER_310_PLUS = sys.version_info >= (3, 10)
