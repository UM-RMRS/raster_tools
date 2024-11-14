"""Module for providing test data."""

import glob
import os

import raster_tools as rts


class _DataProvider:
    def __init__(self, opener, data_dir, data_aliases, preferred_ext=None):
        self._opener = opener
        self._data_dir = data_dir
        self._data_aliases = data_aliases
        self._preferred_ext = preferred_ext

    def __getattr__(self, key):
        if key in self._data_aliases:
            value = self._data_aliases[key]
            if callable(value):
                return value(self._opener)
            return self._opener(value)

        # Try to find matching file
        paths = glob.glob(os.path.join(self._data_dir, f"{key}.*"))
        if len(paths):
            if self._preferred_ext is not None:
                for p in paths:
                    if os.path.splitext(p)[1] == self._preferred_ext:
                        return self._opener(p)
            # Try grabbing first one.
            return self._opener(paths[0])
        else:
            raise ValueError(f"Could not provide data for {key!r}")

    def open_file(self, file, *args, **kwargs):
        return self._opener(
            os.path.join(self._data_dir, file), *args, **kwargs
        )


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
RASTER_DIR = os.path.join(DATA_ROOT, "raster")
VECTOR_DIR = os.path.join(DATA_ROOT, "vector")


_raster_aliases = {
    "dem_aspect": os.path.join(RASTER_DIR, "aspect.tif"),
    "dem_curv": os.path.join(RASTER_DIR, "curv.tif"),
    "dem_easting": os.path.join(RASTER_DIR, "easting.tif"),
    "dem_hillshade": os.path.join(RASTER_DIR, "hillshade.tif"),
    "dem_northing": os.path.join(RASTER_DIR, "northing.tif"),
    "dem_slope": os.path.join(RASTER_DIR, "slope.tif"),
    "dem_slope_percent": os.path.join(RASTER_DIR, "slope_percent.tif"),
}
raster = _DataProvider(rts.Raster, RASTER_DIR, _raster_aliases, ".tif")

_vector_aliases = {
    "pods": os.path.join(VECTOR_DIR, "pods.shp"),
    "lmus": lambda x: x(os.path.join(VECTOR_DIR, "Zones.gdb"), leyers="LMUs"),
}
vector = _DataProvider(rts.open_vectors, VECTOR_DIR, _vector_aliases, ".shp")
