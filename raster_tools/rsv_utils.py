from raster_tools._utils import is_str
from raster_tools.io import RasterDataError, RasterIOError
from raster_tools.raster import Raster
from raster_tools.vector import Vector, open_vectors


def get_raster(src):
    if isinstance(src, Raster):
        return src
    else:
        try:
            return Raster(src)
        except (ValueError, TypeError, RasterDataError, RasterIOError):
            raise ValueError(f"Could not convert input to Raster: {repr(src)}")


def get_vector(src):
    if isinstance(src, Vector):
        return src
    elif is_str(src):
        result = open_vectors(src)
        if not isinstance(result, Vector):
            # More than one layer was found
            raise ValueError("Input source must only have 1 layer")
        return result
