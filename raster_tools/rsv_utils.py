import dask.array as da
import numpy as np

from raster_tools._types import promote_dtype_to_float
from raster_tools._utils import is_str
from raster_tools.io import RasterDataError, RasterIOError
from raster_tools.raster import Raster
from raster_tools.vector import Vector, open_vectors


def get_raster(src, null_to_nan=False):
    rs = None
    if isinstance(src, Raster):
        rs = src
    else:
        try:
            rs = Raster(src)
        except (ValueError, TypeError, RasterDataError, RasterIOError):
            raise ValueError(f"Could not convert input to Raster: {repr(src)}")
    if null_to_nan and rs._masked:
        rs = rs.copy()
        data = rs._rs.data
        new_dtype = promote_dtype_to_float(data.dtype)
        if new_dtype != data.dtype:
            data = data.astype(new_dtype)
        rs._rs.data = da.where(rs._mask, np.nan, data)
    return rs


def get_vector(src):
    if isinstance(src, Vector):
        return src
    elif is_str(src):
        result = open_vectors(src)
        if not isinstance(result, Vector):
            # More than one layer was found
            raise ValueError("Input source must only have 1 layer")
        return result
