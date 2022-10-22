import os

import dask
import numpy as np
import xarray as xr


def validate_file(path):
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Could not find file: '{path}'")


def validate_path(path):
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Path does not exist: '{path}'")


def is_xarray(rs):
    return isinstance(rs, (xr.DataArray, xr.Dataset))


def is_numpy(rs):
    return isinstance(rs, np.ndarray) or is_numpy_masked(rs)


def is_numpy_masked(rs):
    return isinstance(rs, np.ma.MaskedArray)


def is_dask(rs):
    return isinstance(rs, dask.array.Array)
