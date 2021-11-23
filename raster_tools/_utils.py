import os
from numbers import Integral, Number

import dask
import numpy as np
import xarray as xr


def validate_file(path):
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Could not find file: '{path}'")


def is_str(value):
    return isinstance(value, str)


def is_scalar(value):
    return isinstance(value, Number)


def is_int(value_or_dtype):
    if isinstance(value_or_dtype, np.dtype):
        return value_or_dtype.kind in ("u", "i")
    return isinstance(value_or_dtype, Integral)


def is_float(value_or_dtype):
    if isinstance(value_or_dtype, np.dtype):
        return value_or_dtype.kind == "f"
    return is_scalar(value_or_dtype) and not is_int(value_or_dtype)


def is_bool(value_or_dtype):
    if isinstance(value_or_dtype, np.dtype):
        return value_or_dtype.kind == "b"
    return isinstance(value_or_dtype, (bool, np.bool_))


def is_xarray(rs):
    return isinstance(rs, (xr.DataArray, xr.Dataset))


def is_numpy(rs):
    return isinstance(rs, np.ndarray) or is_numpy_masked(rs)


def is_numpy_masked(rs):
    return isinstance(rs, np.ma.MaskedArray)


def is_dask(rs):
    return isinstance(rs, dask.array.Array)


def create_null_mask(xrs, null_value):
    if null_value is not None:
        if not np.isnan(null_value):
            mask = xrs.data == null_value
        else:
            mask = np.isnan(xrs.data)
    else:
        mask = dask.array.zeros_like(xrs, dtype=bool)
    return mask
