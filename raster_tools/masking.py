import warnings

import dask
import numpy as np

from raster_tools.dtypes import (
    BOOL,
    DTYPE_INPUT_TO_DTYPE,
    F16,
    F32,
    F64,
    F128,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    get_dtype_min_max,
    is_bool,
    is_float,
)


def create_null_mask(xrs, null_value):
    if null_value is not None:
        if not np.isnan(null_value):
            mask = xrs.data == null_value
        else:
            mask = np.isnan(xrs.data)
    else:
        mask = dask.array.zeros_like(xrs, dtype=bool)
    return mask


DTYPE_TO_DEFAULT_NULL = {
    BOOL: True,
    U8: U8.type(99),
    U16: U16.type(9_999),
    U32: U32.type(999_999_999),
    U64: U64.type(999_999_999_999),
    I8: I8.type(-99),
    I16: I16.type(-9_999),
    I32: I32.type(-999_999),
    I64: I64.type(-999_999),
    F16: np.finfo(F16).min,
    F32: F32.type(-999_999.0),
    F64: F64.type(-999_999.0),
    F128: F128.type(-999_999.0),
}


def get_default_null_value(dtype):
    """Get the default null value for a given dtype."""
    dtype = DTYPE_INPUT_TO_DTYPE[dtype]
    return DTYPE_TO_DEFAULT_NULL[dtype]


def reconcile_nullvalue_with_dtype(null_value, dtype, warn=False):
    """Make sure that the null value is consistent with the given dtype

    The null value is cast to the given dtype, if possible. If not, a warning
    is raised and the default for the dtype is returned.
    """
    if null_value is None:
        return None
    if np.isnan(null_value):
        return get_default_null_value(dtype)

    dtype = np.dtype(dtype)
    null_value = np.min_scalar_type(null_value).type(null_value)
    if np.can_cast(null_value, dtype):
        return null_value

    original_nv = null_value
    # TODO: stop checking if the value is an integer. Too much complexity. If
    # it is a float, treat it as a float.
    if is_float(original_nv) and float(original_nv).is_integer():
        if any(
            np.allclose(original_nv, v)
            for v in (*get_dtype_min_max(F32), *get_dtype_min_max(F64))
        ):
            nv = original_nv
        else:
            nv = int(original_nv)
            min_type = np.min_scalar_type(nv)
            nv = min_type.type(nv)
            if min_type.kind == "O":
                nv = original_nv
    else:
        nv = original_nv
    if np.can_cast(nv, dtype) or (is_bool(dtype) and nv in (0, 1)):
        return dtype.type(nv)
    nv = get_default_null_value(dtype)
    if warn:
        warnings.warn(
            f"The null value {original_nv!r} could not be cast to {dtype}. "
            f"It has been automatically changed to {nv!r}",
            stacklevel=2,
        )
    return nv
