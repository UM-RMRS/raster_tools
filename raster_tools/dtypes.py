from functools import reduce
from numbers import Integral, Number

import numpy as np
from xarray.core.dtypes import maybe_promote as xr_maybe_promote

U8 = np.dtype(np.uint8)
U16 = np.dtype(np.uint16)
U32 = np.dtype(np.uint32)
U64 = np.dtype(np.uint64)
I8 = np.dtype(np.int8)
I16 = np.dtype(np.int16)
I32 = np.dtype(np.int32)
I64 = np.dtype(np.int64)
F16 = np.dtype(np.float16)
F32 = np.dtype(np.float32)
F64 = np.dtype(np.float64)
# Some machines don't support np.float128 so use longdouble instead. This
# aliases f128 on machines that support it and f64 on machines that don't
F128 = np.dtype(np.longdouble)
BOOL = np.dtype(bool)
DTYPE_INPUT_TO_DTYPE = {
    # Unsigned int
    U8: U8,
    "uint8": U8,
    np.dtype("uint8"): U8,
    U16: U16,
    "uint16": U16,
    np.dtype("uint16"): U16,
    U32: U32,
    "uint32": U32,
    np.dtype("uint32"): U32,
    U64: U64,
    "uint64": U64,
    np.dtype("uint64"): U64,
    # Signed int
    I8: I8,
    "int8": I8,
    np.dtype("int8"): I8,
    I16: I16,
    "int16": I16,
    np.dtype("int16"): I16,
    I32: I32,
    "int32": I32,
    np.dtype("int32"): I32,
    I64: I64,
    "int64": I64,
    np.dtype("int64"): I64,
    int: I64,
    "int": I64,
    # Float
    F16: F16,
    "float16": F16,
    np.dtype("float16"): F16,
    F32: F32,
    "float32": F32,
    np.dtype("float32"): F32,
    F64: F64,
    "float64": F64,
    np.dtype("float64"): F64,
    F128: F128,
    "float128": F128,
    np.dtype("longdouble"): F128,
    float: F64,
    "float": F64,
    # Boolean
    BOOL: BOOL,
    bool: BOOL,
    "bool": BOOL,
    np.dtype("bool"): BOOL,
}


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


def get_common_dtype(values):
    return reduce(np.promote_types, map(np.min_scalar_type, values))


def promote_dtype_to_float(dtype):
    if dtype.kind == "f":
        return dtype
    if np.can_cast(dtype, F16):
        return F16
    elif np.can_cast(dtype, F32):
        return F32
    return F64


def maybe_promote(dtype):
    """Returns a dtype that can support missing values based on the input"""
    return xr_maybe_promote(dtype)[0]


def should_promote_to_fit(dtype, value):
    return is_float(value) and is_int(dtype)


def promote_data_dtype(xrs):
    dtype = maybe_promote(xrs.dtype)
    if dtype == xrs.dtype:
        return xrs
    return xrs.astype(dtype)


DTYPE_TO_DEFAULT_NULL = {
    BOOL: True,
    U8: 99,
    U16: 9999,
    U32: 999999,
    U64: 999999,
    I8: -99,
    I16: -9999,
    I32: -999999,
    I64: -999999,
    F16: -999999.0,
    F32: -999999.0,
    F64: -999999.0,
}


def get_default_null_value(dtype):
    """Get the default null value for a given dtype."""
    return DTYPE_TO_DEFAULT_NULL[dtype]
