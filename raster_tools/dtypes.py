from functools import reduce
from numbers import Integral, Number

import numpy as np
from xarray.core.dtypes import maybe_promote as xr_maybe_promote


def _add_type(mapping, dt, aliases):
    for d in [dt] + aliases:
        mapping[d] = dt


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

DTYPE_INPUT_TO_DTYPE = {}
# Unsigned int
_add_type(DTYPE_INPUT_TO_DTYPE, U8, ["u1", "uint8", "B", np.uint8])
_add_type(DTYPE_INPUT_TO_DTYPE, U16, ["u2", "uint16", np.uint16])
_add_type(DTYPE_INPUT_TO_DTYPE, U32, ["u4", "uint32", np.uint32])
_add_type(DTYPE_INPUT_TO_DTYPE, U64, ["u8", "uint64", np.uint64])
# Signed int
_add_type(DTYPE_INPUT_TO_DTYPE, I8, ["i1", "int8", "b", "byte", np.int8])
_add_type(DTYPE_INPUT_TO_DTYPE, I16, ["i2", "int16", np.int16])
_add_type(DTYPE_INPUT_TO_DTYPE, I32, ["i4", "int32", "i", np.int32])
_add_type(DTYPE_INPUT_TO_DTYPE, I64, ["i8", "int64", "int", int, np.int64])
# Float
_add_type(DTYPE_INPUT_TO_DTYPE, F16, ["f2", "float16", np.float16])
_add_type(DTYPE_INPUT_TO_DTYPE, F32, ["f4", "float32", "f", np.float32])
_add_type(
    DTYPE_INPUT_TO_DTYPE,
    F64,
    ["f8", "float64", "d", "float", float, np.float64],
)
_add_type(DTYPE_INPUT_TO_DTYPE, F128, ["f16", "float128", np.longdouble])
# Boolean
_add_type(
    DTYPE_INPUT_TO_DTYPE,
    BOOL,
    ["bool", "?", bool, np.bool_, np.dtype("bool"), np.bool8],
)


def is_str(value):
    return isinstance(value, str)


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


def is_scalar(value_or_dtype):
    if not isinstance(value_or_dtype, np.dtype):
        return isinstance(value_or_dtype, Number)
    return is_int(value_or_dtype) or is_float(value_or_dtype)


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
