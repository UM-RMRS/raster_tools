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
    DTYPE_INPUT_TO_DTYPE, BOOL, ["bool", "?", bool, np.bool_, np.dtype("bool")]
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


def safe_min_scalar_type(value):
    """wraps np.min_scalar_type but accounts for precision loss"""
    suggested_dtype = np.min_scalar_type(value)

    # If it's a float, check for precision loss
    if np.issubdtype(suggested_dtype, np.floating):
        if np.isnan(value) or np.isinf(value):
            return suggested_dtype

        for dtype in [np.float16, np.float32, np.float64]:
            if np.dtype(dtype).itemsize < suggested_dtype.itemsize:
                continue

            casted_value = dtype(value)
            if value == float(casted_value):
                return np.dtype(dtype)
        return np.dtype("float64")

    return suggested_dtype


def get_common_dtype(items):
    if not items:
        raise TypeError("Cannot determine common dtype of an empty sequence")

    dtypes_to_promote = []
    int_literals = []

    for x in items:
        # Pass explicit dtypes straight through
        if isinstance(x, np.dtype):
            dtypes_to_promote.append(x)
        # Pool raw integers (excluding bools) for collective bounds checking
        elif isinstance(x, int) and not isinstance(x, bool):
            int_literals.append(x)
        # Handle floats individually so they pass through your precision
        # checks
        else:
            dtypes_to_promote.append(safe_min_scalar_type(x))

    # Find the absolute minimum dtype that fits the entire range of pooled
    # integers
    if int_literals:
        min_val = min(int_literals)
        max_val = max(int_literals)
        # Ordered to match np.min_scalar_type's exact preference for unsigned
        # types
        int_types = [U8, I8, U16, I16, U32, I32, U64, I64]
        for dt in int_types:
            # If we have negative numbers, immediately skip any unsigned types
            if min_val < 0 and np.issubdtype(dt, np.unsignedinteger):
                continue

            info = np.iinfo(dt)
            if info.min <= min_val and max_val <= info.max:
                dtypes_to_promote.append(np.dtype(dt))
                break

    return reduce(np.promote_types, dtypes_to_promote)


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


def get_dtype_info(dtype):
    dtype = np.dtype(dtype)
    if is_bool(dtype):
        raise ValueError("No info for bool")
    return np.iinfo(dtype) if is_int(dtype) else np.finfo(dtype)


def get_dtype_min_max(dtype):
    dtype = np.dtype(dtype)
    if is_bool(dtype):
        return (False, True)
    info = get_dtype_info(dtype)
    # iinfo does not return np.<type> values for min and max
    min_ = dtype.type(info.min)
    max_ = dtype.type(info.max)
    return min_, max_


INT_DTYPE_TO_FLOAT_DTYPE = {
    U8: F16,
    U16: F32,
    U32: F64,
    I8: F16,
    I16: F32,
    I32: F64,
    F64: F64,
}
