import numpy as np
import pytest

from raster_tools import dtypes
from raster_tools.dtypes import F16, F32, F64, I8, I32, U8, U16


@pytest.mark.parametrize(
    "value,expected_dtype",
    [
        # Integers (should use standard numpy logic)
        (10, np.dtype("uint8")),
        (-10, np.dtype("int8")),
        (65535, np.dtype("uint16")),
        # Floats without precision loss in float16
        (1.5, np.dtype("float16")),
        (0.0, np.dtype("float16")),
        # range fits float16, but precision requires float32
        (-9999.0, np.dtype("float32")),
        (-65503.0, np.dtype("float32")),
        # Needs float64 for exact precision
        (1.123456789123456, np.dtype("float64")),
        # Special floats (numpy defaults these to float16)
        (np.inf, np.dtype("float16")),
        (-np.inf, np.dtype("float16")),
        (np.nan, np.dtype("float16")),
    ],
)
def test_safe_min_scalar_type(value, expected_dtype):
    assert dtypes.safe_min_scalar_type(value) == expected_dtype


@pytest.mark.parametrize(
    "values,expected_dtype",
    [
        # -- Values only
        ([1, 2, 3], U8),
        ([-10, 10, 100], I8),
        ([1.5, 2.5], F16),
        # -9999.0 forces float32
        ([1.5, -9999.0], F32),
        # High precision float forces float64
        ([1, 1.123456789123456], F64),
        # -- Dtypes only
        ([F16, F32], F32),
        # Standard numpy promotion
        ([I8, U16], I32),
        # -- Mixed values and dtypes
        ([10, F16], F16),
        # Value requires higher precision
        ([-9999.0, F16], F32),
        ([-9999, F16], F32),
        ([I32, 1.123456789123456], F64),
        ([F64, 1], F64),
        # -- Single element iterables
        ([42], U8),
        ([F32], F32),
        ([-9999.0], F32),
    ],
)
def test_get_common_dtype(values, expected_dtype):
    assert dtypes.get_common_dtype(values) == expected_dtype


def test_get_common_dtype_empty_iterable():
    # reduce raises TypeError on an empty sequence without an initial value
    with pytest.raises(TypeError):
        dtypes.get_common_dtype([])
