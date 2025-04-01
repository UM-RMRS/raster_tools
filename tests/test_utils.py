import numpy as np
import pytest

from raster_tools import utils


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (0, 0, True),
        (0, 1, False),
        (1.0, 1, True),
        (np.nan, np.nan, True),
        (np.nan, np.float32(np.nan), True),
        (np.nan, -9999, False),
        (-999, np.nan, False),
    ],
)
def test_nan_euqal(left, right, expected):
    assert utils.nan_equal(left, right) == expected


@pytest.mark.parametrize(
    "left,right,expected",
    [
        (0, 0, True),
        (0, 1, False),
        (1.0, 1, True),
        (np.nan, np.nan, True),
        (np.nan, -9999, False),
        (None, None, True),
        (None, 1, False),
        (1, None, False),
        (np.nan, None, False),
        (None, np.nan, False),
    ],
)
def test_null_values_equal(left, right, expected):
    assert utils.null_values_equal(left, right) == expected
