import numpy as np
import pytest

from raster_tools import utils
from tests.utils import (
    assert_datasets_similar,
    assert_raster_dataset_data_equal_any_nv,
    make_raster,
)


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


def test_assert_datasets_similar():
    left = make_raster("arange", shape=(1, 4, 4))._ds
    right = make_raster("arange", shape=(1, 4, 4))._ds
    # Distinct objects so the comparison body runs rather than the
    # identity shortcut.
    assert left is not right
    assert_datasets_similar(left, right)


def test_assert_datasets_similar_rejects_different_vars():
    left = make_raster("arange", shape=(1, 4, 4))._ds
    right = make_raster("arange", shape=(1, 4, 4))._ds.rename(
        {"mask": "other"}
    )
    with pytest.raises(AssertionError):
        assert_datasets_similar(left, right)


def test_assert_raster_ds_data_equal_any_nv_allows_different_nulls():
    left = make_raster("arange", shape=(1, 4, 4)).set_null_value(-9999)
    right = make_raster("arange", shape=(1, 4, 4)).set_null_value(-1)
    assert_raster_dataset_data_equal_any_nv(left._ds, right._ds)


def test_assert_raster_ds_data_equal_any_nv_rejects_nodata_mismatch():
    # Same data and masks, but only one side has a null value set. The
    # masks are both all-False, so only the nodata presence check can
    # catch the difference.
    left = make_raster("arange", shape=(1, 4, 4)).set_null_value(-9999)
    right = make_raster("arange", shape=(1, 4, 4))
    with pytest.raises(AssertionError):
        assert_raster_dataset_data_equal_any_nv(left._ds, right._ds)
