import operator
import pathlib
import re
import unittest
import warnings

import affine
import dask
import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import rioxarray as riox
import shapely
import xarray as xr
from affine import Affine
from shapely.geometry import box

import raster_tools as rts
import raster_tools.raster
from raster_tools import Raster, stack_bands
from raster_tools._compat import NUMPY_GE_2, NUMPY_GE_2_2
from raster_tools.dtypes import (
    BOOL,
    DTYPE_INPUT_TO_DTYPE,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    is_bool,
    is_float,
    is_int,
    is_scalar,
)
from raster_tools.masking import (
    get_default_null_value,
    reconcile_nullvalue_with_dtype,
)
from raster_tools.raster import (
    RasterQuadrantsResult,
    data_to_raster,
    rowcol_to_xy,
    xy_to_rowcol,
)
from raster_tools.utils import is_strictly_decreasing, is_strictly_increasing
from raster_tools.vector import Vector
from tests import testdata
from tests.utils import (
    arange_nd,
    assert_dataarrays_similar,
    assert_rasters_similar,
    assert_valid_raster,
    make_raster,
)


def dummy_dataarray(data, dtype=None):
    data = np.asarray(data, dtype=dtype)
    dims = [f"d{i}" for i in range(data.ndim)]
    coords = [np.arange(n) for n in data.shape]
    return xr.DataArray(data, dims=dims, coords=coords)


@pytest.mark.parametrize(
    "data,nv,expected",
    [
        (np.arange(4), 0, np.array([1, 0, 0, 0])),
        (np.arange(4), np.nan, np.array([0, 0, 0, 0])),
        (np.arange(4), None, np.array([0, 0, 0, 0])),
        ([1, np.nan, 2, 3], np.nan, np.array([0, 1, 0, 0])),
        ([1, np.nan, 2, 3], None, np.array([0, 0, 0, 0])),
        (da.arange(4), 0, da.from_array([1, 0, 0, 0])),
        (
            da.from_array([0, np.nan, 1, 2]),
            np.nan,
            da.from_array([0, 1, 0, 0]),
        ),
        (da.from_array([0, np.nan, 1, 2]), None, da.from_array([0, 0, 0, 0])),
        (dummy_dataarray([1, 2, 3]), 1, dummy_dataarray([1, 0, 0])),
        (
            dummy_dataarray([1, 2, 3]).chunk(),
            1,
            dummy_dataarray([1, 0, 0]).chunk(),
        ),
        (dummy_dataarray([1, 2, 3]), np.nan, dummy_dataarray([0, 0, 0])),
        (dummy_dataarray([1, np.nan, 3]), np.nan, dummy_dataarray([0, 1, 0])),
        (dummy_dataarray([1, np.nan, 3]), None, dummy_dataarray([0, 0, 0])),
    ],
)
def test_get_mask_from_data(data, nv, expected):
    expected = expected.astype(bool)
    result = raster_tools.raster.get_mask_from_data(data, nv)
    if dask.is_dask_collection(expected):
        assert dask.is_dask_collection(result)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "data", [da.ones(4), xr.ones_like(testdata.raster.dem_small.xdata)]
)
def test_get_mask_from_data_single_chunk_result_writeable(data):
    result = raster_tools.raster.get_mask_from_data(data, None)
    if isinstance(data, da.Array):
        assert data.npartitions == 1
        assert result.npartitions == 1
        mask = result.compute()
    else:
        assert data.data.npartitions == 1
        assert result.data.npartitions == 1
        mask = result.data.compute()
    assert mask.flags.writeable


@pytest.mark.parametrize("nv", [None, np.nan, 0])
@pytest.mark.parametrize(
    "data",
    [
        arange_nd((1, 100, 100)).astype(float),
        arange_nd((3, 100, 100)).astype(float),
        arange_nd((100, 100)).astype(float),
        da.from_array(arange_nd((3, 100, 100))).astype(float),
        da.from_array(arange_nd((3, 100, 100)), chunks=(1, 10, 10)).astype(
            float
        ),
    ],
)
def test_data_to_xr_raster(data, xdem_small, nv):
    xlike = xdem_small
    expected = xlike.copy()
    nbands = 1 if data.ndim == 2 or data.shape[0] == 1 else data.shape[0]
    if nbands > 1:
        expected = xr.concat(
            [expected for i in range(nbands)], dim="band", join="inner"
        )
        expected["band"] = np.arange(nbands) + 1
    data_copy = data.copy()
    if data_copy.ndim == 2:
        data_copy = data_copy[None]
    expected.data = da.asarray(data_copy).rechunk(xlike.data.chunksize)

    # Check using x/y
    result1 = raster_tools.raster.data_to_xr_raster(
        data,
        x=xlike.x.to_numpy(),
        y=xlike.y.to_numpy(),
        crs=xlike.rio.crs,
        nv=nv,
    )
    # Check using affine
    result2 = raster_tools.raster.data_to_xr_raster(
        data, affine=xlike.rio.transform(True), crs=xlike.rio.crs, nv=nv
    )
    results = [result1, result2]
    for result in results:
        assert_dataarrays_similar(
            result, xlike, check_nbands=False, check_chunks=False
        )
        assert np.allclose(result, expected)
        assert np.allclose(result.band, np.arange(nbands) + 1)
        assert dask.is_dask_collection(result)
        if nv is None:
            assert result.rio.nodata is None
        elif np.isnan(nv):
            assert np.isnan(result.rio.nodata)
        else:
            assert result.rio.nodata == nv


@pytest.mark.parametrize("nv", [None, np.nan, 0])
@pytest.mark.parametrize(
    "data",
    [
        arange_nd((1, 100, 100)).astype(float),
        arange_nd((3, 100, 100)).astype(float),
        arange_nd((100, 100)).astype(float),
        da.from_array(arange_nd((3, 100, 100))).astype(float),
        da.from_array(arange_nd((3, 100, 100)), chunks=(1, 10, 10)).astype(
            float
        ),
    ],
)
def test_data_to_xr_raster_like(data, xdem_small, nv):
    xlike = xdem_small.chunk(chunks={"band": 1, "y": 15, "x": 15})
    expected = xlike.copy()
    nbands = 1 if data.ndim == 2 or data.shape[0] == 1 else data.shape[0]
    if nbands > 1:
        expected = xr.concat(
            [expected for i in range(nbands)], dim="band", join="inner"
        )
        expected["band"] = np.arange(nbands) + 1
    data_copy = data.copy()
    if data_copy.ndim == 2:
        data_copy = data_copy[None]
    expected.data = da.asarray(data_copy).rechunk(xlike.data.chunksize)

    result = raster_tools.raster.data_to_xr_raster_like(data, xlike, nv=nv)
    assert_dataarrays_similar(
        result, xlike, check_nbands=False, check_chunks=True
    )
    assert np.allclose(result, expected)
    assert np.allclose(result.band, np.arange(nbands) + 1)
    assert dask.is_dask_collection(result)
    if nv is None:
        assert result.rio.nodata is None
    elif np.isnan(nv):
        assert np.isnan(result.rio.nodata)
    else:
        assert result.rio.nodata == nv


TEST_ARRAY = np.array(
    [
        [1, 3, 4, 4, 3, 2],
        [7, 3, 2, 6, 4, 6],
        [5, 8, 7, 5, 6, 6],
        [1, 4, 5, -1, 5, 1],
        [4, 7, 5, -1, 2, 6],
        [1, 2, 2, 1, 3, 4],
    ]
)


@pytest.mark.parametrize(
    "data",
    [
        np.ones((4, 4)),
        np.ones((1, 4, 4)),
        np.ones((1, 1)),
        arange_nd((100, 100)),
        arange_nd((10, 10, 10)),
        arange_nd((2, 10)),
        *[arange_nd((10, 10), dtype=t) for t in [I8, I32, U16, F16, F32]],
        np.where(TEST_ARRAY == -1, np.nan, TEST_ARRAY),
    ],
)
def test_raster_from_array(data):
    raster = Raster(data)
    assert_valid_raster(raster)
    assert raster._ds.raster.data is not data
    if data.ndim == 2:
        assert raster.shape[0] == 1
        assert raster.shape[1:] == data.shape
    else:
        assert raster.shape == data.shape
    assert raster.dtype == data.dtype
    if not np.isnan(data).any():
        assert raster.null_value is None
        assert not raster.mask.any()
        assert np.allclose(raster, data)
    else:
        mask = np.isnan(data)
        data_cp = data.copy()
        data_cp[mask] = get_default_null_value(data.dtype)
        assert raster.null_value == get_default_null_value(data.dtype)
        assert np.allclose(raster, data_cp)

    # Band dim starts at 1
    band = [1] if data.ndim == 2 else np.arange(data.shape[0]) + 1
    assert np.allclose(raster._ds.raster.band.data, band)
    # 2D coords are offset by 0.5
    assert np.allclose(
        raster._ds.raster.x.data, np.arange(data.shape[-1]) + 0.5
    )
    assert np.allclose(
        raster._ds.raster.y.data, (np.arange(data.shape[-2]) + 0.5)[::-1]
    )


@pytest.mark.parametrize(
    "data",
    [
        da.ones((4, 4)),
        da.ones((1, 4, 4)),
        da.ones((1, 1)),
        arange_nd((100, 100), mod=da),
        arange_nd((10, 10, 10), mod=da),
        arange_nd((2, 10), mod=da),
        *[
            arange_nd((10, 10), dtype=t, mod=da)
            for t in [I8, I32, U16, F16, F32]
        ],
        da.from_array(np.where(TEST_ARRAY == -1, np.nan, TEST_ARRAY)),
    ],
)
def test_raster_from_dask_array(data):
    assert dask.is_dask_collection(data)

    raster = Raster(data)
    assert_valid_raster(raster)
    assert raster._ds.raster.data is not data
    if data.ndim == 2:
        assert raster.shape == (1, *data.shape)
    else:
        assert raster.shape == data.shape
    assert raster.dtype == data.dtype
    assert dask.is_dask_collection(raster._ds.raster)
    assert dask.is_dask_collection(raster._ds.mask)
    if is_float(data.dtype):
        mask = np.isnan(data)
        assert np.allclose(raster._ds.mask.data.compute(), mask.compute())
        data_cp = data.copy()
        data_cp[mask] = get_default_null_value(data.dtype)
        assert raster.null_value == get_default_null_value(data.dtype)
        assert np.allclose(raster, data_cp.compute())
    else:
        assert raster.null_value is None
        assert not raster.mask.any().compute()
        assert np.allclose(raster, data.compute())
    assert raster._ds.raster.dims == ("band", "y", "x")
    assert raster._ds.mask.dims == ("band", "y", "x")
    # Band dim starts at 1
    band = [1] if data.ndim == 2 else np.arange(data.shape[0]) + 1
    assert np.allclose(raster._ds.raster.band.data, band)
    # 2D coords are offset by 0.5
    assert np.allclose(
        raster._ds.raster.x.data, np.arange(data.shape[-1]) + 0.5
    )
    # y dim goes from high to low
    if data.shape[-2] > 1:
        assert is_strictly_decreasing(raster._ds.y)
    assert np.allclose(
        raster._ds.raster.y.data, (np.arange(data.shape[-2]) + 0.5)[::-1]
    )


@pytest.mark.parametrize(
    "xdata",
    [
        xr.DataArray(
            np.arange(20).reshape((4, 5)),
            coords=(np.arange(4)[::-1], np.arange(5)),
        ),
        xr.DataArray(
            np.arange(20.0).reshape((4, 5)),
            coords=(np.arange(4)[::-1], np.arange(5)),
        )
        .rio.write_nodata(np.nan)
        .rio.write_crs(3857),
        xr.DataArray(
            np.arange(20).reshape((4, 5)),
            # increasing y
            coords=(np.arange(4), np.arange(5)),
        ),
        xr.DataArray(
            np.arange(20).reshape((4, 5)),
            dims=("y", "x"),
            coords=(np.arange(4)[::-1], np.arange(5)),
        ),
        xr.DataArray(
            np.arange(40).reshape((2, 4, 5)),
            coords=([1, 2], np.arange(4)[::-1], np.arange(5)),
        ),
        xr.DataArray(
            np.arange(40).reshape((2, 4, 5)),
            dims=("band", "y", "x"),
            coords=([1, 2], np.arange(4)[::-1], np.arange(5)),
        ),
        xr.DataArray(
            np.arange(40).reshape((2, 4, 5)),
            dims=("band", "y", "x"),
            coords=([1, 2], np.arange(4)[::-1], np.arange(5)),
        ).rio.write_crs("EPSG:3857"),
        xr.DataArray(
            np.arange(20).reshape((1, 4, 5)),
            dims=("other", "lat", "lon"),
            coords=([1], np.arange(4)[::-1], np.arange(5)),
        ),
        # Dask
        xr.DataArray(
            da.from_array(np.arange(20).reshape((4, 5))),
            coords=(np.arange(4)[::-1], np.arange(5)),
        ),
        xr.DataArray(
            da.from_array(np.arange(20).reshape((1, 4, 5))),
            dims=("other", "lat", "lon"),
            coords=([1], np.arange(4)[::-1], np.arange(5)),
        ).chunk({"other": 1, "lat": 1, "lon": 1}),
    ],
)
def test_raster_from_dataarray(xdata):
    raster = Raster(xdata)
    assert_valid_raster(raster)
    assert raster._ds.raster is not xdata

    yc = xdata.dims[1] if xdata.ndim == 3 else xdata.dims[0]
    xc = xdata.dims[2] if xdata.ndim == 3 else xdata.dims[1]
    if is_strictly_increasing(xdata[yc]):
        xdata = xdata.isel({yc: slice(None, None, -1)})
    xdata = xdata.compute()
    assert raster.crs == xdata.rio.crs

    if xdata.ndim == 2:
        assert raster.shape[0] == 1
        assert raster.shape[1:] == xdata.shape
    else:
        assert raster.shape == xdata.shape
    assert raster.dtype == xdata.dtype
    if not is_float(xdata) or not np.isnan(xdata.data).any():
        if xdata.rio.nodata is None:
            assert raster.null_value is None
            assert not raster.mask.any().compute()
        elif np.isnan(xdata.rio.nodata):
            assert raster.null_value == get_default_null_value(xdata.dtype)
            assert np.allclose(raster.xmask, xdata == xdata.rio.nodata)
        else:
            assert raster.null_value == xdata.rio.nodata
            assert np.allclose(raster.xmask, xdata == xdata.rio.nodata)
        assert np.allclose(raster, xdata)
    elif np.isnan(xdata.data).any():
        mask = np.isnan(xdata.data)
        data_cp = xdata.data.copy()
        data_cp[mask] = get_default_null_value(xdata.dtype)
        assert raster.null_value == get_default_null_value(xdata.dtype)
        assert np.allclose(raster, data_cp)
    else:
        raise AssertionError()
    # Band dim starts at 1
    band = [1] if xdata.ndim == 2 else np.arange(xdata.shape[0]) + 1
    assert np.allclose(raster._ds.band, band)
    assert np.allclose(raster._ds.x, xdata[xc])
    assert np.allclose(raster._ds.y, xdata[yc])


def test_raster_from_dataarray_str_band(xdem_small):
    xdem_small["band"] = ["one"]
    raster = Raster(xdem_small)
    assert np.allclose(raster.band, np.array([1]))
    assert np.allclose(raster.xdata.band.to_numpy(), np.array([1]))
    assert raster.xdata.band.to_numpy().dtype == I64


@pytest.mark.parametrize(
    "path",
    [
        "tests/data/raster/dem_small.tif",
        pathlib.Path("tests/data/raster/dem_small.tif"),
    ],
)
def test_raster_from_path(path):
    raster = Raster(path)
    assert_valid_raster(raster)


@pytest.mark.parametrize(
    "ds",
    [
        # Vanilla
        xr.Dataset(
            {
                "raster": (("y", "x"), arange_nd((10, 10))),
                "mask": (("y", "x"), np.zeros((10, 10), dtype=bool)),
            },
            coords={"y": np.arange(10)[::-1], "x": np.arange(10)},
        ),
        xr.Dataset(
            {
                "raster": (
                    ("y", "x"),
                    np.where(TEST_ARRAY == -1, np.nan, TEST_ARRAY),
                ),
                "mask": (("y", "x"), TEST_ARRAY == -1),
            },
            coords={
                "y": np.arange(TEST_ARRAY.shape[0])[::-1],
                "x": np.arange(TEST_ARRAY.shape[1]),
            },
        ),
        xr.Dataset(
            {
                "raster": (("band", "y", "x"), arange_nd((1, 10, 10))),
                "mask": (
                    ("y", "x"),
                    np.zeros((10, 10), dtype=bool),
                ),
            },
            coords={
                "band": [1],
                "y": np.arange(10)[::-1],
                "x": np.arange(10),
            },
            attrs={"_FillValue": 10},
        ),
        xr.Dataset(
            {
                "raster": (("y", "x"), arange_nd((10, 10))),
                "mask": (
                    ("band", "y", "x"),
                    np.zeros((1, 10, 10), dtype=bool),
                ),
            },
            coords={
                "band": [1],
                "y": np.arange(10)[::-1],
                "x": np.arange(10),
            },
        ),
        xr.Dataset(
            {
                "raster": (("lat", "lon"), arange_nd((10, 10))),
                "mask": (
                    ("lat", "lon"),
                    np.zeros((10, 10), dtype=bool),
                ),
            },
            coords={
                "lat": np.arange(10)[::-1],
                "lon": np.arange(10),
            },
        ),
        xr.Dataset(
            {
                "raster": (("band", "lat", "lon"), arange_nd((2, 10, 10))),
                "mask": (
                    ("band", "lat", "lon"),
                    np.zeros((2, 10, 10), dtype=bool),
                ),
            },
            coords={
                "band": [1, 2],
                "lat": np.arange(10)[::-1],
                "lon": np.arange(10),
            },
        ),
        # Lazy
        xr.Dataset(
            {
                "raster": (("y", "x"), arange_nd((10, 10))),
                "mask": (("y", "x"), np.zeros((10, 10), dtype=bool)),
            },
            coords={"y": np.arange(10)[::-1], "x": np.arange(10)},
        ).chunk(),
        xr.Dataset(
            {
                "raster": (("y", "x"), arange_nd((10, 10), mod=da)),
                "mask": (("y", "x"), np.zeros((10, 10), dtype=bool)),
            },
            coords={"y": np.arange(10)[::-1], "x": np.arange(10)},
        ),
        xr.Dataset(
            {
                "raster": (("y", "x"), arange_nd((10, 10))),
                "mask": (("y", "x"), da.zeros((10, 10), dtype=bool)),
            },
            coords={"y": np.arange(10)[::-1], "x": np.arange(10)},
        ),
        # flipped y dim
        xr.Dataset(
            {
                "raster": (("band", "y", "x"), arange_nd((1, 10, 10))),
                "mask": (
                    ("band", "y", "x"),
                    np.zeros((1, 10, 10), dtype=bool),
                ),
            },
            coords={
                "band": [1],
                "y": np.arange(10),
                "x": np.arange(10),
            },
        ),
        # CRS
        xr.Dataset(
            {
                "raster": (("band", "lat", "lon"), arange_nd((2, 10, 10))),
                "mask": (
                    ("band", "lat", "lon"),
                    np.zeros((2, 10, 10), dtype=bool),
                ),
            },
            coords={
                "band": [1, 2],
                "lat": np.arange(10)[::-1],
                "lon": np.arange(10),
            },
        ).rio.write_crs("EPSG:3857"),
    ],
)
def test_raster_from_dataset(ds):
    raster = Raster(ds)
    assert raster._ds is not ds
    assert_valid_raster(raster)
    assert raster.dtype == ds.raster.dtype
    assert raster.null_value == get_default_null_value(ds.raster.dtype)
    assert raster.crs == ds.rio.crs
    ds = ds.compute()
    yc = "y" if "y" in ds.dims else "lat"
    xc = "x" if "x" in ds.dims else "lon"
    if is_strictly_increasing(ds[yc]):
        ds = ds.isel({yc: slice(None, None, -1)})

    if ds.mask.any():
        assert np.allclose(
            raster,
            xr.where(
                ds.mask, get_default_null_value(ds.raster.dtype), ds.raster
            ),
        )
    else:
        assert np.allclose(raster._ds.raster, ds.raster)
    assert np.allclose(raster._ds.mask, ds.mask)
    assert np.allclose(raster._ds.x, ds[xc])
    assert np.allclose(raster._ds.y, ds[yc])


@pytest.mark.parametrize(
    "path",
    [
        "tests/data/raster/dem.tif",
        "tests/data/raster/dem_clipped.tif",
    ],
)
def test_raster_from_file(path):
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert_valid_raster(raster)
    assert raster.dtype == xdata.dtype
    if xdata.rio.nodata is None:
        assert raster.null_value is None
    else:
        assert np.allclose(raster.null_value, xdata.rio.nodata)
    assert raster.shape == xdata.shape
    if xdata.rio.nodata is not None and np.isnan(xdata.rio.nodata):
        assert np.allclose(
            raster,
            xr.where(
                xdata.mask,
                get_default_null_value(xdata.dtype),
                xdata,
            ),
        )
    else:
        assert np.allclose(raster, xdata)


@pytest.mark.parametrize(
    "data,error_type",
    [
        (np.ones(4), ValueError),
        (np.array(9), ValueError),
        (np.ones(1), ValueError),
        (da.ones(4), ValueError),
        # dims without coords
        (xr.DataArray(np.ones((4, 4))), ValueError),
    ],
)
def test_raster_from_any_errors(data, error_type):
    with pytest.raises(error_type):
        Raster(data)


def test_raster_from_float_data_with_small_null_value():
    xdata = (
        xr.DataArray(
            da.ones((4, 4), dtype=float) * -9999.0,
            dims=("y", "x"),
            coords=([3, 2, 1, 0], [0, 1, 2, 3]),
        )
        # This nv can fit in a float16 but loses precision.
        .rio.write_nodata(-9_999.0)
    )
    raster = Raster(xdata)
    # Make sure that the value wasn't altered
    assert raster.null_value == np.float64(-9_999.0)
    assert np.allclose(raster.to_numpy(), -9_999.0)


@pytest.fixture
def raster_ones_missing_data():
    nv = -1
    data = np.ones((1, 6, 6), dtype=int)
    data[0, 0] = nv
    return rts.data_to_raster(data, nv=nv, crs="EPSG:3857").chunk((1, 3, 3))


@pytest.mark.parametrize(
    "op",
    [operator.add, operator.sub, operator.mul],
    ids=["add", "sub", "mult"],
)
@pytest.mark.parametrize(
    "use_raster_operand", [False, True], ids=["scalar", "raster"]
)
@pytest.mark.parametrize("dtype", [I8, I16, U8, U16])
def test_null_value_overflow_underflow(
    raster_ones_missing_data, dtype, use_raster_operand, op
):
    raster = raster_ones_missing_data.astype(
        dtype, new_null_value=get_default_null_value(dtype)
    )
    nv = get_default_null_value(dtype)

    mask = raster_ones_missing_data.mask.compute()
    pre_expected = raster_ones_missing_data.to_numpy()
    pre_expected = np.where(mask, np.nan, pre_expected)

    if use_raster_operand:
        operand = rts.data_to_raster(np.full(raster.shape, 2, dtype=dtype))
    else:
        operand = dtype.type(2)

    # Check if the operation causes the null values to overflow/underflow
    result = op(raster, operand)
    expected = op(pre_expected, operand)
    expected = np.where(mask, nv, expected).astype(dtype)
    assert np.allclose(result.to_numpy(), expected)


@pytest.mark.parametrize(
    "raster_input,masked",
    [
        ("tests/data/raster/dem_small.tif", True),
        ("tests/data/raster/dem_clipped_small.tif", True),
        (np.ones((1, 3, 3)), False),
        (np.array([np.nan, 3.2, 4, 5]).reshape((2, 2)), True),
        (
            riox.open_rasterio(
                "tests/data/raster/dem_clipped_small.tif"
            ).rio.write_nodata(None),
            False,
        ),
    ],
)
def test_property__masked(raster_input, masked):
    raster = Raster(raster_input)
    assert hasattr(raster, "_masked")
    assert raster._masked == masked


def test_property_values():
    raster = testdata.raster.dem_clipped_small
    assert hasattr(raster, "values")
    assert isinstance(raster.to_numpy(), np.ndarray)
    assert raster.shape == raster.to_numpy().shape
    assert np.allclose(raster.to_numpy(), raster._ds.raster.data.compute())


def test_property_null_value():
    path = "tests/data/raster/dem_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "null_value")
    assert raster.null_value == xdata.rio.nodata


def test_property_dtype():
    path = "tests/data/raster/dem_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "dtype")
    assert isinstance(raster.dtype, np.dtype)
    assert raster.dtype == xdata.dtype


def test_property_shape():
    path = "tests/data/raster/dem_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "shape")
    assert isinstance(raster.shape, tuple)
    assert raster.shape == xdata.shape


@pytest.mark.parametrize(
    "raster",
    [
        stack_bands([testdata.raster.dem_small] * 3),
        make_raster("arange", shape=(4, 20, 25), crs=None),
    ],
)
def test_property_size(raster):
    data = raster.data.compute()

    assert raster.size == data.size


def test_property_crs():
    path = "tests/data/raster/dem_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "crs")
    assert isinstance(raster.crs, rio.crs.CRS)
    assert raster.crs == xdata.rio.crs


def test_property_affine():
    path = "tests/data/raster/dem_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "affine")
    assert isinstance(raster.affine, affine.Affine)
    assert raster.affine == xdata.rio.transform()


def test_property_resolution():
    path = "tests/data/raster/dem_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "resolution")
    assert isinstance(raster.resolution, tuple)
    assert raster.resolution == xdata.rio.resolution()


def test_property_xdata():
    rs = testdata.raster.dem_small
    assert hasattr(rs, "xdata")
    assert isinstance(rs.xdata, xr.DataArray)
    assert rs.xdata.identical(rs._ds.raster)


def test_property_data():
    rs = testdata.raster.dem_small
    assert hasattr(rs, "data")
    assert isinstance(rs.data, da.Array)
    assert rs.data is rs._ds.raster.data


def test_property_bounds():
    path = "tests/data/raster/dem_small.tif"
    rs = Raster(path)
    rds = rio.open(path)
    assert hasattr(rs, "bounds")
    assert isinstance(rs.bounds, tuple)
    assert len(rs.bounds) == 4
    assert rs.bounds == tuple(rds.bounds)

    rs = Raster(
        np.array(
            [
                [0, 0],
                [0, 0],
            ]
        )
    )
    assert rs.bounds == (0, 0, 2, 2)


def test_property_geobox():
    raster = testdata.raster.dem
    assert hasattr(raster, "geobox")
    gb = raster.geobox
    assert gb.affine == raster.affine
    assert gb.shape == raster.shape[1:]
    assert gb.crs == raster.crs
    assert gb.resolution.xy == raster.resolution


def test_property_mask():
    rs = (
        make_raster("arange", shape=(10, 10), null_pattern="%4", crs=None) % 4
    ).set_null_value(0)
    assert rs._ds.mask.data.sum().compute() > 0

    assert hasattr(rs, "mask")
    assert isinstance(rs.mask, dask.array.Array)
    assert np.allclose(rs.mask.compute(), rs._ds.mask.data.compute())


def test_property_xmask():
    rs = (
        make_raster("arange", shape=(10, 10), null_pattern="%4") % 4
    ).set_null_value(0)
    assert rs.mask.sum().compute() > 0

    assert hasattr(rs, "xmask")
    assert isinstance(rs.xmask, xr.DataArray)
    assert np.allclose(rs.xmask.data.compute(), rs._ds.mask.data.compute())
    assert rs.xmask.rio.crs == rs.crs
    assert np.allclose(rs.xmask.x.data, rs.xdata.x.data)
    assert np.allclose(rs.xmask.y.data, rs.xdata.y.data)


@pytest.mark.parametrize("name", ["band", "x", "y"])
def test_properties_coords(name):
    rs = testdata.raster.dem_small

    assert hasattr(rs, name)
    assert isinstance(getattr(rs, name), np.ndarray)
    assert np.allclose(getattr(rs, name), rs.xdata[name].data)


def test_to_numpy():
    raster = testdata.raster.dem_clipped_small
    assert hasattr(raster, "values")
    assert isinstance(raster.to_numpy(), np.ndarray)
    assert raster.shape == raster.to_numpy().shape
    assert np.allclose(raster.to_numpy(), raster._ds.raster.data.compute())


@pytest.mark.parametrize(
    "rs",
    [
        make_raster(
            "ones", shape=(4, 100, 100), chunksize=(1, 5, 5), crs=None
        ),
        make_raster("ones", shape=(100, 100), chunksize=(5, 5), crs=None),
        make_raster("ones", shape=(100, 100), crs=None),
        testdata.raster.dem,
    ],
)
def test_get_chunked_coords(rs):
    xc, yc = rs.get_chunked_coords()

    assert isinstance(xc, da.Array)
    assert isinstance(yc, da.Array)
    assert xc.dtype == rs._ds.x.dtype
    assert yc.dtype == rs._ds.y.dtype
    assert xc.ndim == 3
    assert yc.ndim == 3
    assert xc.shape == (1, 1, rs._ds.x.size)
    assert yc.shape == (1, rs._ds.y.size, 1)
    assert xc.chunks[:2] == ((1,), (1,))
    assert xc.chunks[2] == rs.data.chunks[2]
    assert yc.chunks[::2] == ((1,), (1,))
    assert yc.chunks[1] == rs.data.chunks[1]
    assert np.allclose(xc.ravel().compute(), rs._ds.x.data)
    assert np.allclose(yc.ravel().compute(), rs._ds.y.data)


_BINARY_ARITHMETIC_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.pow,
    operator.truediv,
    operator.floordiv,
    operator.mod,
]
_BINARY_COMPARISON_OPS = [
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.eq,
    operator.ne,
]


@pytest.fixture
def ops_test_array_data():
    data = arange_nd((4, 5, 5))
    data = np.where((data >= 0) & (data < 10), 0, data)
    return data


def _safe_operand(operand, raster_dtype):
    """Scalar to use with the raw numpy data to get a raster op's result.

    Float rasters follow numpy's promotion for Python scalars, so the scalar
    is used as is. Under numpy 2, integer and bool rasters are promoted as if
    the scalar were a 64-bit numpy scalar, which prevents wraparound.
    """
    if (
        NUMPY_GE_2
        and not is_float(raster_dtype)
        and type(operand) in (int, float)
    ):
        return (
            np.int64(operand) if type(operand) is int else np.float64(operand)
        )
    return operand


def _assert_binary_result(result, expected_np, mask, raster):
    """Common assertions for binary op results."""
    assert_valid_raster(result)
    assert_rasters_similar(result, raster)
    assert result._masked
    assert result.dtype == expected_np.dtype
    assert np.allclose(result, expected_np, equal_nan=True)
    assert np.allclose(result.mask.compute(), mask)


def _apply_null_to_expected(expected_np, mask):
    return np.where(
        mask, get_default_null_value(expected_np.dtype), expected_np
    )


@pytest.mark.parametrize(
    "op",
    (operator.add, operator.mul, operator.truediv, operator.floordiv),
)
@pytest.mark.parametrize("operand", [-1, 0, 3.0])
@pytest.mark.parametrize("raster_type", [F32, F64, I32, I8, U16])
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_simple_binary_ops_arithmetic_against_scalar(
    ops_test_array_data, op, operand, raster_type
):
    x = ops_test_array_data.astype(raster_type)
    raster = Raster(x).set_null_value(0).set_crs("EPSG:3857")
    mask = raster.mask.compute()
    safe = _safe_operand(operand, x.dtype)

    # Forward: raster op scalar
    expected = _apply_null_to_expected(op(x, safe), mask)
    _assert_binary_result(op(raster, operand), expected, mask, raster)

    # Reflected: scalar op raster
    expected = _apply_null_to_expected(op(safe, x), mask)
    _assert_binary_result(op(operand, raster), expected, mask, raster)


@pytest.mark.parametrize("operand", [-1, 0, 2, 3.0])
@pytest.mark.parametrize("raster_type", [F32, F64, I32, U16])
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:overflow")
def test_binary_op_pow_against_scalar(
    ops_test_array_data, operand, raster_type
):
    x = ops_test_array_data.astype(raster_type)
    raster = Raster(x).set_null_value(0).set_crs("EPSG:3857")
    mask = raster.mask.compute()
    safe = _safe_operand(operand, x.dtype)
    # In numpy 2, np.power and operator.pow sometimes produce different
    # output dtypes. __array_ufunc__ on Raster uses np.power so we test
    # against that API.
    safe_power = np.power if NUMPY_GE_2 else operator.pow

    if (
        raster_type != U64
        and is_int(raster_type)
        and is_int(operand)
        and operand < 0
    ):
        with pytest.raises(TypeError):
            operator.pow(raster, operand)
        with pytest.raises(ValueError):
            safe_power(x, safe)
    else:
        expected = _apply_null_to_expected(safe_power(x, safe), mask)
        _assert_binary_result(
            operator.pow(raster, operand), expected, mask, raster
        )

    # Reflected
    expected = _apply_null_to_expected(safe_power(safe, x), mask)
    _assert_binary_result(
        operator.pow(operand, raster), expected, mask, raster
    )


@pytest.mark.parametrize("op", [operator.lt, operator.eq, operator.ge])
@pytest.mark.parametrize("operand", [-1, 0, 3.0])
@pytest.mark.parametrize("raster_type", [F32, I32, U16])
def test_binary_comparison_ops_against_scalar(
    ops_test_array_data, op, operand, raster_type
):
    x = ops_test_array_data.astype(raster_type)
    raster = Raster(x).set_null_value(0).set_crs("EPSG:3857")
    mask = raster.mask.compute()
    safe = _safe_operand(operand, x.dtype)

    for args, np_args in [
        ((raster, operand), (x, safe)),
        ((operand, raster), (safe, x)),
    ]:
        expected = _apply_null_to_expected(op(*np_args), mask)
        result = op(*args)
        _assert_binary_result(result, expected, mask, raster)
        # TODO(bool): update once we switch to int8 for boolean results
        assert is_bool(result.dtype)
        assert is_bool(result.null_value)
        assert result.null_value == get_default_null_value(bool)


@pytest.mark.parametrize(
    "raster_type,op,operand,expected_type",
    [
        (F32, operator.mul, 2.0, F32),
        (F32, operator.add, 1, F32),
        (F32, operator.truediv, 3, F32),
        (F16, operator.add, 1.5, F16),
        (F64, operator.add, 1, F64),
        (U8, operator.add, 1, I64),
        (U16, operator.mul, 10, I64),
        (I8, operator.sub, 1, I64),
        (U8, operator.mul, 1.5, F64),
        (I32, operator.mul, 2.0, F64),
        (BOOL, operator.add, 1, I64),
        (np.complex64, operator.add, 1.0, np.complex64),
        (np.complex64, operator.mul, 2, np.complex64),
        (np.complex128, operator.sub, 1.5, np.complex128),
    ],
)
def test_binary_op_python_scalar_promotion(
    raster_type, op, operand, expected_type
):
    data = np.ones((1, 3, 3), dtype=raster_type)
    raster = Raster(data).set_crs("EPSG:3857")
    if NUMPY_GE_2:
        assert op(raster, operand).dtype == expected_type
        assert op(operand, raster).dtype == expected_type
    else:
        assert op(raster, operand).dtype == op(data, operand).dtype
        assert op(operand, raster).dtype == op(operand, data).dtype


@pytest.mark.skipif(not NUMPY_GE_2, reason="numpy 2 promotion rules")
def test_binary_op_python_scalar_no_wraparound():
    data = np.full((1, 3, 3), 60_000, dtype=U16)
    raster = Raster(data)
    result = raster * 10
    assert np.all(result.to_numpy() == 600_000)


unknown_chunk_array = dask.array.ones((5, 5))
unknown_chunk_array = unknown_chunk_array[unknown_chunk_array > 0]


@pytest.mark.parametrize("op", [operator.add, operator.mul, operator.lt])
@pytest.mark.parametrize(
    "other",
    [
        [1],
        np.array([1]),
        np.array([[1]]),
        np.ones((5, 5)),
        np.ones((1, 5, 5)),
        np.ones((4, 5, 5)),
        dask.array.ones((5, 5)),
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:elementwise comparison failed")
def test_binary_ops_against_array(ops_test_array_data, op, other):
    x = ops_test_array_data
    raster = Raster(x).set_null_value(0).set_crs("EPSG:3857")
    mask = raster.mask.compute()
    data = raster.to_numpy()

    for raster_args, np_args in [
        ((raster, other), (data, other)),
        ((other, raster), (other, data)),
    ]:
        truth = op(*np_args)
        (truth,) = dask.compute(truth)
        truth[mask] = get_default_null_value(truth.dtype)
        result = op(*raster_args)
        assert_valid_raster(result)
        assert_rasters_similar(result, raster)
        assert result.crs == raster.crs
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._ds.mask.compute(), mask)


@pytest.mark.parametrize(
    "other",
    [np.zeros(4), np.array([1, 1]), unknown_chunk_array],
)
def test_binary_ops_against_array_invalid_shape(ops_test_array_data, other):
    raster = Raster(ops_test_array_data).set_null_value(0)
    with pytest.raises(ValueError):
        raster + other
    with pytest.raises(ValueError):
        other + raster


@pytest.mark.parametrize("op", [operator.add, operator.truediv, operator.eq])
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_binary_ops_arithmetic_against_raster(ops_test_array_data, op):
    x = ops_test_array_data
    raster = Raster(x).set_null_value(0).set_crs("EPSG:3857")
    mask = raster.mask.compute()
    data = raster.to_numpy()
    data2 = np.ones_like(data, dtype=int) * 2
    raster2 = Raster(data2).set_crs("EPSG:3857")

    for raster_args, np_args in [
        ((raster, raster2), (data, data2)),
        ((raster2, raster), (data2, data)),
    ]:
        truth = op(*np_args)
        truth[mask] = get_default_null_value(truth.dtype)
        result = op(*raster_args)
        assert_valid_raster(result)
        assert_rasters_similar(result, raster)
        assert result._masked
        assert result.dtype == truth.dtype
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._ds.mask, raster._ds.mask)


@pytest.mark.parametrize(
    "iop,np_op",
    [
        (operator.iadd, operator.add),
        (operator.isub, operator.sub),
        (operator.imul, operator.mul),
        (operator.ipow, operator.pow),
        (operator.itruediv, operator.truediv),
        (operator.ifloordiv, operator.floordiv),
        (operator.imod, operator.mod),
    ],
)
def test_binary_ops_arithmetic_inplace(iop, np_op):
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5))
    rs = Raster(data).set_crs("EPSG:3857")

    r = rs.copy()
    rr = r
    r = iop(r, 3)
    t = np_op(data, 3)
    assert np.allclose(r, t)
    assert rr is r
    assert rr.crs == rs.crs


_UNSUPPORED_UFUNCS = [np.isnat, np.matmul]
if NUMPY_GE_2:
    _UNSUPPORED_UFUNCS.append(np.vecdot)
if NUMPY_GE_2_2:
    _UNSUPPORED_UFUNCS.extend([np.matvec, np.vecmat])
_UNSUPPORED_UFUNCS = tuple(_UNSUPPORED_UFUNCS)
# Representative ufuncs covering distinct code paths in _apply_ufunc.
# Single-input: nout==1 (sqrt, abs, sin, isfinite, signbit, bitwise_count),
#               nout==2 (frexp, modf)
_NP_UFUNCS_NIN_SINGLE = [
    np.sqrt,
    np.absolute,
    np.sin,
    np.isfinite,
    np.signbit,
    np.frexp,
    np.modf,
]
if NUMPY_GE_2:
    _NP_UFUNCS_NIN_SINGLE.append(np.bitwise_count)
# Multi-input: standard (add, multiply, less), nout==2 (divmod),
#              bitwise TypeError (bitwise_and), shift TypeError (left_shift),
#              int-only (gcd), special casting (ldexp)
_NP_UFUNCS_NIN_MULT = [
    np.add,
    np.multiply,
    np.less,
    np.divmod,
    np.bitwise_and,
    np.left_shift,
    np.gcd,
    np.ldexp,
]


@pytest.mark.parametrize("ufunc", _UNSUPPORED_UFUNCS)
def test_ufuncs_unsupported(ufunc):
    rs = make_raster("arange", shape=(4, 5, 5), crs=None)
    args = [rs for i in range(ufunc.nin)]
    with pytest.raises(
        NotImplementedError if ufunc.__name__ == "vecdot" else TypeError
    ):
        ufunc(*args)


@pytest.mark.parametrize(
    "call,types",
    [
        (lambda rs: rs - rs, "Raster<bool> and Raster<bool>"),
        (lambda rs: rs - True, "Raster<bool> and <class 'bool'>"),
        (lambda rs: True - rs, "<class 'bool'> and Raster<bool>"),
    ],
)
def test_ufuncs_no_loop_for_types_error(call, types):
    rs = Raster(np.ones((1, 2, 2), dtype=BOOL))
    match = re.escape(f"Could not apply <ufunc 'subtract'> to types {types}")
    with pytest.raises(TypeError, match=match):
        call(rs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": F32},
        {"where": True},
        {"casting": "unsafe"},
        {"order": "C"},
        {"subok": False},
        {"signature": (F64, F64, F64)},
    ],
)
def test_ufuncs_reject_keyword_args(kwargs):
    rs = make_raster("arange", shape=(2, 3, 3), dtype=float)
    (name,) = kwargs
    with pytest.raises(TypeError, match=f"'{name}'"):
        np.add(rs, 1, **kwargs)
    with pytest.raises(TypeError, match=f"'{name}'"):
        np.add(1, rs, **kwargs)
    with pytest.raises(TypeError, match=f"'{name}'"):
        np.add(rs.bandwise, [1, 2], **kwargs)


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_SINGLE)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_ufuncs_single_input(ufunc):
    nv = 0
    raster = make_raster("arange", shape=(4, 5, 5), dtype=float, null=nv)
    mask = raster.mask.compute()
    assert raster.null_value == nv
    assert raster._masked
    assert raster.crs == "EPSG:3857"
    data = raster.to_numpy()

    # bitwise_count added in numpy 2.0
    if ufunc.__name__ in ("invert", "bitwise_count") and is_float(data.dtype):
        with pytest.raises(TypeError):
            ufunc(raster)
        return
    truth = ufunc(data)
    result = ufunc(raster)
    if ufunc.nout == 1:
        assert_valid_raster(result)
        truth[mask] = get_default_null_value(truth.dtype)
        assert result._masked
        assert result.dtype == truth.dtype
        assert result.crs == raster.crs
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._ds.mask, raster._ds.mask, equal_nan=True)
    else:
        for r, t in zip(result, truth, strict=True):
            assert_valid_raster(r)
            t[mask] = get_default_null_value(t.dtype)
            assert r._masked
            assert r.dtype == t.dtype
            assert r.crs == raster.crs
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._ds.mask, raster._ds.mask, equal_nan=True)


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_MULT)
@pytest.mark.parametrize("dtype", [int, float])
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_ufuncs_multiple_input_against_scalar(
    ops_test_array_data, ufunc, dtype
):
    x = ops_test_array_data.astype(dtype)
    nv = 0
    raster = Raster(x).set_null_value(nv).set_crs("EPSG:3857")
    mask = raster.mask.compute()
    assert raster.null_value == nv
    assert raster._masked
    assert raster.crs == "EPSG:3857"
    extra = ufunc.nin - 1

    args = [raster] + [2 for i in range(extra)]
    args_np = [getattr(a, "values", a) for a in args]

    ufname = ufunc.__name__
    # These should all fail for regular and reflected cases
    if (
        ufname.startswith("bitwise")
        or ufname.endswith("shift")
        or ufname in ("gcd", "lcm")
    ) and any(is_float(getattr(a, "dtype", a)) for a in args):
        with pytest.raises(TypeError):
            ufunc(*args)
        with pytest.raises(TypeError):
            ufunc(*args_np)
        with pytest.raises(TypeError):
            ufunc(*args[::-1])
        with pytest.raises(TypeError):
            ufunc(*args_np[::-1])
        return

    expected_np = ufunc(*args_np)
    result = ufunc(*args)
    if ufunc.nout == 1:
        assert_valid_raster(result)
        assert_rasters_similar(result, raster)
        expected_np[mask] = get_default_null_value(expected_np.dtype)
        assert result._masked
        assert result.dtype == expected_np.dtype
        assert result.crs == raster.crs
        assert np.allclose(result, expected_np, equal_nan=True)
        assert np.allclose(result._ds.mask, raster._ds.mask, equal_nan=True)
    else:
        for r, t in zip(result, expected_np, strict=True):
            assert_valid_raster(r)
            assert_rasters_similar(r, raster)
            t[mask] = get_default_null_value(t.dtype)
            assert r._masked
            assert r.dtype == t.dtype
            assert r.crs == raster.crs
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._ds.mask, raster._ds.mask, equal_nan=True)

    # Reflected
    args = args[::-1]
    args_np = args_np[::-1]
    # Should only fail for reflected
    if ufname == "ldexp" and is_float(args[-1].dtype):
        with pytest.raises(TypeError):
            ufunc(*args)
        return
    expected_np = ufunc(*args_np)
    result = ufunc(*args)
    if ufunc.nout == 1:
        assert_valid_raster(result)
        assert_rasters_similar(result, raster)
        expected_np[mask] = get_default_null_value(expected_np.dtype)
        assert result._masked
        assert result.dtype == expected_np.dtype
        assert np.allclose(result, expected_np, equal_nan=True)
        assert np.allclose(result._ds.mask, raster._ds.mask, equal_nan=True)
    else:
        for r, t in zip(result, expected_np, strict=True):
            assert_valid_raster(r)
            assert_rasters_similar(r, raster)
            t[mask] = get_default_null_value(t.dtype)
            assert r._masked
            assert r.dtype == t.dtype
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._ds.mask, raster._ds.mask, equal_nan=True)


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_MULT)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_ufuncs_multiple_input_against_raster(ops_test_array_data, ufunc):
    x = ops_test_array_data
    nv = 0
    raster = Raster(x).set_null_value(nv).set_crs("EPSG:3857")
    mask = raster.mask.compute()
    assert raster.null_value == nv
    assert raster._masked
    assert raster.crs == "EPSG:3857"
    extra = ufunc.nin - 1
    other = make_raster(np.ones_like(x) * 2, crs=None)
    other.xdata.data[0, 0, 0] = 25
    other = other.set_null_value(25)
    assert other._ds.mask.data.compute().sum() == 1
    mask = raster._ds.mask | other._ds.mask

    args = [raster] + [other.copy() for i in range(extra)]
    args_np = [a.to_numpy() for a in args]

    if ufunc.__name__.startswith("bitwise") and any(
        is_float(a.dtype) for a in args
    ):
        with pytest.raises(TypeError):
            ufunc(*args)
        with pytest.raises(TypeError):
            ufunc(*args[::-1])
        return

    expected_np = ufunc(*args_np)
    result = ufunc(*args)
    if ufunc.nout == 1:
        assert_valid_raster(result)
        expected_np[mask] = get_default_null_value(expected_np.dtype)
        assert result._masked
        assert result.dtype == expected_np.dtype
        assert result.crs == raster.crs
        assert np.allclose(result, expected_np, equal_nan=True)
        assert np.allclose(result._ds.mask, mask, equal_nan=True)
    else:
        for r, t in zip(result, expected_np, strict=True):
            assert_valid_raster(r)
            t[mask] = get_default_null_value(t.dtype)
            assert r._masked
            assert r.dtype == t.dtype
            assert r.crs == raster.crs
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._ds.mask, mask, equal_nan=True)

    # Test reflected
    args = args[::-1]
    args_np = args_np[::-1]
    expected_np = ufunc(*args_np)
    result = ufunc(*args)
    if ufunc.nout == 1:
        assert_valid_raster(result)
        expected_np[mask] = get_default_null_value(expected_np.dtype)
        assert result.crs == raster.crs
        assert np.allclose(result, expected_np, equal_nan=True)
        assert np.allclose(result._ds.mask, mask)
    else:
        for r, t in zip(result, expected_np, strict=True):
            assert_valid_raster(r)
            t[mask] = get_default_null_value(t.dtype)
            assert r.crs == raster.crs
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._ds.mask, mask, equal_nan=True)


@pytest.mark.parametrize(
    "ufunc,args",
    [
        (np.modf, ()),
        (np.frexp, ()),
        (np.divmod, (3,)),
    ],
)
@pytest.mark.parametrize("null", [None, 0])
def test_ufuncs_multiple_outputs(ufunc, args, null):
    rs = make_raster("arange", shape=(2, 3, 3), dtype=float, null=null)
    rs = rs / 4
    data = rs.to_numpy()
    mask = rs.mask.compute()
    results = ufunc(rs, *args)
    expected = ufunc(data, *args)
    assert isinstance(results, tuple)
    assert len(results) == ufunc.nout
    for result, truth in zip(results, expected, strict=True):
        assert_valid_raster(result)
        assert_rasters_similar(result, rs)
        assert result.dtype == truth.dtype
        assert result._masked == (null is not None)
        assert np.array_equal(result.mask.compute(), mask)
        assert np.array_equal(result.to_numpy()[~mask], truth[~mask])
        if null is not None:
            assert np.all(
                result.to_numpy()[mask] == get_default_null_value(truth.dtype)
            )


def test_ufunc_different_masks():
    r1 = make_raster("arange", shape=(3, 3), crs=None).set_null_value(0)
    r1.mask[..., :2, :] = True
    r2 = make_raster("arange", shape=(3, 3), crs=None).set_null_value(0)
    r2.mask[..., :, :2] = True
    r1_mask = np.array(
        [
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 0],
            ]
        ]
    )
    r2_mask = np.array(
        [
            [
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 0],
            ]
        ]
    )
    result_mask = r1_mask | r2_mask
    assert np.allclose(r1.mask.compute(), r1_mask)
    assert np.allclose(r2.mask.compute(), r2_mask)
    assert np.allclose(
        result_mask, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0]])
    )

    result = r1 * r2
    # Make sure that there where no side effects on the input raster masks
    assert np.allclose(r1.mask.compute(), r1_mask)
    assert np.allclose(r2.mask.compute(), r2_mask)
    result_data = np.where(result_mask, result.null_value, 64)
    assert np.allclose(result, result_data)


def _grid_raster(x0, y0=4, res=1, crs="EPSG:3857", null=None):
    data = np.arange(1, 17, dtype=U8).reshape((4, 4))
    rs = make_raster(data, affine=Affine(res, 0, x0, 0, -res, y0), crs=crs)
    if null is not None:
        rs = rs.set_null_value(null)
    return rs


@pytest.mark.parametrize("left_null", [None, 6])
@pytest.mark.parametrize("right_null", [None, 2])
def test_ufunc_rasters_offset_by_whole_cells(left_null, right_null):
    # right is shifted one cell right and one cell down from left
    left = _grid_raster(0, null=left_null)
    right = _grid_raster(1, 3, null=right_null)
    left_data = left.to_numpy()[:, 1:, 1:]
    right_data = right.to_numpy()[:, :3, :3]
    mask = np.zeros((1, 3, 3), dtype=bool)
    if left_null is not None:
        mask |= left_data == left_null
    if right_null is not None:
        mask |= right_data == right_null

    for result, expected in [
        (left + right, left_data + right_data),
        (right - left, right_data - left_data),
    ]:
        assert_valid_raster(result)
        assert result.shape == (1, 3, 3)
        assert np.array_equal(result.x, left.x[1:])
        assert np.array_equal(result.y, left.y[1:])
        assert result.crs == left.crs
        assert result._masked == (
            left_null is not None or right_null is not None
        )
        assert np.array_equal(result.mask.compute(), mask)
        assert np.array_equal(result.to_numpy()[~mask], expected[~mask])


def test_ufunc_rasters_offset_within_tolerance():
    left = _grid_raster(0, null=6)
    right = _grid_raster(1 + 1e-6, 4 - 1e-6)
    result = left * right
    assert_valid_raster(result)
    assert result.shape == (1, 4, 3)
    assert np.array_equal(result.x, left.x[1:])
    assert np.array_equal(result.y, left.y)
    expected = left.to_numpy()[:, :, 1:] * right.to_numpy()[:, :, :3]
    mask = left.mask.compute()[:, :, 1:]
    assert np.array_equal(result.mask.compute(), mask)
    assert np.array_equal(result.to_numpy()[~mask], expected[~mask])


def test_ufunc_rasters_missing_crs_matches_any_crs():
    left = _grid_raster(0, crs=None)
    right = _grid_raster(1)
    assert (left + right).crs == right.crs
    assert (right + left).crs == right.crs


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "right_grid,match",
    [
        ({"x0": 0.5}, "non-integer number of cells"),
        ({"x0": 0, "y0": 3.5}, "non-integer number of cells"),
        ({"x0": 0, "res": 2}, "resolution differs"),
        ({"x0": 0, "crs": "EPSG:4326"}, "CRS differs"),
        ({"x0": 1, "crs": "EPSG:4326"}, "CRS differs"),
        ({"x0": 4}, "do not overlap"),
        ({"x0": -1, "y0": 8}, "do not overlap"),
    ],
)
def test_ufunc_rasters_grid_mismatch_errors(right_grid, match, masked):
    left = _grid_raster(0, null=6 if masked else None)
    right = _grid_raster(**right_grid)
    with pytest.raises(ValueError, match=match):
        left + right
    with pytest.raises(ValueError, match=match):
        right + left
    with pytest.raises(ValueError) as excinfo:
        np.add(left, right)
    # align cannot help rasters that do not overlap, so only the other
    # errors point at it.
    hint = "align([raster, other])"
    assert (hint in str(excinfo.value)) == (match != "do not overlap")


def test_ufunc_rasters_no_overlap_error_reports_bounds():
    left = _grid_raster(0)
    right = _grid_raster(4)
    match = (
        r"do not overlap.*\(0\.0, 0\.0, 4\.0, 4\.0\) and"
        r" \(4\.0, 0\.0, 8\.0, 4\.0\)\.$"
    )
    with pytest.raises(ValueError, match=match):
        left + right


def _strip_raster(shape, x0, res, null=None):
    data = np.arange(1, np.prod(shape) + 1, dtype=F64).reshape(shape)
    rs = make_raster(data, affine=Affine(res, 0, x0, 0, -res, 8))
    if null is not None:
        rs = rs.set_null_value(null)
    return rs


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "rebuild",
    [
        lambda rs: rs,
        lambda rs: rs.set_null_value(3),
        lambda rs: rs.where(rs > 0, 0),
        lambda rs: Raster(rs.xdata),
    ],
    ids=["as_built", "set_null_value", "where", "from_dataarray"],
)
def test_ufunc_rasters_one_cell_wide_on_lattice(masked, rebuild):
    # The column's single x coordinate (3) is the second cell center of the
    # grid (1, 3, 5, 7). The result must not depend on how the column was
    # produced.
    column = rebuild(_strip_raster((4, 1), 2, 2, null=3 if masked else None))
    grid = _strip_raster((4, 4), 0, 2)
    assert np.array_equal(column.x, [3])
    expected = column.to_numpy() + grid.to_numpy()[:, :, 1:2]
    column_mask = column.mask.compute()
    for result in (column + grid, grid + column):
        assert_valid_raster(result)
        assert result.shape == (1, 4, 1)
        assert np.array_equal(result.x, column.x)
        assert np.array_equal(result.y, grid.y)
        assert result._masked == column._masked
        mask = result.mask.compute()
        assert np.array_equal(mask, column_mask)
        assert np.array_equal(result.to_numpy()[~mask], expected[~mask])


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "left_grid,right_grid,match",
    [
        # The column's x coordinate (2) falls between the grid's cell centers
        (((4, 1), 1, 2), ((4, 4), 0, 2), "non-integer number of cells"),
        # One cell each, with different cell centers
        (((1, 1), 0, 2), ((1, 1), 2, 2), "cell size is unknown"),
        (((1, 1), 0, 2), ((1, 1), 0.5, 2), "cell size is unknown"),
    ],
)
def test_ufunc_rasters_one_cell_wide_mismatch_errors(
    left_grid, right_grid, match, masked
):
    left = _strip_raster(*left_grid, null=1 if masked else None)
    right = _strip_raster(*right_grid)
    with pytest.raises(ValueError, match=match):
        left + right
    with pytest.raises(ValueError, match=match):
        right + left


def test_ufunc_rasters_one_cell_float_noise():
    left = _strip_raster((1, 1), 0, 2, null=9)
    noisy = _strip_raster((1, 1), 1e-12, 2)
    assert not np.array_equal(left.x, noisy.x)
    expected = left.to_numpy() + noisy.to_numpy()
    for result, first in [(left + noisy, left), (noisy + left, noisy)]:
        assert_valid_raster(result)
        assert result.shape == (1, 1, 1)
        assert np.array_equal(result.x, first.x)
        assert np.array_equal(result.y, first.y)
        assert np.array_equal(result.to_numpy(), expected)
    shifted = _strip_raster((1, 1), 1e-3, 2)
    with pytest.raises(ValueError, match="cell size is unknown"):
        left + shifted


def _long_raster(res):
    data = np.ones((2, 5000), dtype=F32)
    return make_raster(data, affine=Affine(res, 0, 0, 0, -1, 2))


def test_ufunc_rasters_resolution_drift_errors():
    # The cell sizes differ by 0.05%, which drifts by 2.5 cells across 5000
    # cells.
    left = _long_raster(1.0)
    right = _long_raster(1.0005)
    with pytest.raises(ValueError, match="resolution differs along x"):
        left + right
    with pytest.raises(ValueError, match="resolution differs along x"):
        right + left


def test_ufunc_rasters_resolution_within_drift_tolerance():
    left = _long_raster(1.0)
    right = _long_raster(1.0 + 1e-9)
    result = left + right
    assert_valid_raster(result)
    assert result.shape == left.shape
    assert np.array_equal(result.x, left.x)


@pytest.mark.parametrize("masked", [False, True])
def test_ufunc_rasters_flipped_axis_errors(masked):
    left = _grid_raster(0, null=6 if masked else None)
    flipped = Raster(left._ds.isel(y=slice(None, None, -1)))
    assert np.array_equal(flipped.y, left.y[::-1])
    with pytest.raises(ValueError, match="y axis orientation differs"):
        left + flipped
    with pytest.raises(ValueError, match="y axis orientation differs"):
        flipped + left


def _band_raster(nbands, null=None):
    data = np.arange(1, nbands * 16 + 1, dtype=F64).reshape((nbands, 4, 4))
    rs = make_raster(data)
    if null is not None:
        rs = rs.set_null_value(null)
    return rs


@pytest.mark.parametrize("multi_null", [None, 20])
@pytest.mark.parametrize("single_null", [None, 2])
def test_ufunc_rasters_single_band_broadcasts(multi_null, single_null):
    multi = _band_raster(3, null=multi_null)
    single = _band_raster(1, null=single_null)
    multi_data = multi.to_numpy()
    single_data = single.to_numpy()
    mask = multi.mask.compute() | single.mask.compute()
    for result, expected in [
        (multi - single, multi_data - single_data),
        (single - multi, single_data - multi_data),
    ]:
        assert_valid_raster(result)
        assert result.shape == (3, 4, 4)
        assert np.array_equal(result.band, [1, 2, 3])
        assert result._masked == (
            multi_null is not None or single_null is not None
        )
        assert np.array_equal(result.mask.compute(), mask)
        assert np.array_equal(result.to_numpy()[~mask], expected[~mask])


@pytest.mark.parametrize("masked", [False, True])
def test_ufunc_rasters_bands_match_by_position(masked):
    left = _band_raster(2, null=5 if masked else None)
    # Band labels other than 1..n are not standard, but they can reach a
    # ufunc through the Dataset constructor.
    right = Raster(_band_raster(2, null=20)._ds.assign_coords(band=[3, 4]))
    assert np.array_equal(right.band, [3, 4])
    mask = left.mask.compute() | right.mask.compute()
    assert_valid_raster(left + right)
    for result, expected, band in [
        (left + right, left.to_numpy() + right.to_numpy(), [1, 2]),
        (right + left, right.to_numpy() + left.to_numpy(), [3, 4]),
    ]:
        assert result.shape == (2, 4, 4)
        assert np.array_equal(result.band, band)
        assert np.array_equal(result.mask.compute(), mask)
        assert np.array_equal(result.to_numpy()[~mask], expected[~mask])


@pytest.mark.parametrize("masked", [False, True])
def test_ufunc_rasters_band_count_mismatch_errors(masked):
    left = _band_raster(3, null=5 if masked else None)
    right = _band_raster(2)
    with pytest.raises(ValueError, match=r"band counts do not match \(3 vs 2"):
        left + right
    with pytest.raises(ValueError, match=r"band counts do not match \(2 vs 3"):
        right + left


def test_ufunc_rasters_inplace_keeps_grid():
    rs = _band_raster(3, null=5)
    single = _band_raster(1)
    expected = rs.to_numpy() + single.to_numpy()
    mask = rs.mask.compute()
    result = rs
    result += single
    assert result is rs
    assert_valid_raster(result)
    assert result.shape == (3, 4, 4)
    assert np.array_equal(result.mask.compute(), mask)
    assert np.array_equal(result.to_numpy()[~mask], expected[~mask])


@pytest.mark.parametrize(
    "make_other",
    [
        lambda: _grid_raster(1),
        lambda: _grid_raster(0, 3),
        lambda: _band_raster(3),
    ],
    ids=["offset_x", "offset_y", "more_bands"],
)
def test_ufunc_rasters_inplace_grid_change_errors(make_other):
    rs = _grid_raster(0, null=6)
    ds = rs._ds
    with pytest.raises(ValueError, match="in-place operation cannot change"):
        rs += make_other()
    assert rs._ds is ds


def test_ufunc_rasters_inplace_against_containing_raster():
    rs = _grid_raster(1, 3, null=6)
    # A 6x6 grid whose cells 1..4 along each axis are rs's cells
    big_data = np.arange(100, 136, dtype=U8).reshape((6, 6))
    big = make_raster(big_data, affine=Affine(1, 0, 0, 0, -1, 4))
    x, y = rs.x, rs.y
    expected = rs.to_numpy() + big_data[1:5, 1:5]
    mask = rs.mask.compute()
    result = rs
    result += big
    assert result is rs
    assert_valid_raster(result)
    assert result.shape == (1, 4, 4)
    assert np.array_equal(result.x, x)
    assert np.array_equal(result.y, y)
    assert np.array_equal(result.mask.compute(), mask)
    assert np.array_equal(result.to_numpy()[~mask], expected[~mask])


@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_ufunc_rasters_multiple_outputs_offset_grids_and_bands():
    multi_data = np.arange(1, 49, dtype=F64).reshape((3, 4, 4))
    multi = make_raster(
        multi_data, affine=Affine(1, 0, 0, 0, -1, 4)
    ).set_null_value(20)
    # One cell right of and one cell below multi
    single_data = np.arange(2, 18, dtype=F64).reshape((1, 4, 4)) % 5
    single = make_raster(
        single_data, affine=Affine(1, 0, 1, 0, -1, 3)
    ).set_null_value(0)
    multi_crop = multi_data[:, 1:, 1:]
    single_crop = single_data[:, :3, :3]
    mask = (multi_crop == 20) | (single_crop == 0)
    assert mask.any()
    for results, expected in [
        (np.divmod(multi, single), np.divmod(multi_crop, single_crop)),
        (np.divmod(single, multi), np.divmod(single_crop, multi_crop)),
    ]:
        assert len(results) == 2
        for result, truth in zip(results, expected, strict=True):
            assert_valid_raster(result)
            assert result.shape == (3, 3, 3)
            assert np.array_equal(result.x, multi.x[1:])
            assert np.array_equal(result.y, multi.y[1:])
            assert result._masked
            assert np.array_equal(result.mask.compute(), mask)
            assert np.array_equal(result.to_numpy()[~mask], truth[~mask])


def test_invert():
    rs = make_raster("arange", shape=(4, 5, 5), null_pattern="<10", null=0)
    data = rs._ds.raster.data.compute()
    mask = rs._ds.mask.data.compute()
    assert rs.null_value == 0
    assert rs._masked
    assert rs.crs == "EPSG:3857"

    result = ~rs
    truth = np.invert(data)
    truth[mask] = get_default_null_value(truth.dtype)
    assert_valid_raster(result)
    assert result._masked
    assert result.dtype == truth.dtype
    assert (result).crs == rs.crs
    assert np.allclose(np.invert(rs), truth)
    assert np.allclose(result, truth)
    truth = np.invert(data.astype(bool))
    truth[mask] = get_default_null_value(truth.dtype)
    assert np.allclose(np.invert(rs.astype(bool)), truth)
    with pytest.raises(TypeError):
        ~rs.astype(float)


@pytest.mark.parametrize(
    "func",
    [
        np.all,
        np.any,
        np.max,
        np.mean,
        np.min,
        np.prod,
        np.std,
        np.sum,
        np.var,
    ],
)
def test_reductions(func):
    data = np.arange(3 * 5 * 5).reshape((3, 5, 5)) - 25
    data = data.astype(float)
    rs = Raster(data).set_null_value(4)
    rs.data[:, :2, :2] = np.nan
    valid = ~(rs._ds.mask.data | np.isnan(rs._ds.raster.data)).compute()

    fname = func.__name__
    if fname in ("amin", "amax"):
        # Drop leading 'a'
        fname = fname[1:]
    assert hasattr(rs, fname)
    assert not np.allclose(valid, rs._ds.mask.data.compute())
    truth = func(data[valid])
    for result in (func(rs), getattr(rs, fname)()):
        assert isinstance(result, dask.array.Array)
        assert result.size == 1
        assert result.shape == ()
        if fname in ("all", "any"):
            assert isinstance(result.compute(), (bool, np.bool_))
        else:
            assert is_scalar(result.compute())
        assert np.allclose(result, truth, equal_nan=True)
    if fname in ("all", "any"):
        rs = rs > 0
        data = data > 0
        truth = func(data[valid])
        assert np.allclose(func(rs), truth, equal_nan=True)
        assert np.allclose(getattr(rs, fname)(), truth, equal_nan=True)


@pytest.mark.parametrize(
    "call",
    [
        lambda rs: np.sum(rs, keepdims=False),
        lambda rs: np.std(rs, ddof=0),
        lambda rs: rs.sum(axis=None, out=None),
        lambda rs: rs.mean(dtype=None),
    ],
)
def test_reductions_accept_default_args(call):
    rs = make_raster("arange", shape=(2, 3, 3), dtype=float)
    data = rs.to_numpy()
    result = call(rs)
    assert isinstance(result, dask.array.Array)
    assert np.allclose(result, call(data))


@pytest.mark.parametrize(
    "call,error",
    [
        (lambda rs: np.sum(rs, axis=0), ValueError),
        (lambda rs: np.max(rs, axis=(1, 2)), ValueError),
        (lambda rs: np.sum(rs, keepdims=True), ValueError),
        (lambda rs: np.mean(rs, dtype=F32), ValueError),
        (lambda rs: np.std(rs, ddof=1), ValueError),
        (lambda rs: np.sum(rs, out=np.zeros(())), ValueError),
        (lambda rs: np.sum(rs, initial=1), TypeError),
        (lambda rs: np.all(rs, where=True), TypeError),
        (lambda rs: rs.sum(0), TypeError),
        (lambda rs: rs.max(foo=1), TypeError),
        (lambda rs: rs.mean(ddof=0), TypeError),
    ],
)
def test_reductions_reject_args(call, error):
    rs = make_raster("arange", shape=(2, 3, 3), dtype=float)
    with pytest.raises(error):
        call(rs)


@pytest.mark.parametrize(
    "other",
    [
        5,
        [9],
        (7,),
        (1, 2, 3, 4),
        [1, 2, 3, 4],
        range(1, 5),
        np.array([1, 2, 3, 4]),
        da.asarray([1, 2, 3, 4]),
    ],
)
@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_MULT)
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_bandwise(ufunc, other):
    rs = make_raster("arange", shape=(4, 5, 5), null_pattern="<10", null=0)
    data = rs.to_numpy()
    mask = rs.mask.compute()
    assert rs.null_value == 0
    assert rs._masked
    assert rs.crs == "EPSG:3857"
    assert hasattr(rs, "bandwise")
    tother = dask.compute(np.atleast_1d(other).reshape((-1, 1, 1)))[0]

    truth = ufunc(data, tother)
    result = ufunc(rs.bandwise, other)
    if ufunc.nout == 1:
        assert_valid_raster(result)
        truth[mask] = get_default_null_value(truth.dtype)
        assert result._masked
        assert truth.dtype == result.dtype
        assert result.crs == rs.crs
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._ds.mask, mask)
    else:
        for r, t in zip(result, truth, strict=True):
            assert_valid_raster(r)
            t[mask] = get_default_null_value(t.dtype)
            assert r._masked
            assert t.dtype == r.dtype
            assert r.crs == rs.crs
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._ds.mask, mask)
    # Reflected
    truth = ufunc(tother, data)
    result = ufunc(other, rs.bandwise)
    if ufunc.nout == 1:
        assert_valid_raster(result)
        truth[mask] = get_default_null_value(truth.dtype)
        assert result._masked
        assert truth.dtype == result.dtype
        assert result.crs == rs.crs
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._ds.mask, mask)
    else:
        for r, t in zip(result, truth, strict=True):
            assert_valid_raster(r)
            t[mask] = get_default_null_value(t.dtype)
            assert r._masked
            assert t.dtype == r.dtype
            assert r.crs == rs.crs
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._ds.mask, mask)


@pytest.mark.parametrize(
    "other",
    [
        [1, 1],
        (1, 1),
        range(0, 3),
        np.ones(3),
        np.ones((2, 2)),
        np.ones((5, 5)),
        unknown_chunk_array,
    ],
)
def test_bandwise_errors(other):
    rs = make_raster("arange", shape=(4, 5, 5), crs=None)

    with pytest.raises(ValueError):
        rs.bandwise * other


NON_SQUARE_AFFINE = Affine(3, 0, 0, 0, -2, 30)
ONE_CELL_WIDE_SHAPES = pytest.mark.parametrize(
    "shape", [(1, 3), (3, 1), (1, 1)], ids=["row", "column", "cell"]
)


def _one_cell_wide_raster(shape, masked):
    # Non-square cells, so the cell size along the length-1 axis can only
    # come from the stored transform
    data = np.arange(1.0, np.prod(shape) + 1).reshape((1, *shape))
    if masked:
        data[0, -1, -1] = np.nan
    return data_to_raster(
        data,
        affine=NON_SQUARE_AFFINE,
        crs=5070,
        nv=np.nan if masked else None,
    )


def _assert_keeps_non_square_cells(result):
    assert_valid_raster(result)
    assert result.affine == NON_SQUARE_AFFINE
    assert tuple(abs(v) for v in result.resolution) == (3, 2)
    assert result._ds.raster.rio.transform() == NON_SQUARE_AFFINE
    assert result._ds.mask.rio.transform() == NON_SQUARE_AFFINE


@pytest.mark.filterwarnings("ignore:The null value ")
@pytest.mark.parametrize("masked", [False, True])
@ONE_CELL_WIDE_SHAPES
@pytest.mark.parametrize(
    "op,np_op",
    [
        (lambda r: r + 1, lambda x: x + 1),
        (lambda r: r * r, lambda x: x * x),
        (lambda r: -r, lambda x: -x),
        (lambda r: r > 1, lambda x: x > 1),
        (lambda r: r.round(), np.round),
        (lambda r: r.astype("float32"), lambda x: x.astype("float32")),
        (lambda r: r.astype("int16"), lambda x: x.astype("int16")),
        (
            lambda r: r.astype("float32", new_null_value=-1),
            lambda x: x.astype("float32"),
        ),
    ],
    ids=[
        "add",
        "multiply",
        "negate",
        "compare",
        "round",
        "astype",
        "astype_int",
        "astype_new_null",
    ],
)
def test_ops_keep_non_square_cells_of_one_cell_wide_raster(
    op, np_op, shape, masked
):
    rs = _one_cell_wide_raster(shape, masked)
    result = op(rs)
    _assert_keeps_non_square_cells(result)
    assert result._masked == masked
    mask = rs.mask.compute()
    assert np.array_equal(result.mask.compute(), mask)
    data = rs.to_numpy()
    assert np.array_equal(result.to_numpy()[~mask], np_op(data[~mask]))


def test_round():
    data = np.arange(5 * 5).reshape((5, 5)).astype(float)
    data += np.linspace(0, 4, 5 * 5).reshape((5, 5))
    rs = Raster(data).set_crs(3857)

    assert hasattr(rs, "round")
    truth = np.round(data)
    assert_valid_raster(rs.round())
    assert isinstance(rs.round(), Raster)
    assert np.allclose(rs.round(), truth)
    truth = np.round(data, decimals=2)
    assert np.allclose(rs.round(2), truth)
    assert rs.round().crs == rs.crs

    rs = Raster(data).set_crs(3857).set_null_value(24.5)
    truth = np.round(data)
    truth[rs.xmask.to_numpy()[0]] = 24.5
    assert_valid_raster(rs.round())
    assert isinstance(rs.round(), Raster)
    assert np.allclose(rs.round(), truth)
    truth = np.round(data, decimals=2)
    truth[rs.xmask.to_numpy()[0]] = 24.5
    assert np.allclose(rs.round(2), truth)
    assert rs.round().crs == rs.crs


@pytest.mark.filterwarnings("ignore:The null value ")
@pytest.mark.parametrize(
    "dtype", sorted(set(DTYPE_INPUT_TO_DTYPE.values()), key=str)
)
@pytest.mark.parametrize(
    "rs",
    [
        testdata.raster.dem_small,
        make_raster("arange", shape=(10, 10), crs=None).set_null_value(99),
    ],
)
def test_astype(rs, dtype):
    result = rs.astype(dtype)
    assert_valid_raster(result)
    assert result.dtype == dtype
    assert result.load().dtype == dtype
    assert result.crs == rs.crs
    assert result.null_value == reconcile_nullvalue_with_dtype(
        rs.null_value, dtype
    )


@pytest.mark.filterwarnings("ignore:The null value ")
def test_astype_alias_resolution():
    rs = make_raster("arange", shape=(10, 10), crs=None)
    for type_code, expected_dtype in DTYPE_INPUT_TO_DTYPE.items():
        result = rs.astype(type_code)
        assert result.dtype == expected_dtype


def test_astype_new_null_value():
    raster = (
        make_raster("arange", shape=(4, 4), crs=None)
        .set_null_value(0)
        .set_null_value(-1)
    )
    assert raster.dtype == "int64"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = raster.astype("uint64", 99)
    assert result.dtype == "uint64"
    assert result.null_value == 99


def test_astype_new_null_value_same_dtype_new_null():
    raster = (
        make_raster("arange", shape=(4, 4), crs=None)
        .set_null_value(0)
        .set_null_value(-1)
    )
    assert raster.dtype == "int64"
    assert raster.null_value == -1
    result = raster.astype(raster.dtype, new_null_value=-99)
    assert result.dtype == "int64"
    assert result.null_value == -99


def test_copy():
    rs = testdata.raster.dem_clipped_small
    copy = rs.copy()
    assert_valid_raster(copy)
    assert rs is not copy
    assert rs._ds is not copy._ds
    assert rs._ds.equals(copy._ds)
    assert copy._ds.attrs == rs._ds.attrs
    assert copy._ds.raster.attrs == rs._ds.raster.attrs
    assert copy._ds.mask.attrs == rs._ds.mask.attrs
    assert np.allclose(rs, copy)
    # make sure a deep copy has occurred
    copy._ds.raster.data[0, -1, -1] = 0
    assert not np.allclose(rs, copy)


def test_set_crs():
    rs = testdata.raster.dem_small
    assert rs.crs != 4326

    rs4326 = rs.set_crs(4326)
    assert rs4326.crs != rs.crs
    assert rs4326.crs == 4326
    assert np.allclose(rs.to_numpy(), rs4326.to_numpy())


@pytest.mark.filterwarnings("ignore:The null value")
def test_set_null_value():
    rs = testdata.raster.dem_clipped_small
    assert rs.null_value is not None
    truth = rs.to_numpy()
    mask = rs._ds.mask.data.compute()
    ndv = rs.null_value
    rs2 = rs.set_null_value(0)
    assert_valid_raster(rs2)
    truth[mask] = 0
    assert rs.null_value == ndv
    assert rs.xdata.rio.nodata == ndv
    assert rs2.xdata.rio.nodata == 0
    assert np.allclose(rs2, truth)
    assert rs2.crs == rs.crs

    rs = make_raster("arange", shape=(10, 10), crs=None)
    assert rs.null_value is None
    assert rs._ds.raster.rio.nodata is None
    rs2 = rs.set_null_value(99)
    assert rs2.null_value == 99
    assert rs2._ds.raster.rio.nodata == 99
    assert np.allclose(
        rs2._ds.mask.to_numpy(), rs2._ds.raster.to_numpy() == 99
    )
    assert rs2.crs == rs2.crs

    rs = testdata.raster.dem_small
    nv = rs.null_value
    rs2 = rs.set_null_value(None)
    assert rs.null_value == nv
    assert rs2.null_value is None
    assert not rs2._ds.mask.to_numpy().any()
    assert np.allclose(rs, rs2)
    assert rs2.crs == rs2.crs

    rs = testdata.raster.dem_clipped_small.astype(int)
    assert rs.dtype == np.dtype(int)
    rs2 = rs.set_null_value(get_default_null_value(float))
    assert rs2.dtype == np.dtype(float)
    assert rs2.crs == rs2.crs


def test_set_null_value_float16_precision_loss():
    rs = rts.data_to_raster(np.ones((2, 2), dtype="float16"), nv=-1)
    rs2 = rs.set_null_value(-9999.0)
    assert rs2.dtype == np.dtype("float32")
    assert rs2.null_value == np.float64(-9999)

    rs2 = rs.set_null_value(-9999)
    assert rs2.dtype == np.dtype("float32")
    assert rs2.null_value == np.float64(-9999)


@pytest.mark.parametrize(
    "raster,value,expected_dtype",
    [
        (
            rts.creation.zeros_like(testdata.raster.dem_small, dtype="int8"),
            0,
            I8,
        ),
        (
            rts.creation.zeros_like(testdata.raster.dem_small, dtype="int64"),
            0,
            I64,
        ),
        (
            rts.creation.zeros_like(testdata.raster.dem_small, dtype="int32"),
            -999999,
            I32,
        ),
        (
            rts.creation.zeros_like(testdata.raster.dem_small, dtype="uint8"),
            99_000,
            U32,
        ),
        (
            rts.creation.zeros_like(testdata.raster.dem_small, dtype="uint8"),
            99_000.0,
            F32,
        ),
        (
            rts.creation.zeros_like(testdata.raster.dem_small, dtype="uint8"),
            -1,
            I16,
        ),
        (
            rts.creation.zeros_like(
                testdata.raster.dem_small, dtype="float16"
            ),
            999,
            F16,
        ),
        (
            rts.creation.zeros_like(
                testdata.raster.dem_small, dtype="float16"
            ),
            -65500,
            # Even though -65500 is the minimum F16 value, np.min_scalar_type
            # says to cast up so that is what we do.
            F32,
        ),
        (
            rts.creation.zeros_like(
                testdata.raster.dem_small, dtype="float16"
            ),
            1.0,
            F16,
        ),
        (
            rts.creation.zeros_like(
                testdata.raster.dem_small, dtype="float16"
            ),
            65501.0,
            F32,
        ),
        (
            rts.creation.zeros_like(testdata.raster.dem_small, dtype="int8"),
            np.nan,
            F16,
        ),
        (
            rts.creation.zeros_like(testdata.raster.dem_small, dtype="uint16"),
            np.nan,
            F32,
        ),
        (
            rts.creation.zeros_like(
                testdata.raster.dem_small, dtype="float16"
            ),
            np.nan,
            F16,
        ),
    ],
)
def test_set_null_value_result_dtype(raster, value, expected_dtype):
    raster = raster.set_null_value(value)
    assert raster.dtype == expected_dtype
    assert np.allclose(raster.null_value, value, equal_nan=True)


def test_replace_null():
    fill_value = 0
    rs = testdata.raster.dem_clipped_small
    rsnp = rs.to_numpy()
    rsnp_replaced = rsnp.copy()
    mask = rsnp == rs.null_value
    rsnp_replaced[mask] = fill_value
    rs = rs.replace_null(fill_value)
    assert np.allclose(rs.to_numpy(), rsnp_replaced)
    assert rs.null_value == fill_value
    assert np.allclose(rs._ds.mask.to_numpy(), mask)

    with pytest.raises(TypeError):
        rs.replace_null(None)


def test_set_null():
    raster = testdata.raster.dem_small
    truth = raster.to_numpy()
    truth_mask = truth < 1500
    truth[truth_mask] = raster.null_value
    result = raster.set_null(raster < 1500)
    assert_valid_raster(result)
    assert result.null_value == raster.null_value
    assert np.allclose(result, truth)
    assert np.allclose(result.mask.compute(), truth_mask)
    assert result._ds.raster.attrs == raster._ds.raster.attrs

    # Make sure broadcasting works
    raster = stack_bands([testdata.raster.dem_small] * 3)
    assert raster.shape == (3, 100, 100)
    mask = (testdata.raster.dem_small < 1500).to_numpy()
    truth[:, mask[0]] = raster.null_value
    result = raster.set_null(testdata.raster.dem_small < 1500)
    assert_valid_raster(result)
    assert result.null_value == raster.null_value
    assert np.allclose(result, truth)
    assert np.allclose(result.mask.compute(), truth_mask)
    assert result._ds.raster.attrs == raster._ds.raster.attrs

    # Make sure that a null value is added if not already present
    raster = testdata.raster.dem_small.set_null_value(None)
    nv = get_default_null_value(raster.dtype)
    truth = raster.to_numpy()
    truth_mask = truth < 1500
    truth[truth_mask] = nv
    result = raster.set_null(raster < 1500)
    assert_valid_raster(result)
    assert result.null_value == nv
    assert np.allclose(result, truth)
    assert np.allclose(result.mask.compute(), truth_mask)
    attrs = raster._ds.raster.attrs
    attrs["_FillValue"] = nv
    assert result._ds.raster.attrs == attrs


def test_to_null_mask():
    rs = testdata.raster.dem_clipped_small
    nv = rs.null_value
    rsnp = rs.to_numpy()
    truth = rsnp == nv
    assert rs.null_value is not None
    nmask = rs.to_null_mask()
    assert_valid_raster(nmask)
    assert nmask.null_value is None
    assert np.allclose(nmask, truth)
    # Test case where no null values
    rs = testdata.raster.dem_small
    truth = np.full(rs.shape, False, dtype=bool)
    nmask = rs.to_null_mask()
    assert_valid_raster(nmask)
    assert nmask.null_value is None
    assert np.allclose(nmask, truth)


def test_eval_emits_deprecation_warning():
    rs = testdata.raster.dem_small
    with pytest.warns(DeprecationWarning, match="Raster.eval"):
        rs.eval()


@pytest.mark.filterwarnings("ignore:'Raster.eval' is deprecated")
@pytest.mark.parametrize("method", ["eval", "load"])
def test_eval_and_load(method):
    rs = testdata.raster.dem_small
    rsnp = rs.xdata.to_numpy()
    rs += 2
    rsnp += 2
    rs -= rs
    rsnp -= rsnp
    rs *= -1
    rsnp *= -1
    result = getattr(rs, method)()
    assert_valid_raster(result)
    assert len(result.data.dask) == 1
    # Make sure new raster returned
    assert rs is not result
    assert rs._ds is not result._ds
    assert np.allclose(result, rsnp)


def test_get_bands():
    rs = make_raster("arange", shape=(4, 100, 100), crs=None)
    rsnp = rs.to_numpy()
    for band_num in range(1, 5):
        r = rs.get_bands(band_num)
        assert_valid_raster(r)
        assert np.allclose(r, rsnp[band_num - 1 : band_num])
    for bands in [[1], [1, 2], [1, 1], [3, 1, 2], [4, 3, 2, 1]]:
        np_bands = [i - 1 for i in bands]
        result = rs.get_bands(bands)
        assert_valid_raster(result)
        assert np.allclose(result, rsnp[np_bands])
        bnd_dim = list(range(1, len(bands) + 1))
        assert np.allclose(result.xdata.band, bnd_dim)

    assert len(rs.get_bands(1).shape) == 3

    for bands in [0, 5, [1, 5], [0]]:
        with pytest.raises(IndexError):
            rs.get_bands(bands)
    with pytest.raises(ValueError):
        rs.get_bands([])
    with pytest.raises(TypeError):
        rs.get_bands(1.0)


def test_burn_mask_with_mask():
    raster = (
        make_raster("arange", shape=(1, 5, 5))
        .set_null_value(4)
        .set_null_value(-1)
    )
    raster._ds.mask.data[raster.data > 20] = True
    truth = raster.copy()
    truth._ds.raster.data = da.where(truth.mask, truth.null_value, truth.data)
    # Confirm state
    assert_valid_raster(truth)
    assert truth.crs == 3857
    assert truth.null_value == -1
    assert np.allclose(
        truth.mask.compute(),
        np.array(
            [
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1],
                ]
            ]
        ).astype(bool),
    )
    # Confirm raster is invalid because the data does not match the mask
    with pytest.raises(AssertionError):
        assert_valid_raster(raster)

    result = raster.burn_mask()
    assert_valid_raster(result)
    assert_rasters_similar(result, truth)
    assert result is not raster
    assert result.null_value == truth.null_value
    # Make sure nothing else changed on the raster
    assert result.xdata.attrs == raster.xdata.attrs
    assert np.allclose(result, truth)
    assert np.allclose(result.mask.compute(), truth.mask.compute())


def test_burn_mask_no_mask():
    raster = make_raster("arange", shape=(1, 5, 5))
    assert raster.null_value is None
    result = raster.burn_mask()
    assert_valid_raster(result)
    assert result is not raster
    assert result.null_value is None
    # Make sure nothing else changed on the raster
    assert result.xdata.attrs == raster.xdata.attrs
    assert np.allclose(result, raster)
    assert np.allclose(result.mask.compute(), raster.mask.compute())


def test_burn_mask_boolean():
    raster = (
        make_raster("arange", shape=(1, 5, 5))
        .set_null_value(0)
        .set_null_value(-1)
    )
    raster = raster > 20
    raster._ds.mask.data[raster.data <= 12] = True
    assert raster.dtype == np.dtype(bool)
    assert raster.null_value == get_default_null_value(bool)
    truth = raster.copy()
    truth._ds.raster.data = da.where(
        truth.mask, get_default_null_value(bool), truth.data
    )
    truth._ds["raster"] = truth.xdata.rio.write_nodata(
        get_default_null_value(bool)
    )
    result = raster.burn_mask()
    assert_valid_raster(result)
    assert_rasters_similar(result, truth)
    assert result is not raster
    # Make sure nothing else changed on the raster
    assert result.xdata.attrs == raster.xdata.attrs
    assert np.allclose(result, truth)
    assert np.allclose(result.mask.compute(), truth.mask.compute())


@pytest.mark.parametrize("index", [(0, 0), (0, 1), (1, 0), (-1, -1), (1, -1)])
@pytest.mark.parametrize("offset", ["center", "ul", "ll", "ur", "lr"])
def test_xy(index, offset):
    rs = testdata.raster.dem_small
    i, j = index
    if i == -1:
        i = rs.shape[1] - 1
    if j == -1:
        j = rs.shape[2] - 1

    transform = rs.affine
    assert rs.xy(i, j, offset) == rio.transform.xy(
        transform, i, j, offset=offset
    )


@pytest.mark.parametrize(
    "x,y",
    [
        (0, 0),
        (-47848.38283538996, 65198.938461863494),
        (-47278.38283538996, 65768.9384618635),
        (-44878.38283538996, 68168.9384618635),
        (-46048.38283538996, 66308.9384618635),
        (-46048.0, 66328.1),
        (-46025.0, 66328.1),
    ],
)
def test_index(x, y):
    rs = testdata.raster.dem_small
    transform = rs.affine

    assert rs.index(x, y) == rio.transform.rowcol(transform, x, y)


@pytest.mark.parametrize("offset_name", ["center", "ul", "ur", "ll", "lr"])
def test_rowcol_to_xy(offset_name):
    rs = testdata.raster.dem_small
    affine = rs.affine
    _, ny, nx = rs.shape
    rr, cc = np.mgrid[:ny, :nx]
    r = rr.ravel()
    c = cc.ravel()
    xy_rio = rio.transform.xy(affine, r, c, offset=offset_name)
    result = rowcol_to_xy(r, c, affine, offset_name)
    assert np.allclose(xy_rio, result)


def test_xy_to_rowcol():
    rs = testdata.raster.dem_small
    affine = rs.affine
    xx, yy = np.meshgrid(rs.x, rs.y)
    x = xx.ravel()
    y = yy.ravel()
    rc_rio = rio.transform.rowcol(affine, x, y)
    result = xy_to_rowcol(x, y, affine)
    assert np.allclose(rc_rio, result)
    assert result[0].dtype == np.dtype(int)
    assert result[1].dtype == np.dtype(int)


@pytest.mark.parametrize(
    "raster,expected",
    [
        (
            make_raster("arange", shape=(4, 4), dtype="float32", crs="5070")
            .set_null(make_raster("arange", shape=(4, 4), crs="5070") < 3)
            .chunk((1, 2, 2)),
            dd.concat(
                [
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([4, 5], dtype="float32"),
                            "band": np.array([1, 1], dtype="int64"),
                            "row": np.array([1, 1], dtype="int64"),
                            "col": np.array([0, 1], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [0.5, 1.5], [2.5, 2.5], crs="5070"
                            ),
                        },
                        crs="5070",
                        index=np.array([4, 5], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([3, 6, 7], dtype="float32"),
                            "band": np.array([1, 1, 1], dtype="int64"),
                            "row": np.array([0, 1, 1], dtype="int64"),
                            "col": np.array([3, 2, 3], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [3.5, 2.5, 3.5], [3.5, 2.5, 2.5], crs="5070"
                            ),
                        },
                        crs="5070",
                        index=np.array([3, 6, 7], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([8, 9, 12, 13], dtype="float32"),
                            "band": np.array([1, 1, 1, 1], dtype="int64"),
                            "row": np.array([2, 2, 3, 3], dtype="int64"),
                            "col": np.array([0, 1, 0, 1], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [0.5, 1.5, 0.5, 1.5],
                                [1.5, 1.5, 0.5, 0.5],
                                crs="5070",
                            ),
                        },
                        crs="5070",
                        index=np.array([8, 9, 12, 13], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array(
                                [10, 11, 14, 15], dtype="float32"
                            ),
                            "band": np.array([1, 1, 1, 1], dtype="int64"),
                            "row": np.array([2, 2, 3, 3], dtype="int64"),
                            "col": np.array([2, 3, 2, 3], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [2.5, 3.5, 2.5, 3.5],
                                [1.5, 1.5, 0.5, 0.5],
                                crs="5070",
                            ),
                        },
                        crs="5070",
                        index=np.array([10, 11, 14, 15], dtype="int64"),
                    ),
                ]
            ),
        ),
        (
            make_raster("arange", shape=(2, 4, 4), dtype="int8", crs="5070")
            .set_null(make_raster("arange", shape=(2, 4, 4), crs="5070") < 3)
            .chunk((1, 2, 2)),
            dd.concat(
                [
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([4, 5], dtype="int8"),
                            "band": np.array([1, 1], dtype="int64"),
                            "row": np.array([1, 1], dtype="int64"),
                            "col": np.array([0, 1], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [0.5, 1.5], [2.5, 2.5], crs="5070"
                            ),
                        },
                        crs="5070",
                        index=np.array([4, 5], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([3, 6, 7], dtype="int8"),
                            "band": np.array([1, 1, 1], dtype="int64"),
                            "row": np.array([0, 1, 1], dtype="int64"),
                            "col": np.array([3, 2, 3], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [3.5, 2.5, 3.5], [3.5, 2.5, 2.5], crs="5070"
                            ),
                        },
                        crs="5070",
                        index=np.array([3, 6, 7], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([8, 9, 12, 13], dtype="int8"),
                            "band": np.array([1, 1, 1, 1], dtype="int64"),
                            "row": np.array([2, 2, 3, 3], dtype="int64"),
                            "col": np.array([0, 1, 0, 1], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [0.5, 1.5, 0.5, 1.5],
                                [1.5, 1.5, 0.5, 0.5],
                                crs="5070",
                            ),
                        },
                        crs="5070",
                        index=np.array([8, 9, 12, 13], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([10, 11, 14, 15], dtype="int8"),
                            "band": np.array([1, 1, 1, 1], dtype="int64"),
                            "row": np.array([2, 2, 3, 3], dtype="int64"),
                            "col": np.array([2, 3, 2, 3], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [2.5, 3.5, 2.5, 3.5],
                                [1.5, 1.5, 0.5, 0.5],
                                crs="5070",
                            ),
                        },
                        crs="5070",
                        index=np.array([10, 11, 14, 15], dtype="int64"),
                    ),
                    # next band
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([16, 17, 20, 21], dtype="int8"),
                            "band": np.array([2, 2, 2, 2], dtype="int64"),
                            "row": np.array([0, 0, 1, 1], dtype="int64"),
                            "col": np.array([0, 1, 0, 1], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [0.5, 1.5, 0.5, 1.5],
                                [3.5, 3.5, 2.5, 2.5],
                                crs="5070",
                            ),
                        },
                        crs="5070",
                        index=np.array([16, 17, 20, 21], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([18, 19, 22, 23], dtype="int8"),
                            "band": np.array([2, 2, 2, 2], dtype="int64"),
                            "row": np.array([0, 0, 1, 1], dtype="int64"),
                            "col": np.array([2, 3, 2, 3], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [2.5, 3.5, 2.5, 3.5],
                                [3.5, 3.5, 2.5, 2.5],
                                crs="5070",
                            ),
                        },
                        crs="5070",
                        index=np.array([18, 19, 22, 23], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([24, 25, 28, 29], dtype="int8"),
                            "band": np.array([2, 2, 2, 2], dtype="int64"),
                            "row": np.array([2, 2, 3, 3], dtype="int64"),
                            "col": np.array([0, 1, 0, 1], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [0.5, 1.5, 0.5, 1.5],
                                [1.5, 1.5, 0.5, 0.5],
                                crs="5070",
                            ),
                        },
                        crs="5070",
                        index=np.array([24, 25, 28, 29], dtype="int64"),
                    ),
                    gpd.GeoDataFrame(
                        {
                            "value": np.array([26, 27, 30, 31], dtype="int8"),
                            "band": np.array([2, 2, 2, 2], dtype="int64"),
                            "row": np.array([2, 2, 3, 3], dtype="int64"),
                            "col": np.array([2, 3, 2, 3], dtype="int64"),
                            "geometry": gpd.points_from_xy(
                                [2.5, 3.5, 2.5, 3.5],
                                [1.5, 1.5, 0.5, 0.5],
                                crs="5070",
                            ),
                        },
                        crs="5070",
                        index=np.array([26, 27, 30, 31], dtype="int64"),
                    ),
                ]
            ),
        ),
    ],
)
def test_to_points(raster, expected):
    result = raster.to_points()
    assert isinstance(result, dgpd.GeoDataFrame)
    assert result.npartitions == raster.data.npartitions
    assert result.crs == raster.crs
    assert result.columns.tolist() == [
        "value",
        "band",
        "row",
        "col",
        "geometry",
    ]
    assert result.dtypes.to_dict() == {
        "value": raster.dtype,
        "band": np.dtype("int64"),
        "row": np.dtype("int64"),
        "col": np.dtype("int64"),
        "geometry": expected.geometry.dtype,
    }
    cresult = result.compute()
    cexpected = expected.compute()
    assert cresult.equals(cexpected)

    # Check that index is the flattened index from original array
    index = np.array(sorted(cresult.index))
    assert np.allclose(
        index,
        np.ravel_multi_index(np.nonzero(~raster.mask.compute()), raster.shape),
    )
    band = cresult.band.to_numpy() - 1
    row = cresult.row.to_numpy()
    col = cresult.col.to_numpy()
    assert np.allclose(
        cresult.index.to_numpy(),
        np.ravel_multi_index((band, row, col), raster.shape),
    )
    # Check that index is unique
    assert np.allclose(np.unique(cresult.index), sorted(cexpected.index))
    # Check that index values are independent of chunking
    cresult = raster.chunk((1, 1, 1)).to_points().compute()
    assert cresult.sort_index().equals(cexpected.sort_index())


@pytest.mark.parametrize(
    "raster",
    [
        testdata.raster.dem,
        make_raster("arange", shape=(3, 35, 50))
        .remap_range([(0, 10, 0), (3490, 3500, 0), (5200, 5210, 0)])
        .set_null_value(0),
        make_raster("arange", shape=(3, 1, 1), crs=None),
    ],
)
def test_to_quadrants(raster):
    nb, ny, nx = raster.shape
    assert hasattr(raster, "to_quadrants")
    quads = raster.to_quadrants()
    assert isinstance(quads, RasterQuadrantsResult)
    assert len(quads) == 4
    assert hasattr(quads, "nw")
    assert hasattr(quads, "ne")
    assert hasattr(quads, "sw")
    assert hasattr(quads, "se")
    for q in quads:
        assert_valid_raster(q)
        assert q.crs == raster.crs
        assert q.null_value == raster.null_value
        assert len(q.band) == nb
        assert np.allclose(q.band, raster.band)
    assert quads.nw == quads[0]
    assert quads.ne == quads[1]
    assert quads.sw == quads[2]
    assert quads.se == quads[3]
    assert quads.nw.x.size + quads.ne.x.size == nx
    assert quads.sw.x.size + quads.se.x.size == nx
    assert quads.nw.y.size + quads.sw.y.size == ny
    assert quads.ne.y.size + quads.se.y.size == ny
    assert np.allclose(np.concatenate([quads.nw.x, quads.ne.x]), raster.x)
    assert np.allclose(np.concatenate([quads.sw.x, quads.se.x]), raster.x)
    assert np.allclose(np.concatenate([quads.nw.y, quads.sw.y]), raster.y)
    assert np.allclose(np.concatenate([quads.ne.y, quads.se.y]), raster.y)
    xreconstructed = xr.concat(
        [
            xr.concat([quads.nw.xdata, quads.ne.xdata], dim="x"),
            xr.concat([quads.sw.xdata, quads.se.xdata], dim="x"),
        ],
        dim="y",
    )
    assert np.allclose(xreconstructed, raster.xdata)
    xmask_reconstructed = xr.concat(
        [
            xr.concat([quads.nw.xmask, quads.ne.xmask], dim="x"),
            xr.concat([quads.sw.xmask, quads.se.xmask], dim="x"),
        ],
        dim="y",
    )
    assert np.allclose(xmask_reconstructed, raster.xmask)


def test_get_chunk_bounding_boxes():
    raster = make_raster("arange", shape=(6, 6), crs="EPSG:5070").chunk(
        (1, 3, 3)
    )
    boxes = [
        box(0.0, 3.0, 3.0, 6.0),
        box(3.0, 3.0, 6.0, 6.0),
        box(0.0, 0.0, 3.0, 3.0),
        box(3.0, 0.0, 6.0, 3.0),
    ]
    rows = [0, 0, 1, 1]
    cols = [0, 1, 0, 1]
    truth = gpd.GeoDataFrame(
        {"chunk_row": rows, "chunk_col": cols, "geometry": boxes},
        crs=raster.crs,
    )

    boxes_df = raster.get_chunk_bounding_boxes()
    assert truth.crs == boxes_df.crs
    assert boxes_df.geometry.geom_equals_exact(
        truth.geometry, 0, align=False
    ).all()
    assert boxes_df.equals(truth)

    raster = raster_tools.stack_bands([raster, raster]).chunk((1, 3, 3))
    rows += rows
    cols += cols
    boxes += boxes
    bands = ([0] * 4) + ([1] * 4)
    truth = gpd.GeoDataFrame(
        {
            "chunk_band": bands,
            "chunk_row": rows,
            "chunk_col": cols,
            "geometry": boxes,
        },
        crs=raster.crs,
    )
    boxes_df = raster.get_chunk_bounding_boxes(True)
    assert truth.crs == boxes_df.crs
    assert boxes_df.geometry.geom_equals_exact(
        truth.geometry, 0, align=False
    ).all()
    assert boxes_df.equals(truth)


def _build_chunk_truth(raster, nv=None):
    blocks = raster.data.blocks
    numblocks = raster.data.numblocks
    y_splits = np.concatenate([[0], np.cumsum(raster.data.chunks[1])]).astype(
        int
    )
    x_splits = np.concatenate([[0], np.cumsum(raster.data.chunks[2])]).astype(
        int
    )
    truth = np.empty(numblocks, dtype="O")
    kwargs = {"nv": nv} if nv is not None else {}
    for bi, yi, xi in np.ndindex(numblocks):
        truth[bi, yi, xi] = data_to_raster(
            blocks[bi, yi, xi],
            x=raster.x[x_splits[xi] : x_splits[xi + 1]],
            y=raster.y[y_splits[yi] : y_splits[yi + 1]],
            crs=raster.crs,
            **kwargs,
        )
    return truth


def _assert_chunk_rasters_match(rasters, truth):
    assert rasters.shape == truth.shape
    for r, t in zip(rasters.ravel(), truth.ravel(), strict=True):
        assert_valid_raster(r)
        assert r.data.npartitions == 1
        assert r.data.chunksize == r.shape
        assert np.allclose(r, t)
        assert np.allclose(r.mask.compute(), t.mask.compute())
        assert_rasters_similar(r, t)


def test_get_chunk_rasters():
    raster = make_raster("arange", shape=(6, 6), crs="EPSG:5070").chunk(
        (1, 3, 3)
    )
    truth = _build_chunk_truth(raster)
    _assert_chunk_rasters_match(raster.get_chunk_rasters(), truth)

    raster = raster.set_null_value(0).set_null_value(3).set_null_value(6)
    truth = _build_chunk_truth(raster, nv=6)
    _assert_chunk_rasters_match(raster.get_chunk_rasters(), truth)

    raster = raster_tools.stack_bands([raster, raster + 3]).set_null_value(6)
    truth = _build_chunk_truth(raster, nv=6)
    _assert_chunk_rasters_match(raster.get_chunk_rasters(), truth)


def test_get_chunk_rasters_chunk_size_1_has_geobox_and_affine():
    raster = Raster(
        xr.DataArray(
            np.ones((2, 2)), dims=("y", "x"), coords=([1, 0], [0, 1])
        ).rio.write_crs("5070")
    ).chunk((1, 1, 1))
    chunk_rasters = raster.get_chunk_rasters()
    for b, i, j in np.ndindex(chunk_rasters.shape):
        r = chunk_rasters[b, i, j]
        assert r.geobox is not None
        assert np.allclose(
            list(r.affine), list(raster.affine * Affine.translation(j, i))
        )


@pytest.mark.parametrize("neighbors", [4, 8])
@pytest.mark.parametrize(
    "raster",
    [
        testdata.raster.dem_small,
        testdata.raster.dem_small.chunk((1, 20, 20)),
        stack_bands(
            [testdata.raster.dem_small, testdata.raster.dem_small]
        ).chunk((1, 20, 20)),
    ],
)
def test_to_polygons(raster, neighbors):
    hist = np.histogram_bin_edges(raster.to_numpy().ravel(), bins=10)
    mapping = [(hist[i], hist[i + 1], i) for i in range(len(hist) - 1)]
    mapping[-1] = (mapping[-1][0], mapping[-1][1] + 1, mapping[-1][-1])
    raster.data[:, 1100:2000, 1200:2200] = True
    raster = raster.burn_mask()
    feats = raster.remap_range(mapping)
    band_truths = []
    for values, mask, band in zip(
        feats.to_numpy(), feats.mask, np.arange(raster.nbands) + 1, strict=True
    ):
        raw_shapes_results = list(
            rio.features.shapes(
                values,
                # Invert mask to match rasterio interpretation
                ~mask.compute(),
                connectivity=neighbors,
                transform=feats.affine,
            )
        )
        shapes = [shapely.geometry.shape(s[0]) for s in raw_shapes_results]
        values = [s[1] for s in raw_shapes_results]
        truth = (
            gpd.GeoDataFrame(
                {
                    "value": values,
                    "band": [band] * len(values),
                    "geometry": shapes,
                },
                crs=feats.crs,
            )
            .dissolve(by="value")
            .reset_index()
        )
        truth = truth[["value", "band", "geometry"]]
        band_truths.append(truth)
    truth = pd.concat(band_truths)
    # Reset index to make comparison easier below
    truth = truth.sort_values(by=["band", "value"]).reset_index(drop=True)
    assert len(truth) == raster.nbands * 10

    result = feats.to_polygons(neighbors)
    assert result.npartitions == 1
    assert result.crs == raster.crs
    assert result.columns.equals(truth.columns)
    result = result.compute()
    result = result.sort_values(by=["band", "value"]).reset_index(drop=True)
    assert result[["value", "band"]].equals(truth[["value", "band"]])
    for g1, g2 in zip(
        truth.geometry.to_numpy(), result.geometry.to_numpy(), strict=True
    ):
        # Resort to shapely's equals because the points for each polygon are
        # the same but the order is shifted due to the starting/ending points
        # being different. geopandas doesn't consider this in equality checks.
        assert g1.equals(g2)


@pytest.mark.parametrize("neighbors", [4, 8])
def test_to_polygons_and_back_to_raster(neighbors):
    raster = testdata.raster.dem_small
    vec = Vector(raster.to_polygons(neighbors).compute())
    new_rast = vec.to_raster(raster, field="value").set_null_value(
        raster.null_value
    )
    assert np.allclose(raster, new_rast)


if __name__ == "__main__":
    unittest.main()
