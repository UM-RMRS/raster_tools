import operator
import unittest

import affine
import dask
import dask.array as da
import numpy as np
import pytest
import rasterio as rio
import rioxarray as riox
import xarray as xr
from dask_geopandas import GeoDataFrame

from raster_tools import Raster, band_concat
from raster_tools.dtypes import (
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
    rowcol_to_xy,
    xy_to_rowcol,
)
from raster_tools.utils import is_strictly_decreasing, is_strictly_increasing
from tests.utils import arange_nd, arange_raster, assert_valid_raster

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
    if not is_float(data.dtype):
        assert raster.null_value is None
        assert not raster.mask.any().compute()
        assert np.allclose(raster, data.compute())
    else:
        mask = np.isnan(data)
        assert np.allclose(raster._ds.mask.data.compute(), mask.compute())
        data_cp = data.copy()
        data_cp[mask] = get_default_null_value(data.dtype)
        assert raster.null_value == get_default_null_value(data.dtype)
        assert np.allclose(raster, data_cp.compute())
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
        ).chunk((1, 1, 1)),
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
        assert False
    # Band dim starts at 1
    band = [1] if xdata.ndim == 2 else np.arange(xdata.shape[0]) + 1
    assert np.allclose(raster._ds.band, band)
    assert np.allclose(raster._ds.x, xdata[xc])
    assert np.allclose(raster._ds.y, xdata[yc])


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
        "tests/data/raster/elevation.tif",
        "tests/data/raster/elevation_clipped.tif",
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


@pytest.mark.parametrize(
    "raster_input,masked",
    [
        ("tests/data/raster/elevation_small.tif", True),
        ("tests/data/raster/elevation_clipped_small.tif", True),
        (np.ones((1, 3, 3)), False),
        (np.array([np.nan, 3.2, 4, 5]).reshape((2, 2)), True),
        (
            riox.open_rasterio(
                "tests/data/raster/elevation_clipped_small.tif"
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
    raster = Raster("tests/data/raster/elevation_clipped_small.tif")
    assert hasattr(raster, "values")
    assert isinstance(raster.values, np.ndarray)
    assert raster.shape == raster.values.shape
    assert np.allclose(raster.values, raster._ds.raster.data.compute())


def test_property_null_value():
    path = "tests/data/raster/elevation_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "null_value")
    assert raster.null_value == xdata.rio.nodata


def test_property_dtype():
    path = "tests/data/raster/elevation_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "dtype")
    assert isinstance(raster.dtype, np.dtype)
    assert raster.dtype == xdata.dtype


def test_property_shape():
    path = "tests/data/raster/elevation_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "shape")
    assert isinstance(raster.shape, tuple)
    assert raster.shape == xdata.shape


def test_property_crs():
    path = "tests/data/raster/elevation_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "crs")
    assert isinstance(raster.crs, rio.crs.CRS)
    assert raster.crs == xdata.rio.crs


def test_property_affine():
    path = "tests/data/raster/elevation_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "affine")
    assert isinstance(raster.affine, affine.Affine)
    assert raster.affine == xdata.rio.transform()


def test_property_resolution():
    path = "tests/data/raster/elevation_clipped_small.tif"
    raster = Raster(path)
    xdata = riox.open_rasterio(path)
    assert hasattr(raster, "resolution")
    assert isinstance(raster.resolution, tuple)
    assert raster.resolution == xdata.rio.resolution()


def test_property_xdata():
    rs = Raster("tests/data/raster/elevation_small.tif")
    assert hasattr(rs, "xdata")
    assert isinstance(rs.xdata, xr.DataArray)
    assert rs.xdata.identical(rs._ds.raster)


def test_property_data():
    rs = Raster("tests/data/raster/elevation_small.tif")
    assert hasattr(rs, "data")
    assert isinstance(rs.data, da.Array)
    assert rs.data is rs._ds.raster.data


def test_property_bounds():
    rs = Raster("tests/data/raster/elevation_small.tif")
    rds = rio.open("tests/data/raster/elevation_small.tif")
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


def test_property_mask():
    rs = Raster(np.arange(100).reshape((10, 10)) % 4).set_null_value(0)
    assert rs._ds.mask.data.sum().compute() > 0

    assert hasattr(rs, "mask")
    assert isinstance(rs.mask, dask.array.Array)
    assert np.allclose(rs.mask.compute(), rs._ds.mask.data.compute())


def test_property_xmask():
    rs = Raster(np.arange(100).reshape((10, 10)) % 4).set_null_value(0)
    rs._ds = rs._ds.rio.write_crs("EPSG:3857")
    assert rs.mask.sum().compute() > 0

    assert hasattr(rs, "xmask")
    assert isinstance(rs.xmask, xr.DataArray)
    assert np.allclose(rs.xmask.data.compute(), rs._ds.mask.data.compute())
    assert rs.xmask.rio.crs == rs.crs
    assert np.allclose(rs.xmask.x.data, rs.xdata.x.data)
    assert np.allclose(rs.xmask.y.data, rs.xdata.y.data)


@pytest.mark.parametrize("name", ["band", "x", "y"])
def test_properties_coords(name):
    rs = Raster("tests/data/raster/elevation_small.tif")

    assert hasattr(rs, name)
    assert isinstance(getattr(rs, name), np.ndarray)
    assert np.allclose(getattr(rs, name), rs.xdata[name].data)


@pytest.mark.parametrize(
    "rs",
    [
        Raster(da.ones((4, 100, 100), chunks=(1, 5, 5))),
        Raster(da.ones((100, 100), chunks=(5, 5))),
        Raster(np.ones((100, 100))),
        Raster("tests/data/raster/elevation.tif"),
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


@pytest.mark.parametrize("op", _BINARY_ARITHMETIC_OPS + _BINARY_COMPARISON_OPS)
@pytest.mark.parametrize("operand", [-2.0, -1, 0, 2, 3.0, True])
@pytest.mark.parametrize(
    "rs_type", [F16, F32, F64, I16, I32, I64, I8, U16, U32, U64, U8]
)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_binary_ops_arithmetic_against_scalar(op, operand, rs_type):
    x = arange_nd((4, 5, 5), dtype=rs_type)
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), 0, rs._ds.raster
    )
    rs = rs.set_null_value(0).set_crs("EPSG:3857")
    mask = rs.mask.compute()
    assert rs.null_value == 0
    assert rs._masked
    x = rs.values

    result = op(operand, rs)
    truth = op(operand, x)
    truth = np.where(mask, get_default_null_value(truth.dtype), truth)
    assert_valid_raster(result)
    assert result._masked
    assert result.crs == rs.crs
    assert np.allclose(result, truth, equal_nan=True)
    assert result.dtype == truth.dtype
    assert np.allclose(result.mask.compute(), rs.mask.compute())
    assert np.all(result.xdata.spatial_ref == rs.xdata.spatial_ref).values
    if is_bool(result.dtype):
        assert is_bool(result.null_value)
        assert result.null_value == get_default_null_value(bool)
    # Make sure raising ints to negative ints raises TypeError
    if (
        op == operator.pow
        and is_int(rs.dtype)
        and is_int(operand)
        and operand < 0
        # Numpy promotes to float64 in this case
        and rs.dtype != U64
    ):
        with pytest.raises(TypeError):
            op(rs, operand)
    else:
        truth = op(x, operand)
        truth = np.where(mask, get_default_null_value(truth.dtype), truth)
        result = op(rs, operand)
        assert_valid_raster(result)
        assert result._masked
        assert result.dtype == truth.dtype
        assert result.crs == rs.crs
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result.mask.compute(), rs.mask.compute())
        assert np.all(result.xdata.spatial_ref == rs.xdata.spatial_ref).values


unknown_chunk_array = dask.array.ones((5, 5))
unknown_chunk_array = unknown_chunk_array[unknown_chunk_array > 0]


@pytest.mark.parametrize("op", _BINARY_ARITHMETIC_OPS + _BINARY_COMPARISON_OPS)
@pytest.mark.parametrize(
    "other,error",
    [
        ([1], False),
        (np.array([1]), False),
        (np.array([[1]]), False),
        (np.ones((5, 5)), False),
        (np.ones((1, 5, 5)), False),
        (np.ones((4, 5, 5)), False),
        (dask.array.ones((5, 5)), False),
        (np.zeros(4), True),
        (np.array([1, 1]), True),
        (unknown_chunk_array, True),
    ],
)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
@pytest.mark.filterwarnings("ignore:elementwise comparison failed")
def test_binary_ops_arithmetic_against_array(op, other, error):
    x = arange_nd((4, 5, 5), dtype=float)
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), 0, rs._ds.raster
    )
    rs = rs.set_null_value(0).set_crs("EPSG:3857")
    mask = rs.mask.compute()
    assert rs.null_value == 0
    assert rs._masked
    data = rs.values

    if error:
        with pytest.raises(ValueError):
            op(rs, other)
        with pytest.raises(ValueError):
            op(other, rs)
    else:
        truth1 = op(data, other)
        truth2 = op(other, data)
        # Compute because dask throws an error when indexing with a mask
        truth1, truth2 = dask.compute(truth1, truth2)
        truth1[mask] = get_default_null_value(truth1.dtype)
        truth2[mask] = get_default_null_value(truth2.dtype)
        result = op(rs, other)
        assert_valid_raster(result)
        assert result.crs == rs.crs
        (truth1,) = dask.compute(truth1)
        assert np.allclose(result, truth1, equal_nan=True)
        assert np.allclose(result._ds.mask.compute(), rs._ds.mask.compute())
        assert np.all(result.xdata.spatial_ref == rs.xdata.spatial_ref).values
        result = op(other, rs)
        assert_valid_raster(result)
        (truth2,) = dask.compute(truth2)
        assert np.allclose(result, truth2, equal_nan=True)
        assert np.allclose(result._ds.mask.compute(), rs._ds.mask.compute())
        assert np.all(result.xdata.spatial_ref == rs.xdata.spatial_ref).values


@pytest.mark.parametrize("op", _BINARY_ARITHMETIC_OPS + _BINARY_COMPARISON_OPS)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_binary_ops_arithmetic_against_raster(op):
    x = arange_nd((4, 5, 5), dtype=float)
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), 0, rs._ds.raster
    )
    rs = rs.set_null_value(0).set_crs("EPSG:3857")
    mask = rs.mask.compute()
    assert rs.null_value == 0
    assert rs._masked
    data = rs.values
    data2 = np.ones_like(data, dtype=int) * 2
    rs2 = Raster(data2).set_crs("EPSG:3857")

    truth1 = op(data, data2)
    truth2 = op(data2, data)
    truth1[mask] = get_default_null_value(truth1.dtype)
    truth2[mask] = get_default_null_value(truth2.dtype)
    result = op(rs, rs2)
    assert_valid_raster(result)
    assert result.crs == rs.crs
    assert result._masked
    assert result.dtype == truth1.dtype
    assert np.allclose(result, truth1, equal_nan=True)
    assert np.allclose(result._ds.mask, rs._ds.mask)
    assert np.all(result.xdata.spatial_ref == rs.xdata.spatial_ref).values
    result = op(rs2, rs)
    assert_valid_raster(result)
    assert result.crs == rs.crs
    assert result._masked
    assert result.dtype == truth2.dtype
    assert np.allclose(result, truth2, equal_nan=True)
    assert np.allclose(result._ds.mask, rs._ds.mask)
    assert np.all(result.xdata.spatial_ref == rs.xdata.spatial_ref).values


def test_binary_ops_arithmetic_inplace():
    data = np.arange(4 * 5 * 5).reshape((4, 5, 5))
    rs = Raster(data).set_crs("EPSG:3857")

    r = rs.copy()
    rr = r
    r += 3
    t = data + 3
    assert np.allclose(r, t)
    assert rr is r
    assert rr.crs == rs.crs

    r = rs.copy()
    rr = r
    r -= 3
    t = data - 3
    assert np.allclose(r, t)
    assert rr is r
    assert rr.crs == rs.crs

    r = rs.copy()
    rr = r
    r *= 3
    t = data * 3
    assert np.allclose(r, t)
    assert rr is r
    assert rr.crs == rs.crs

    r = rs.copy()
    rr = r
    r **= 3
    t = data**3
    assert np.allclose(r, t)
    assert rr is r
    assert rr.crs == rs.crs

    r = rs.copy()
    rr = r
    r /= 3
    t = data / 3
    assert np.allclose(r, t)
    assert rr is r
    assert rr.crs == rs.crs

    r = rs.copy()
    rr = r
    r //= 3
    t = data // 3
    assert np.allclose(r, t)
    assert rr is r
    assert rr.crs == rs.crs

    r = rs.copy()
    rr = r
    r %= 3
    t = data % 3
    assert np.allclose(r, t)
    assert rr is r
    assert rr.crs == rs.crs


_NP_UFUNCS = list(
    filter(
        lambda x: isinstance(x, np.ufunc)
        # not valid for rasters
        and x not in (np.isnat, np.matmul),
        map(lambda x: getattr(np, x), dir(np)),
    )
)
_NP_UFUNCS_NIN_SINGLE = list(filter(lambda x: x.nin == 1, _NP_UFUNCS))
_NP_UFUNCS_NIN_MULT = list(filter(lambda x: x.nin > 1, _NP_UFUNCS))
_UNSUPPORED_UFUNCS = frozenset((np.isnat, np.matmul))


@pytest.mark.parametrize("ufunc", list(_UNSUPPORED_UFUNCS))
def test_ufuncs_unsupported(ufunc):
    rs = Raster(np.arange(4 * 5 * 5).reshape((4, 5, 5)))
    with pytest.raises(TypeError):
        args = [rs for i in range(ufunc.nin)]
        ufunc(*args)


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_SINGLE)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_ufuncs_single_input(ufunc):
    x = arange_nd((4, 5, 5), dtype=float)
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), 0, rs._ds.raster
    )
    rs = rs.set_null_value(0).set_crs("EPSG:3857")
    mask = rs.mask.compute()
    assert rs.null_value == 0
    assert rs._masked
    assert rs.crs == "EPSG:3857"
    data = rs.values

    if ufunc == np.invert and is_float(data.dtype):
        with pytest.raises(TypeError):
            ufunc(rs)
        return
    truth = ufunc(data)
    result = ufunc(rs)
    if ufunc.nout == 1:
        assert_valid_raster(result)
        truth[mask] = get_default_null_value(truth.dtype)
        assert result._masked
        assert result.dtype == truth.dtype
        assert result.crs == rs.crs
        assert np.allclose(result, truth, equal_nan=True)
        assert np.allclose(result._ds.mask, rs._ds.mask, equal_nan=True)
    else:
        for r, t in zip(result, truth):
            assert_valid_raster(r)
            t[mask] = get_default_null_value(t.dtype)
            assert r._masked
            assert r.dtype == t.dtype
            assert r.crs == rs.crs
            assert np.allclose(r, t, equal_nan=True)
            assert np.allclose(r._ds.mask, rs._ds.mask, equal_nan=True)


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_MULT)
@pytest.mark.parametrize("dtype", [int, float])
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_ufuncs_multiple_input_against_scalar(ufunc, dtype):
    x = arange_nd((4, 5, 5), dtype=dtype)
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), 0, rs._ds.raster
    )
    rs = rs.set_null_value(0).set_crs("EPSG:3857")
    mask = rs.mask.compute()
    assert rs.null_value == 0
    assert rs._masked
    assert rs.crs == "EPSG:3857"
    extra = ufunc.nin - 1

    args = [rs] + [2 for i in range(extra)]
    args_np = [getattr(a, "values", a) for a in args]
    ufname = ufunc.__name__
    if (
        ufname.startswith("bitwise")
        or ufname.endswith("shift")
        or ufname in ("gcd", "lcm")
    ) and any(is_float(getattr(a, "dtype", a)) for a in args):
        with pytest.raises(TypeError):
            ufunc(*args)
    else:
        truth = ufunc(*args_np)
        result = ufunc(*args)
        if ufunc.nout == 1:
            assert_valid_raster(result)
            truth[mask] = get_default_null_value(truth.dtype)
            assert result._masked
            assert result.dtype == truth.dtype
            assert result.crs == rs.crs
            assert np.allclose(result, truth, equal_nan=True)
            assert np.allclose(result._ds.mask, rs._ds.mask, equal_nan=True)
        else:
            for r, t in zip(result, truth):
                assert_valid_raster(r)
                t[mask] = get_default_null_value(t.dtype)
                assert r._masked
                assert r.dtype == t.dtype
                assert r.crs == rs.crs
                assert np.allclose(r, t, equal_nan=True)
                assert np.allclose(r._ds.mask, rs._ds.mask, equal_nan=True)
    # Reflected
    args = args[::-1]
    args_np = args_np[::-1]
    if (
        ufname.startswith("bitwise")
        or ufname.endswith("shift")
        or ufname in ("gcd", "lcm")
    ) and any(is_float(getattr(a, "dtype", a)) for a in args):
        with pytest.raises(TypeError):
            ufunc(*args)
    elif ufname == "ldexp" and is_float(args[-1].dtype):
        with pytest.raises(TypeError):
            ufunc(*args)
    else:
        truth = ufunc(*args_np)
        result = ufunc(*args)
        if ufunc.nout == 1:
            assert_valid_raster(result)
            truth[mask] = get_default_null_value(truth.dtype)
            assert result._masked
            assert result.dtype == truth.dtype
            assert result.crs == rs.crs
            assert np.allclose(result, truth, equal_nan=True)
            assert np.allclose(result._ds.mask, rs._ds.mask, equal_nan=True)
        else:
            for r, t in zip(result, truth):
                assert_valid_raster(r)
                t[mask] = get_default_null_value(t.dtype)
                assert r._masked
                assert r.dtype == t.dtype
                assert r.crs == rs.crs
                assert np.allclose(r, t, equal_nan=True)
                assert np.allclose(r._ds.mask, rs._ds.mask, equal_nan=True)


@pytest.mark.parametrize("ufunc", _NP_UFUNCS_NIN_MULT)
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.filterwarnings("ignore:overflow")
def test_ufuncs_multiple_input_against_raster(ufunc):
    x = arange_nd((4, 5, 5))
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), 0, rs._ds.raster
    )
    rs = rs.set_null_value(0).set_crs("EPSG:3857")
    mask = rs.mask.compute()
    assert rs.null_value == 0
    assert rs._masked
    assert rs.crs == "EPSG:3857"
    extra = ufunc.nin - 1
    other = Raster(np.ones_like(x) * 2)
    other.xdata.data[0, 0, 0] = 25
    other = other.set_null_value(25)
    assert other._ds.mask.data.compute().sum() == 1
    mask = rs._ds.mask | other._ds.mask

    args = [rs] + [other.copy() for i in range(extra)]
    args_np = [a.values for a in args]

    if ufunc.__name__.startswith("bitwise") and any(
        is_float(getattr(a, "dtype", type(a))) for a in args
    ):
        with pytest.raises(TypeError):
            ufunc(*args)
    else:
        truth = ufunc(*args_np)
        result = ufunc(*args)
        if ufunc.nout == 1:
            assert_valid_raster(result)
            truth[mask] = get_default_null_value(truth.dtype)
            assert result._masked
            assert result.dtype == truth.dtype
            assert result.crs == rs.crs
            assert np.allclose(result, truth, equal_nan=True)
            assert np.allclose(result._ds.mask, mask, equal_nan=True)
        else:
            for r, t in zip(result, truth):
                assert_valid_raster(r)
                t[mask] = get_default_null_value(t.dtype)
                assert r._masked
                assert r.dtype == t.dtype
                assert r.crs == rs.crs
                assert np.allclose(r, t, equal_nan=True)
                assert np.allclose(r._ds.mask, mask, equal_nan=True)
    # Test reflected
    args = args[::-1]
    args_np = args_np[::-1]
    if ufunc.__name__.startswith("bitwise") and any(
        is_float(getattr(a, "dtype", type(a))) for a in args
    ):
        with pytest.raises(TypeError):
            ufunc(*args)
    else:
        truth = ufunc(*args_np)
        result = ufunc(*args)
        if ufunc.nout == 1:
            assert_valid_raster(result)
            truth[mask] = get_default_null_value(truth.dtype)
            assert result.crs == rs.crs
            assert np.allclose(result, truth, equal_nan=True)
            assert np.allclose(result._ds.mask, mask)
        else:
            for r, t in zip(result, truth):
                assert_valid_raster(r)
                t[mask] = get_default_null_value(t.dtype)
                assert r.crs == rs.crs
                assert np.allclose(r, t, equal_nan=True)
                assert np.allclose(r._ds.mask, mask, equal_nan=True)


def test_invert():
    x = arange_nd((4, 5, 5))
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), 0, rs._ds.raster
    )
    rs = rs.set_null_value(0).set_crs("EPSG:3857")
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
    x = arange_nd((4, 5, 5))
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), 0, rs._ds.raster
    )
    rs = rs.set_null_value(0).set_crs("EPSG:3857")
    data = rs.values
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
        for r, t in zip(result, truth):
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
        for r, t in zip(result, truth):
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
    rs = Raster(arange_nd((4, 5, 5)))

    with pytest.raises(ValueError):
        rs.bandwise * other


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
    truth[rs.xmask.values[0]] = 24.5
    assert_valid_raster(rs.round())
    assert isinstance(rs.round(), Raster)
    assert np.allclose(rs.round(), truth)
    truth = np.round(data, decimals=2)
    truth[rs.xmask.values[0]] = 24.5
    assert np.allclose(rs.round(2), truth)
    assert rs.round().crs == rs.crs


@pytest.mark.filterwarnings("ignore:The null value ")
@pytest.mark.parametrize("type_code,dtype", list(DTYPE_INPUT_TO_DTYPE.items()))
@pytest.mark.parametrize(
    "rs",
    [
        Raster("tests/data/raster/elevation_small.tif"),
        Raster(np.arange(100).reshape((10, 10))).set_null_value(99),
    ],
)
def test_astype(rs, type_code, dtype):
    result = rs.astype(type_code)
    assert_valid_raster(result)
    assert result.dtype == dtype
    assert result.eval().dtype == dtype
    assert result.crs == rs.crs
    assert result.null_value == reconcile_nullvalue_with_dtype(
        rs.null_value, dtype
    )

    result = rs.astype(dtype)
    assert_valid_raster(rs.astype(dtype))
    assert result.dtype == dtype
    assert result.eval().dtype == dtype
    assert result.crs == rs.crs
    assert result.null_value == reconcile_nullvalue_with_dtype(
        rs.null_value, dtype
    )


@pytest.mark.filterwarnings("ignore:The null value ")
def test_astype_wrong_type_codes():
    rs = Raster("tests/data/raster/elevation_small.tif")
    with pytest.raises(ValueError):
        rs.astype("not float32")
    with pytest.raises(ValueError):
        rs.astype("other")


def test_copy():
    rs = Raster("tests/data/raster/elevation_clipped_small.tif")
    copy = rs.copy()
    assert_valid_raster(copy)
    assert rs is not copy
    assert rs._ds is not copy._ds
    assert rs._ds.equals(copy._ds)
    assert np.allclose(rs, copy)
    # make sure a deep copy has occurred
    copy._ds.raster.data[0, -1, -1] = 0
    assert not np.allclose(rs, copy)


def test_set_crs():
    rs = Raster("tests/data/raster/elevation_small.tif")
    assert rs.crs != 4326

    rs4326 = rs.set_crs(4326)
    assert rs4326.crs != rs.crs
    assert rs4326.crs == 4326
    assert np.allclose(rs.values, rs4326.values)


@pytest.mark.filterwarnings("ignore:The null value")
def test_set_null_value():
    rs = Raster("tests/data/raster/elevation_clipped_small.tif")
    assert rs.null_value is not None
    truth = rs.values
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

    rs = Raster(np.arange(100).reshape((10, 10)))
    assert rs.null_value is None
    assert rs._ds.raster.rio.nodata is None
    rs2 = rs.set_null_value(99)
    assert rs2.null_value == 99
    assert rs2._ds.raster.rio.nodata == 99
    assert np.allclose(rs2._ds.mask.values, rs2._ds.raster.values == 99)
    assert rs2.crs == rs2.crs

    rs = Raster("tests/data/raster/elevation_small.tif")
    nv = rs.null_value
    rs2 = rs.set_null_value(None)
    assert rs.null_value == nv
    assert rs2.null_value is None
    assert not rs2._ds.mask.values.any()
    assert np.allclose(rs, rs2)
    assert rs2.crs == rs2.crs

    rs = Raster("tests/data/raster/elevation_clipped_small.tif").astype(int)
    assert rs.dtype == np.dtype(int)
    rs2 = rs.set_null_value(get_default_null_value(float))
    assert rs2.dtype == np.dtype(float)
    assert rs2.crs == rs2.crs


def test_replace_null():
    fill_value = 0
    rs = Raster("tests/data/raster/elevation_clipped_small.tif")
    rsnp = rs.values
    rsnp_replaced = rsnp.copy()
    mask = rsnp == rs.null_value
    rsnp_replaced[mask] = fill_value
    rs = rs.replace_null(fill_value)
    assert np.allclose(rs.values, rsnp_replaced)
    assert rs.null_value == fill_value
    assert np.allclose(rs._ds.mask.values, mask)

    with pytest.raises(TypeError):
        rs.replace_null(None)


@pytest.mark.filterwarnings("ignore:The null value")
def test_where():
    rs = Raster("tests/data/raster/elevation_small.tif")
    c = rs > 1100

    r = rs.where(c, 0)
    assert_valid_raster(r)
    rsnp = np.asarray(rs)
    truth = np.where(rsnp > 1100, rsnp, 0)
    assert np.allclose(r, truth)
    assert np.allclose(
        rs.where(c, "tests/data/raster/elevation_small.tif"), rs
    )
    assert r.crs == rs.crs

    c = c.astype(int)
    r = rs.where(c, 0)
    assert_valid_raster(r)
    assert np.allclose(r, truth)

    assert rs._masked
    assert r._masked
    assert rs.crs is not None
    assert r.crs is not None
    assert r.crs == rs.crs
    assert r.null_value == get_default_null_value(r.dtype)

    with pytest.raises(TypeError):
        cf = c.astype(float)
        rs.where(cf, 0)
    with pytest.raises(TypeError):
        rs.where(c, None)


def test_to_null_mask():
    rs = Raster("tests/data/raster/elevation_clipped_small.tif")
    nv = rs.null_value
    rsnp = rs.values
    truth = rsnp == nv
    assert rs.null_value is not None
    nmask = rs.to_null_mask()
    assert_valid_raster(nmask)
    assert nmask.null_value is None
    assert np.allclose(nmask, truth)
    # Test case where no null values
    rs = Raster("tests/data/raster/elevation_small.tif")
    truth = np.full(rs.shape, False, dtype=bool)
    nmask = rs.to_null_mask()
    assert_valid_raster(nmask)
    assert nmask.null_value is None
    assert np.allclose(nmask, truth)


class TestEval(unittest.TestCase):
    def test_eval(self):
        rs = Raster("tests/data/raster/elevation_small.tif")
        rsnp = rs.xdata.values
        rs += 2
        rsnp += 2
        rs -= rs
        rsnp -= rsnp
        rs *= -1
        rsnp *= -1
        result = rs.eval()
        assert_valid_raster(result)
        # Make sure new raster returned
        self.assertIsNot(rs, result)
        self.assertIsNot(rs._ds, result._ds)
        # Make sure that original raster is still lazy
        assert np.allclose(result, rsnp)


def test_get_bands():
    rs = Raster(np.arange(4 * 100 * 100).reshape((4, 100, 100)))
    rsnp = rs.values
    r = rs.get_bands(1)
    assert_valid_raster(r)
    assert np.allclose(r, rsnp[:1])
    r = rs.get_bands(2)
    assert_valid_raster(r)
    assert np.allclose(r, rsnp[1:2])
    r = rs.get_bands(3)
    assert_valid_raster(r)
    assert np.allclose(r, rsnp[2:3])
    r = rs.get_bands(4)
    assert_valid_raster(r)
    assert np.allclose(r, rsnp[3:4])
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


def test_burn_mask():
    x = arange_nd((1, 5, 5))
    rs = Raster(x)
    rs._ds["raster"] = xr.where(
        (0 <= rs._ds.raster) & (rs._ds.raster < 10), -999, rs._ds.raster
    )
    rs = rs.set_null_value(-999).set_crs("EPSG:3857")
    data = rs.values
    assert rs.null_value == -999
    assert rs._masked
    assert rs.crs == "EPSG:3857"
    true_mask = data < 10
    true_state = data.copy()
    true_state[true_mask] = -999
    assert np.allclose(true_mask, rs._ds.mask)
    assert np.allclose(true_state, rs)

    rs._ds.raster.data = data
    assert np.allclose(rs, data)
    assert_valid_raster(rs.burn_mask())
    assert np.allclose(rs.burn_mask(), true_state)
    assert rs.burn_mask().crs == rs.crs

    data = arange_nd((1, 5, 5))
    rs = Raster(data)
    rs._ds["raster"] = xr.where(
        (20 <= rs._ds.raster) & (rs._ds.raster < 26), 999, rs._ds.raster
    )
    rs = rs.set_null_value(999).set_crs("EPSG:3857")
    rs = rs > 15
    assert rs.dtype == np.dtype(bool)
    nv = get_default_null_value(bool)
    assert rs.null_value == nv
    true_state = data > 15
    true_state = np.where(data >= 20, nv, true_state)
    assert np.allclose(rs.burn_mask(), true_state)


@pytest.mark.parametrize("index", [(0, 0), (0, 1), (1, 0), (-1, -1), (1, -1)])
@pytest.mark.parametrize("offset", ["center", "ul", "ll", "ur", "lr"])
def test_xy(index, offset):
    rs = Raster("tests/data/raster/elevation_small.tif")
    i, j = index
    if i == -1:
        i = rs.shape[1] - 1
    if j == -1:
        j = rs.shape[2] - 1

    T = rs.affine
    assert rs.xy(i, j, offset) == rio.transform.xy(T, i, j, offset=offset)


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
    rs = Raster("tests/data/raster/elevation_small.tif")
    T = rs.affine

    assert rs.index(x, y) == rio.transform.rowcol(T, x, y)


@pytest.mark.parametrize("offset_name", ["center", "ul", "ur", "ll", "lr"])
def test_rowcol_to_xy(offset_name):
    rs = Raster("tests/data/raster/elevation_small.tif")
    affine = rs.affine
    _, ny, nx = rs.shape
    R, C = np.mgrid[:ny, :nx]
    r = R.ravel()
    c = C.ravel()
    xy_rio = rio.transform.xy(affine, r, c, offset=offset_name)
    result = rowcol_to_xy(r, c, affine, offset_name)
    assert np.allclose(xy_rio, result)


def test_xy_to_rowcol():
    rs = Raster("tests/data/raster/elevation_small.tif")
    affine = rs.affine
    X, Y = np.meshgrid(rs.x, rs.y)
    x = X.ravel()
    y = Y.ravel()
    rc_rio = rio.transform.rowcol(affine, x, y)
    result = xy_to_rowcol(x, y, affine)
    assert np.allclose(rc_rio, result)
    assert result[0].dtype == np.dtype(int)
    assert result[1].dtype == np.dtype(int)


def _compare_raster_to_vectorized(rs, df):
    data = rs.values
    x = rs.xdata.x.values
    y = rs.xdata.y.values
    for dfrow in df.itertuples():
        value = dfrow.value
        band = dfrow.band
        row = dfrow.row
        col = dfrow.col
        p = dfrow.geometry
        assert band >= 1
        assert value == data[band - 1, row, col]
        assert x[col] == p.x
        assert y[row] == p.y


def test_to_vector():
    data = np.array(
        [
            [
                [0, 1, 1, 2],
                [0, 1, 2, 2],
                [0, 0, 1, 0],
                [0, 1, 3, 0],
            ],
            [
                [3, 2, 2, 1],
                [0, 1, 2, 1],
                [0, 0, 2, 0],
                [0, 2, 4, 5],
            ],
        ]
    )
    count = np.sum(data > 0)
    rs = Raster(data).set_null_value(0)
    ddf = rs.to_vector()
    df = ddf.compute()

    assert isinstance(ddf, GeoDataFrame)
    assert ddf.crs == rs.crs
    assert len(df) == count
    assert np.all(ddf.columns == ["value", "band", "row", "col", "geometry"])
    _compare_raster_to_vectorized(rs, df)

    rs = Raster("tests/data/raster/elevation_small.tif")
    rs = band_concat((rs, rs + 100))
    data = rs.values
    rs = rs.chunk((1, 20, 20))
    ddf = rs.to_vector()
    df = ddf.compute()

    assert rs.data.npartitions == 50
    assert rs._ds.mask.values.sum() == 0
    assert ddf.npartitions == rs.data.npartitions
    assert ddf.crs == rs.crs
    assert len(df) == rs.data.size
    assert np.all(ddf.columns == ["value", "band", "row", "col", "geometry"])
    _compare_raster_to_vectorized(rs, df)

    # make sure that empty (all-null) chunks are handled
    data = np.array(
        [
            [
                [0, 0, 1, 2],
                [0, 0, 2, 2],
                [0, 0, 1, 0],
                [0, 1, 3, 0],
            ]
        ]
    )
    count = np.sum(data > 0)
    rs = Raster(data).set_null_value(0).chunk((1, 2, 2))
    ddf = rs.to_vector()
    df = ddf.compute()

    assert len(df) == count
    assert df["value"].dtype == data.dtype
    assert df["band"].dtype == np.dtype(int)
    assert df["row"].dtype == np.dtype(int)
    assert df["col"].dtype == np.dtype(int)
    _compare_raster_to_vectorized(rs, df)


@pytest.mark.parametrize(
    "raster",
    [
        Raster("tests/data/raster/elevation.tif"),
        arange_raster((3, 35, 50))
        .set_crs("EPSG:3857")
        .remap_range([(0, 10, 0), (3490, 3500, 0), (5200, 5210, 0)])
        .set_null_value(0),
        arange_raster((3, 1, 1)),
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


if __name__ == "__main__":
    unittest.main()
