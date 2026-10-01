"""Direct tests for the public raster construction functions.

Covers the data_to_* and dataarray_to_* families in raster_tools.raster.
"""

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine

import raster_tools as rts
from raster_tools import Raster
from raster_tools.masking import get_default_null_value
from tests import testdata
from tests.utils import (
    arange_nd,
    assert_dataarrays_similar,
    assert_rasters_similar,
)

# Arbitrary small grid used where a real-world template is not needed
SMALL_AFFINE = Affine(2.0, 0.0, 100.0, 0.0, -2.0, 200.0)


def _small_coords(ny, nx):
    # Cell center coordinates matching SMALL_AFFINE
    x = 100.0 + 2.0 * (np.arange(nx) + 0.5)
    y = 200.0 - 2.0 * (np.arange(ny) + 0.5)
    return x, y


def _xlike_3band():
    xlike = testdata.raster.dem_small.xdata
    xlike = xr.concat([xlike] * 3, dim="band", join="inner")
    xlike["band"] = np.arange(1, 4)
    return xlike.chunk({"band": 1, "y": 25, "x": 50})


def _assert_nodata(xobj, nv):
    if nv is None:
        assert xobj.rio.nodata is None
    elif np.isnan(nv):
        assert np.isnan(xobj.rio.nodata)
    else:
        assert xobj.rio.nodata == nv


# --------------------------------------------------------------------------
# data_to_xr_raster
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        arange_nd((3, 4)).astype(float),
        arange_nd((2, 3, 4)),
        da.from_array(arange_nd((2, 3, 4)), chunks=(2, 2, 2)),
    ],
)
def test_data_to_xr_raster_default_coords(data):
    result = rts.data_to_xr_raster(data)

    nbands = 1 if data.ndim == 2 else data.shape[0]
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("band", "y", "x")
    assert result.shape == (nbands, 3, 4)
    assert dask.is_dask_collection(result)
    assert result.data.chunksize[0] == 1
    assert result.dtype == data.dtype
    np.testing.assert_array_equal(result.band, np.arange(1, nbands + 1))
    np.testing.assert_array_equal(result.x, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_array_equal(result.y, [2.5, 1.5, 0.5])
    assert result.rio.transform(True) == Affine(1, 0, 0, 0, -1, 3)
    assert result.rio.crs is None
    assert result.rio.nodata is None
    expected = np.asarray(data).reshape((nbands, 3, 4))
    np.testing.assert_array_equal(result.to_numpy(), expected)


@pytest.mark.parametrize("use_affine", [False, True])
def test_data_to_xr_raster_no_crs(use_affine):
    data = arange_nd((3, 4)).astype(float)
    if use_affine:
        result = rts.data_to_xr_raster(data, affine=SMALL_AFFINE)
    else:
        x, y = _small_coords(3, 4)
        result = rts.data_to_xr_raster(data, x=x, y=y)

    assert result.rio.crs is None
    assert result.rio.transform(True) == SMALL_AFFINE
    x, y = _small_coords(3, 4)
    np.testing.assert_allclose(result.x, x)
    np.testing.assert_allclose(result.y, y)


@pytest.mark.parametrize("crs", [3310, "EPSG:3310"])
def test_data_to_xr_raster_crs_inputs(crs):
    result = rts.data_to_xr_raster(
        arange_nd((3, 4)), affine=SMALL_AFFINE, crs=crs
    )
    assert result.rio.crs == "EPSG:3310"


def test_data_to_xr_raster_increasing_y_is_flipped():
    data = arange_nd((3, 4)).astype(float)
    x, y = _small_coords(3, 4)
    result = rts.data_to_xr_raster(data, x=x, y=y[::-1])

    np.testing.assert_allclose(result.y, y)
    np.testing.assert_array_equal(result.to_numpy()[0], data[::-1])


@pytest.mark.parametrize("nv", [None, np.nan, 0, -9])
def test_data_to_xr_raster_nv_int_and_float(nv):
    data = arange_nd((3, 4), dtype="int16")
    if nv is not None and np.isnan(nv):
        data = data.astype("float32")
    result = rts.data_to_xr_raster(data, nv=nv)

    assert result.dtype == data.dtype
    _assert_nodata(result, nv)


@pytest.mark.parametrize(
    "x,y",
    [
        # Valid arrays describing a different grid
        (np.arange(4.0) * 1000, np.arange(3.0)[::-1] * 1000),
        # Arrays with sizes that do not match the data
        (np.arange(7.0), np.arange(9.0)),
        # Not arrays at all
        ([0, 1, 2, 3], "junk"),
        # Only one of x and y
        (np.arange(4.0), None),
    ],
)
def test_data_to_xr_raster_affine_takes_precedence_over_xy(x, y):
    # When affine is given, x and y are ignored without any validation
    data = arange_nd((3, 4)).astype(float)
    result = rts.data_to_xr_raster(data, x=x, y=y, affine=SMALL_AFFINE)

    ex, ey = _small_coords(3, 4)
    np.testing.assert_allclose(result.x, ex)
    np.testing.assert_allclose(result.y, ey)
    assert result.rio.transform(True) == SMALL_AFFINE


@pytest.mark.parametrize(
    "x,y",
    [
        (np.arange(4.0), [2.0, 1.0, 0.0]),
        ([0.0, 1.0, 2.0, 3.0], np.arange(3.0)),
        ([0.0, 1.0, 2.0, 3.0], [2.0, 1.0, 0.0]),
        (np.arange(4.0), (2.0, 1.0, 0.0)),
        (np.arange(4.0), da.arange(3.0)),
    ],
)
def test_data_to_xr_raster_non_array_xy_raises(x, y):
    data = arange_nd((3, 4)).astype(float)
    with pytest.raises(TypeError, match="x and y must be numpy arrays"):
        rts.data_to_xr_raster(data, x=x, y=y)


@pytest.mark.parametrize(
    "x,y",
    [(np.arange(4.0), None), (None, np.arange(3.0))],
)
def test_data_to_xr_raster_only_one_of_xy_raises(x, y):
    data = arange_nd((3, 4)).astype(float)
    with pytest.raises(ValueError, match="both x and y or neither"):
        rts.data_to_xr_raster(data, x=x, y=y)


@pytest.mark.parametrize(
    "x,y",
    [
        (np.arange(5.0), np.arange(3.0)),
        (np.arange(4.0), np.arange(2.0)),
        # Swapped
        (np.arange(3.0), np.arange(4.0)),
    ],
)
def test_data_to_xr_raster_xy_shape_mismatch_raises(x, y):
    data = arange_nd((3, 4)).astype(float)
    with pytest.raises(ValueError, match="do not match data shape"):
        rts.data_to_xr_raster(data, x=x, y=y)


def test_data_to_xr_raster_ravels_xy():
    data = arange_nd((3, 4)).astype(float)
    x, y = _small_coords(3, 4)
    result = rts.data_to_xr_raster(data, x=x[None], y=y[:, None])

    np.testing.assert_allclose(result.x, x)
    np.testing.assert_allclose(result.y, y)


@pytest.mark.parametrize(
    "data,error",
    [
        (arange_nd((3, 4)).tolist(), TypeError),
        (xr.DataArray(arange_nd((3, 4))), TypeError),
        (np.arange(4.0), ValueError),
        (np.ones((1, 1, 3, 4)), ValueError),
    ],
)
def test_data_to_xr_raster_invalid_data_raises(data, error):
    with pytest.raises(error):
        rts.data_to_xr_raster(data)


# --------------------------------------------------------------------------
# data_to_xr_raster_like
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        np.ones((100, 100)),
        np.ones((1, 100, 100)),
        da.ones((1, 100, 100), chunks=(1, 10, 10)),
    ],
)
def test_data_to_xr_raster_like_match_band_dim(data):
    xlike = _xlike_3band()
    result = rts.data_to_xr_raster_like(data, xlike, match_band_dim=True)

    assert result.shape == (3, 100, 100)
    np.testing.assert_array_equal(result.band, [1, 2, 3])
    assert result.data.chunksize[0] == 1
    assert_dataarrays_similar(result, xlike)
    np.testing.assert_array_equal(result.to_numpy(), 1)


def test_data_to_xr_raster_like_match_band_dim_multiband_noop():
    xlike = _xlike_3band()
    data = np.ones((2, 100, 100))
    result = rts.data_to_xr_raster_like(data, xlike, match_band_dim=True)

    assert result.shape == (2, 100, 100)


def test_data_to_xr_raster_like_no_match_band_dim():
    xlike = _xlike_3band()
    result = rts.data_to_xr_raster_like(np.ones((100, 100)), xlike)

    assert result.shape == (1, 100, 100)
    assert_dataarrays_similar(result, xlike, check_nbands=False)


@pytest.mark.parametrize(
    "data,expected_chunks",
    [
        # Numpy input is chunked with "auto", which is a single chunk here
        (np.ones((100, 100)), ((100,), (100,))),
        # Dask input keeps its own chunks
        (
            da.ones((100, 100), chunks=(30, 40)),
            ((30, 30, 30, 10), (40, 40, 20)),
        ),
    ],
)
def test_data_to_xr_raster_like_no_match_chunks(data, expected_chunks):
    xlike = _xlike_3band()
    result = rts.data_to_xr_raster_like(data, xlike, match_chunks=False)

    assert result.data.chunks[1:] == expected_chunks
    assert result.data.chunks[1:] != xlike.data.chunks[1:]
    assert_dataarrays_similar(
        result, xlike, check_nbands=False, check_chunks=False
    )


def test_data_to_xr_raster_like_template_without_crs():
    xlike = rts.data_to_xr_raster(arange_nd((3, 4)), affine=SMALL_AFFINE)
    result = rts.data_to_xr_raster_like(np.zeros((3, 4)), xlike, nv=-1)

    assert result.rio.crs is None
    assert result.rio.transform(True) == SMALL_AFFINE
    assert result.rio.nodata == -1


@pytest.mark.parametrize("match_chunks", [False, True])
@pytest.mark.parametrize("mod", [np, da])
@pytest.mark.parametrize("shape", [(100, 99), (99, 100), (1, 100, 101)])
def test_data_to_xr_raster_like_shape_mismatch_raises(
    shape, mod, match_chunks
):
    xlike = testdata.raster.dem_small.xdata
    # With match_chunks, rechunking to the template's chunks fails before
    # the explicit shape check is reached, so only the type is pinned.
    match = None if match_chunks else "did not match xlike"
    with pytest.raises(ValueError, match=match):
        rts.data_to_xr_raster_like(
            mod.ones(shape), xlike, match_chunks=match_chunks
        )


# --------------------------------------------------------------------------
# data_to_xr_raster_ds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nv", [None, np.nan, 5.0])
@pytest.mark.parametrize(
    "data",
    [
        np.array([[0.0, 5.0, np.nan], [5.0, 1.0, 2.0]]),
        da.from_array(
            np.array([[[0.0, 5.0, np.nan], [5.0, 1.0, 2.0]]] * 2),
            chunks=(1, 1, 2),
        ),
    ],
)
def test_data_to_xr_raster_ds_mask_from_nv(data, nv):
    ds = rts.data_to_xr_raster_ds(data, nv=nv, crs=3310)

    npdata = np.asarray(data)
    if npdata.ndim == 2:
        npdata = npdata[None]
    if nv is None:
        expected_mask = np.zeros(npdata.shape, dtype=bool)
    elif np.isnan(nv):
        expected_mask = np.isnan(npdata)
    else:
        expected_mask = npdata == nv

    assert isinstance(ds, xr.Dataset)
    assert sorted(ds.data_vars) == ["mask", "raster"]
    assert ds.mask.dtype == bool
    assert dask.is_dask_collection(ds.raster)
    assert dask.is_dask_collection(ds.mask)
    assert ds.raster.data.chunksize[0] == 1
    assert ds.mask.data.chunksize[0] == 1
    np.testing.assert_array_equal(ds.mask.to_numpy(), expected_mask)
    # Data is never altered when the mask is derived from nv
    np.testing.assert_array_equal(ds.raster.to_numpy(), npdata)
    _assert_nodata(ds.raster, nv)
    assert ds.mask.rio.nodata is None
    assert ds.raster.rio.crs == "EPSG:3310"
    assert ds.mask.rio.crs == "EPSG:3310"
    assert_dataarrays_similar(ds.raster, ds.mask)


@pytest.mark.parametrize("burn", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "int16", "uint8"])
def test_data_to_xr_raster_ds_mask_default_nv(dtype, burn):
    data = arange_nd((3, 4), dtype=dtype)
    mask = data > 5
    ds = rts.data_to_xr_raster_ds(data, mask=mask, burn=burn)

    nv = get_default_null_value(data.dtype)
    assert ds.raster.rio.nodata == nv
    assert ds.raster.dtype == data.dtype
    assert ds.mask.rio.nodata is None
    np.testing.assert_array_equal(ds.mask.to_numpy()[0], mask)
    expected = np.where(mask, nv, data) if burn else data
    np.testing.assert_array_equal(ds.raster.to_numpy()[0], expected)


@pytest.mark.parametrize("burn", [False, True])
@pytest.mark.parametrize("nv", [-1.0, np.nan])
def test_data_to_xr_raster_ds_mask_with_nv(nv, burn):
    data = arange_nd((2, 3, 4)).astype(float)
    mask = da.from_array(data % 3 == 0, chunks=(2, 1, 4))
    ds = rts.data_to_xr_raster_ds(data, mask=mask, nv=nv, burn=burn)

    _assert_nodata(ds.raster, nv)
    np.testing.assert_array_equal(ds.mask.to_numpy(), data % 3 == 0)
    expected = np.where(data % 3 == 0, nv, data) if burn else data
    np.testing.assert_array_equal(ds.raster.to_numpy(), expected)


def test_data_to_xr_raster_ds_burn_without_mask_is_noop():
    data = np.array([[0.0, 5.0], [5.0, 1.0]])
    ds = rts.data_to_xr_raster_ds(data, nv=5.0, burn=True)

    np.testing.assert_array_equal(ds.raster.to_numpy()[0], data)
    np.testing.assert_array_equal(ds.mask.to_numpy()[0], data == 5)


@pytest.mark.parametrize("use_affine", [False, True])
def test_data_to_xr_raster_ds_georeferencing(use_affine):
    data = arange_nd((3, 4)).astype(float)
    mask = data > 6
    if use_affine:
        # x/y are ignored when affine is given
        kwargs = {"affine": SMALL_AFFINE, "x": "junk", "y": "junk"}
    else:
        x, y = _small_coords(3, 4)
        kwargs = {"x": x, "y": y}
    ds = rts.data_to_xr_raster_ds(data, mask=mask, **kwargs)

    ex, ey = _small_coords(3, 4)
    for var in (ds.raster, ds.mask):
        np.testing.assert_allclose(var.x, ex)
        np.testing.assert_allclose(var.y, ey)
        assert var.rio.transform(True) == SMALL_AFFINE
        assert var.rio.crs is None


@pytest.mark.parametrize("mask_shape", [(3, 5), (2, 3, 4), (1, 4, 4)], ids=str)
def test_data_to_xr_raster_ds_mask_shape_mismatch_raises(mask_shape):
    data = arange_nd((3, 4)).astype(float)
    mask = np.zeros(mask_shape, dtype=bool)
    with pytest.raises(ValueError, match="data and mask dimensions"):
        rts.data_to_xr_raster_ds(data, mask=mask)


def test_data_to_xr_raster_ds_invalid_mask_type_raises():
    data = arange_nd((3, 4)).astype(float)
    with pytest.raises(TypeError):
        rts.data_to_xr_raster_ds(data, mask=(data > 5).tolist())


# --------------------------------------------------------------------------
# data_to_xr_raster_ds_like
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        arange_nd((100, 100)).astype(float),
        arange_nd((3, 100, 100)).astype(float),
        da.from_array(arange_nd((3, 100, 100)).astype(float), chunks=7),
    ],
)
def test_data_to_xr_raster_ds_like_match_chunks(data):
    xlike = _xlike_3band()
    mask = data > 500
    ds = rts.data_to_xr_raster_ds_like(data, xlike, mask=mask)

    nbands = 1 if data.ndim == 2 else 3
    for var in (ds.raster, ds.mask):
        assert var.shape == (nbands, 100, 100)
        assert var.data.chunksize[0] == 1
        assert_dataarrays_similar(var, xlike, check_nbands=False)


def test_data_to_xr_raster_ds_like_no_match_chunks():
    xlike = _xlike_3band()
    data = np.ones((100, 100))
    mask = da.from_array(data > 0, chunks=(20, 20))
    ds = rts.data_to_xr_raster_ds_like(
        data, xlike, mask=mask, match_chunks=False
    )

    assert ds.raster.data.chunks[1:] == ((100,), (100,))
    assert ds.mask.data.chunks[1:] == ((20,) * 5, (20,) * 5)
    for var in (ds.raster, ds.mask):
        assert_dataarrays_similar(
            var, xlike, check_nbands=False, check_chunks=False
        )


@pytest.mark.parametrize("nv", [None, np.nan, 3.0])
def test_data_to_xr_raster_ds_like_mask_from_nv(nv):
    xlike = testdata.raster.dem_small.xdata
    data = arange_nd((100, 100)).astype(float) % 7
    data[0, 0] = np.nan
    ds = rts.data_to_xr_raster_ds_like(data, xlike, nv=nv)

    if nv is None:
        expected_mask = np.zeros(data.shape, dtype=bool)
    elif np.isnan(nv):
        expected_mask = np.isnan(data)
    else:
        expected_mask = data == nv
    _assert_nodata(ds.raster, nv)
    assert ds.mask.rio.nodata is None
    np.testing.assert_array_equal(ds.mask.to_numpy()[0], expected_mask)
    np.testing.assert_array_equal(ds.raster.to_numpy()[0], data)
    assert ds.raster.rio.crs == xlike.rio.crs


@pytest.mark.parametrize("burn", [False, True])
@pytest.mark.parametrize("nv", [None, -5])
def test_data_to_xr_raster_ds_like_mask_and_burn(nv, burn):
    xlike = testdata.raster.dem_small.xdata
    data = arange_nd((100, 100), dtype="int32")
    mask = data % 11 == 0
    ds = rts.data_to_xr_raster_ds_like(
        data, xlike, mask=mask, nv=nv, burn=burn
    )

    expected_nv = get_default_null_value(data.dtype) if nv is None else nv
    assert ds.raster.rio.nodata == expected_nv
    assert ds.raster.dtype == data.dtype
    np.testing.assert_array_equal(ds.mask.to_numpy()[0], mask)
    expected = np.where(mask, expected_nv, data) if burn else data
    np.testing.assert_array_equal(ds.raster.to_numpy()[0], expected)


@pytest.mark.parametrize(
    "data_shape,mask_shape,match_chunks,match",
    [
        ((100, 99), (100, 99), False, "did not match xlike"),
        ((100, 100), (100, 99), False, "data and mask dimensions"),
        ((100, 100), (2, 100, 100), False, "data and mask dimensions"),
        ((100, 100), (2, 100, 100), True, "data and mask dimensions"),
        # Rechunking to the template's chunks fails first
        ((100, 99), (100, 99), True, None),
        ((100, 100), (100, 99), True, None),
    ],
)
def test_data_to_xr_raster_ds_like_shape_mismatch_raises(
    data_shape, mask_shape, match_chunks, match
):
    xlike = testdata.raster.dem_small.xdata
    data = np.ones(data_shape)
    mask = np.zeros(mask_shape, dtype=bool)
    with pytest.raises(ValueError, match=match):
        rts.data_to_xr_raster_ds_like(
            data, xlike, mask=mask, match_chunks=match_chunks
        )


# --------------------------------------------------------------------------
# data_to_raster
# --------------------------------------------------------------------------


@pytest.mark.parametrize("burn", [False, True])
def test_data_to_raster_mask_and_affine(burn):
    data = da.from_array(arange_nd((2, 3, 4), dtype="int16"))
    mask = data < 4
    rs = rts.data_to_raster(
        data, mask=mask, affine=SMALL_AFFINE, crs=3310, burn=burn
    )

    nv = get_default_null_value(data.dtype)
    assert isinstance(rs, Raster)
    assert rs.crs == "EPSG:3310"
    assert rs.affine == SMALL_AFFINE
    assert rs.null_value == nv
    assert rs.dtype == data.dtype
    assert dask.is_dask_collection(rs.data)
    np.testing.assert_array_equal(rs.mask.compute(), mask)
    expected = np.where(mask, nv, data) if burn else data
    np.testing.assert_array_equal(rs.data.compute(), expected)


def test_data_to_raster_xy_and_nv():
    data = np.array(
        [
            [0.0, 2.0, np.nan, 1.0],
            [2.0, 1.0, 4.0, 2.0],
            [np.nan, 3.0, 5.0, 6.0],
        ]
    )
    x, y = _small_coords(3, 4)
    rs = rts.data_to_raster(data, x=x, y=y, nv=np.nan)

    assert rs.crs is None
    assert rs.affine == SMALL_AFFINE
    assert np.isnan(rs.null_value)
    np.testing.assert_allclose(rs.x, x)
    np.testing.assert_allclose(rs.y, y)
    np.testing.assert_array_equal(rs.mask.compute()[0], np.isnan(data))


# --------------------------------------------------------------------------
# data_to_raster_like
# --------------------------------------------------------------------------


@pytest.mark.parametrize("like_type", ["raster", "dataarray"])
def test_data_to_raster_like_template_types(like_type):
    like = testdata.raster.dem_small
    if like_type == "dataarray":
        like = like.xdata
    data = arange_nd((100, 100)).astype(float)
    rs = rts.data_to_raster_like(data, like, nv=0.0)

    assert isinstance(rs, Raster)
    assert_rasters_similar(rs, testdata.raster.dem_small)
    assert rs.null_value == 0.0
    expected_mask = (data == 0)[None]
    np.testing.assert_array_equal(rs.mask.compute(), expected_mask)
    np.testing.assert_array_equal(rs.data.compute()[0], data)


def test_data_to_raster_like_passes_options():
    like = Raster(_xlike_3band())
    data = da.from_array(
        arange_nd((3, 100, 100)).astype("float32"), chunks=(1, 50, 20)
    )
    mask = data % 13 == 0
    rs = rts.data_to_raster_like(
        data, like, mask=mask, burn=True, match_chunks=False
    )

    nv = get_default_null_value(data.dtype)
    assert_rasters_similar(rs, like, check_chunks=False)
    assert rs.data.chunks[1:] == ((50, 50), (20,) * 5)
    assert rs.null_value == nv
    np.testing.assert_array_equal(rs.mask.compute(), mask)
    np.testing.assert_array_equal(rs.data.compute(), np.where(mask, nv, data))


# --------------------------------------------------------------------------
# dataarray_to_xr_raster
# --------------------------------------------------------------------------


def _latlon_dataarray(nodata=None, crs=4326):
    # 2D, numpy backed, with increasing y and lat/lon dim names
    xdata = xr.DataArray(
        arange_nd((3, 4)).astype(float),
        dims=("lat", "lon"),
        coords={"lat": [0.5, 1.5, 2.5], "lon": [0.5, 1.5, 2.5, 3.5]},
    )
    if crs is not None:
        xdata = xdata.rio.write_crs(crs)
    if nodata is not None:
        xdata = xdata.rio.write_nodata(nodata)
    return xdata


def test_dataarray_to_xr_raster_normalizes():
    xdata = _latlon_dataarray(nodata=5.0)
    result = rts.dataarray_to_xr_raster(xdata)

    assert result.dims == ("band", "y", "x")
    assert dask.is_dask_collection(result)
    assert result.data.chunksize[0] == 1
    np.testing.assert_array_equal(result.band, [1])
    np.testing.assert_array_equal(result.x, [0.5, 1.5, 2.5, 3.5])
    # y is flipped to be decreasing and the data is flipped with it
    np.testing.assert_array_equal(result.y, [2.5, 1.5, 0.5])
    np.testing.assert_array_equal(result.to_numpy()[0], xdata.to_numpy()[::-1])
    assert result.rio.crs == "EPSG:4326"
    assert result.rio.nodata == 5.0
    assert result.rio.transform(True) == Affine(1, 0, 0, 0, -1, 3)


def test_dataarray_to_xr_raster_time_dim_becomes_band():
    xdata = xr.DataArray(
        np.ones((2, 3, 4)),
        dims=("time", "y", "x"),
        coords={
            "time": [10, 20],
            "y": [2.5, 1.5, 0.5],
            "x": [0.5, 1.5, 2.5, 3.5],
        },
    )
    result = rts.dataarray_to_xr_raster(xdata)

    assert result.dims == ("band", "y", "x")
    np.testing.assert_array_equal(result.band, [1, 2])
    assert result.rio.crs is None
    assert result.rio.nodata is None


def test_dataarray_to_xr_raster_from_raster_xdata():
    rs = testdata.raster.dem_small
    result = rts.dataarray_to_xr_raster(rs.xdata)

    assert_dataarrays_similar(result, rs.xdata)
    assert result.rio.nodata == rs.null_value


@pytest.mark.parametrize("shape", [(4,), (1, 1, 3, 4)], ids=str)
def test_dataarray_to_xr_raster_invalid_ndim_raises(shape):
    with pytest.raises(ValueError, match="Invalid shape"):
        rts.dataarray_to_xr_raster(xr.DataArray(np.ones(shape)))


# --------------------------------------------------------------------------
# dataarray_to_xr_raster_ds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nodata", [None, np.nan, 5.0])
def test_dataarray_to_xr_raster_ds_mask_from_nodata(nodata):
    xdata = _latlon_dataarray(nodata=nodata)
    xdata[0, 1] = np.nan
    ds = rts.dataarray_to_xr_raster_ds(xdata)

    values = xdata.to_numpy()[::-1]
    if nodata is None:
        expected_mask = np.zeros(values.shape, dtype=bool)
    elif np.isnan(nodata):
        expected_mask = np.isnan(values)
    else:
        expected_mask = values == nodata

    assert isinstance(ds, xr.Dataset)
    assert sorted(ds.data_vars) == ["mask", "raster"]
    assert ds.mask.dtype == bool
    assert dask.is_dask_collection(ds.mask)
    np.testing.assert_array_equal(ds.mask.to_numpy()[0], expected_mask)
    _assert_nodata(ds.raster, nodata)
    assert ds.mask.rio.nodata is None
    assert ds.raster.rio.crs == "EPSG:4326"
    assert_dataarrays_similar(ds.raster, ds.mask)


def test_dataarray_to_xr_raster_ds_explicit_mask():
    xdata = _latlon_dataarray(nodata=5.0)
    xmask = xr.zeros_like(xdata, dtype=bool)
    xmask[0, 0] = True
    ds = rts.dataarray_to_xr_raster_ds(xdata, xmask=xmask)

    # The mask is normalized the same way as the data, including the y flip,
    # and is used instead of a mask derived from the nodata value
    expected_mask = np.zeros((3, 4), dtype=bool)
    expected_mask[-1, 0] = True
    np.testing.assert_array_equal(ds.mask.to_numpy()[0], expected_mask)
    assert ds.raster.rio.nodata == 5.0
    assert_dataarrays_similar(ds.raster, ds.mask)


def _mask_like_latlon(lat=None, lon=None, nbands=None, dtype=bool):
    lat = [0.5, 1.5, 2.5] if lat is None else lat
    lon = [0.5, 1.5, 2.5, 3.5] if lon is None else lon
    shape = (len(lat), len(lon))
    dims = ("lat", "lon")
    coords = {"lat": lat, "lon": lon}
    if nbands is not None:
        shape = (nbands, *shape)
        dims = ("band", *dims)
        coords["band"] = np.arange(1, nbands + 1)
    return xr.DataArray(np.zeros(shape, dtype=dtype), dims=dims, coords=coords)


@pytest.mark.parametrize(
    "xmask",
    [
        _mask_like_latlon(lat=[0.5, 1.5]),
        _mask_like_latlon(lon=[0.5, 1.5, 2.5, 3.5, 4.5]),
        _mask_like_latlon(lon=[0.75, 1.75, 2.75, 3.75]),
        _mask_like_latlon(nbands=2),
    ],
    ids=["fewer_rows", "more_cols", "shifted_x", "extra_band"],
)
@pytest.mark.parametrize(
    "func", [rts.dataarray_to_xr_raster_ds, rts.dataarray_to_raster]
)
def test_dataarray_to_xr_raster_ds_mismatched_mask_raises(func, xmask):
    xdata = _latlon_dataarray(nodata=5.0)
    with pytest.raises(ValueError, match="must match xdata"):
        func(xdata, xmask=xmask)


@pytest.mark.parametrize("dtype", ["float64", "int8", "uint8"])
@pytest.mark.parametrize(
    "func", [rts.dataarray_to_xr_raster_ds, rts.dataarray_to_raster]
)
def test_dataarray_to_xr_raster_ds_non_bool_mask_raises(func, dtype):
    xdata = _latlon_dataarray(nodata=5.0)
    with pytest.raises(TypeError, match="boolean dtype"):
        func(xdata, xmask=_mask_like_latlon(dtype=dtype))


@pytest.mark.parametrize("src_crs", [None, 4326])
def test_dataarray_to_xr_raster_ds_crs_override(src_crs):
    xdata = _latlon_dataarray(crs=src_crs)
    ds = rts.dataarray_to_xr_raster_ds(xdata, crs=3310)

    assert ds.rio.crs == "EPSG:3310"
    assert ds.raster.rio.crs == "EPSG:3310"
    assert ds.mask.rio.crs == "EPSG:3310"


def test_dataarray_to_xr_raster_ds_no_crs():
    ds = rts.dataarray_to_xr_raster_ds(_latlon_dataarray(crs=None))

    assert ds.raster.rio.crs is None
    assert ds.mask.rio.crs is None


# --------------------------------------------------------------------------
# dataarray_to_raster
# --------------------------------------------------------------------------


def test_dataarray_to_raster_reads_rio_metadata():
    src = testdata.raster.dem_clipped_small
    rs = rts.dataarray_to_raster(src.xdata)

    assert isinstance(rs, Raster)
    assert_rasters_similar(rs, src)
    assert rs.null_value == src.null_value
    src_mask = src.mask.compute()
    assert src_mask.any()
    np.testing.assert_array_equal(rs.mask.compute(), src_mask)


def test_dataarray_to_raster_mask_and_crs():
    xdata = _latlon_dataarray(nodata=5.0)
    xmask = xr.zeros_like(xdata, dtype=bool)
    xmask[2, 3] = True
    rs = rts.dataarray_to_raster(xdata, xmask=xmask, crs=3310)

    assert rs.crs == "EPSG:3310"
    assert rs.null_value == 5.0
    expected_mask = np.zeros((1, 3, 4), dtype=bool)
    expected_mask[0, 0, 3] = True
    np.testing.assert_array_equal(rs.mask.compute(), expected_mask)
