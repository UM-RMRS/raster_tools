# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts  # noqa: F401

# isort: on

import dask.array as da
import numpy as np
import pytest
from affine import Affine

from raster_tools.blocks import GeoBlockInfo, geo_block_infos_as_dask
from tests import testdata


def _make_block_info(
    raster,
    *,
    geobox=None,
    band_slice=None,
    row_slice=None,
    col_slice=None,
    chunk_location=(0, 0, 0),
):
    return GeoBlockInfo(
        geobox if geobox is not None else raster.geobox,
        band_slice if band_slice is not None else slice(0, 1),
        row_slice if row_slice is not None else slice(0, 100),
        col_slice if col_slice is not None else slice(0, 100),
        chunk_location,
        raster.affine,
        raster.shape,
    )


def test_geoblockinfo_creation():
    raster = testdata.raster.dem_small.chunk((1, 50, 50))
    chunk_raster = raster.get_chunk_rasters().ravel()[1]
    geobox = chunk_raster.geobox
    gbi = GeoBlockInfo(
        geobox,
        slice(0, 1),
        slice(0, 50),
        slice(50, 100),
        (0, 0, 1),
        raster.affine,
        raster.shape,
    )
    assert gbi.shape == chunk_raster.shape
    assert gbi.geobox == chunk_raster.geobox
    assert gbi.parent_affine == raster.affine
    assert gbi.parent_shape == raster.shape
    assert gbi.affine == chunk_raster.affine
    assert gbi.crs == raster.crs
    assert gbi.bbox == chunk_raster.geobox.extent.geom
    assert gbi.band_slice == slice(0, 1)
    assert gbi.row_slice == slice(0, 50)
    assert gbi.col_slice == slice(50, 100)
    assert gbi.chunk_location == (0, 0, 1)
    assert np.array_equal(gbi.band, np.arange(0, 1))
    assert np.allclose(gbi.x, chunk_raster.x)
    assert np.allclose(gbi.y, chunk_raster.y)


def test_geoblockinfo_to_dataarray():
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    data = np.arange(np.prod(gbi.shape), dtype=np.float32).reshape(gbi.shape)
    da_only = gbi.to_dataarray(data, nodata=-9999.0)
    assert da_only.dims == ("band", "y", "x")
    assert np.array_equal(da_only.values, data)
    assert np.array_equal(da_only.coords["x"].values, gbi.x)
    assert np.array_equal(da_only.coords["y"].values, gbi.y)
    assert np.array_equal(da_only.coords["band"].values, gbi.band)
    assert da_only.rio.crs == raster.crs
    assert da_only.rio.nodata == -9999.0

    mask = np.zeros(gbi.shape, dtype=bool)
    mask[..., 0, 0] = True
    da2, ma2 = gbi.to_dataarray(data, mask=mask, nodata=-9999.0)
    assert ma2.dims == ("band", "y", "x")
    assert np.array_equal(ma2.values, mask)
    assert ma2.rio.crs == raster.crs


def test_geoblockinfo_to_dataarray_shape_mismatch():
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    bad = np.zeros((1, 50, 50), dtype=np.float32)
    with pytest.raises(ValueError):
        gbi.to_dataarray(bad)
    good = np.zeros(gbi.shape, dtype=np.float32)
    bad_mask = np.zeros((1, 50, 50), dtype=bool)
    with pytest.raises(ValueError):
        gbi.to_dataarray(good, mask=bad_mask)


@pytest.mark.parametrize(
    "method,args,expected_shape,expected_affine,expected_row,expected_col",
    [
        # pad_y
        (
            "pad_y",
            (1, 0),
            (1, 101, 100),
            testdata.raster.dem_small.affine * Affine.translation(0, -1),
            slice(-1, 100),
            slice(0, 100),
        ),
        (
            "pad_y",
            (-1, 0),
            (1, 99, 100),
            testdata.raster.dem_small.affine * Affine.translation(0, 1),
            slice(1, 100),
            slice(0, 100),
        ),
        (
            "pad_y",
            (1, 1),
            (1, 102, 100),
            testdata.raster.dem_small.affine * Affine.translation(0, -1),
            slice(-1, 101),
            slice(0, 100),
        ),
        # pad_x
        (
            "pad_x",
            (1, 0),
            (1, 100, 101),
            testdata.raster.dem_small.affine * Affine.translation(-1, 0),
            slice(0, 100),
            slice(-1, 100),
        ),
        (
            "pad_x",
            (1, 2),
            (1, 100, 103),
            testdata.raster.dem_small.affine * Affine.translation(-1, 0),
            slice(0, 100),
            slice(-1, 102),
        ),
    ],
)
def test_geoblockinfo_pad_axes(
    method, args, expected_shape, expected_affine, expected_row, expected_col
):
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    new = getattr(gbi, method)(*args)
    assert new.shape == expected_shape
    assert new.affine == expected_affine
    assert new.parent_affine == raster.affine
    assert new.parent_shape == raster.shape
    assert new.crs == raster.crs
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col
    assert new.chunk_location == gbi.chunk_location


@pytest.mark.parametrize(
    "args,expected_shape,expected_affine,expected_row,expected_col",
    [
        (
            (0, 0),
            (1, 100, 100),
            testdata.raster.dem_small.affine,
            slice(0, 100),
            slice(0, 100),
        ),
        (
            (1, None),
            (1, 102, 102),
            testdata.raster.dem_small.affine * Affine.translation(-1, -1),
            slice(-1, 101),
            slice(-1, 101),
        ),
        (
            (1, 2),
            (1, 102, 104),
            testdata.raster.dem_small.affine * Affine.translation(-2, -1),
            slice(-1, 101),
            slice(-2, 102),
        ),
        (
            (-1, None),
            (1, 98, 98),
            testdata.raster.dem_small.affine * Affine.translation(1, 1),
            slice(1, 99),
            slice(1, 99),
        ),
    ],
)
def test_geoblockinfo_pad_symmetric(
    args, expected_shape, expected_affine, expected_row, expected_col
):
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    new = gbi.pad(*args)
    assert new.shape == expected_shape
    assert new.affine == expected_affine
    assert new.parent_affine == raster.affine
    assert new.crs == raster.crs
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col
    assert new.chunk_location == gbi.chunk_location


@pytest.mark.parametrize(
    "method,args,expected_affine,expected_row,expected_col",
    [
        (
            "shift_y",
            (1,),
            testdata.raster.dem_small.affine * Affine.translation(0, 1),
            slice(1, 101),
            slice(0, 100),
        ),
        (
            "shift_x",
            (2,),
            testdata.raster.dem_small.affine * Affine.translation(2, 0),
            slice(0, 100),
            slice(2, 102),
        ),
    ],
)
def test_geoblockinfo_shift_axes(
    method, args, expected_affine, expected_row, expected_col
):
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    new = getattr(gbi, method)(*args)
    # shifts preserve overall shape
    assert new.shape == (1, 100, 100)
    assert new.affine == expected_affine
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col


@pytest.mark.parametrize(
    "args,expected_affine,expected_row,expected_col",
    [
        (
            (0, 0),
            testdata.raster.dem_small.affine,
            slice(0, 100),
            slice(0, 100),
        ),
        (
            (1, None),
            testdata.raster.dem_small.affine * Affine.translation(1, 1),
            slice(1, 101),
            slice(1, 101),
        ),
        (
            (1, 2),
            testdata.raster.dem_small.affine * Affine.translation(2, 1),
            slice(1, 101),
            slice(2, 102),
        ),
    ],
)
def test_geoblockinfo_shift_symmetric(
    args, expected_affine, expected_row, expected_col
):
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    new = gbi.shift(*args)
    assert new.shape == (1, 100, 100)
    assert new.affine == expected_affine
    assert new.row_slice == expected_row
    assert new.col_slice == expected_col


def test_geoblockinfo_pad_rejects_non_int():
    raster = testdata.raster.dem_small
    gbi = _make_block_info(raster)
    with pytest.raises(TypeError):
        gbi.pad_y(1.5, 0)


def test_geo_block_infos_as_dask():
    raster = testdata.raster.dem_small.chunk((1, 50, 50))
    arr = geo_block_infos_as_dask(raster)
    assert isinstance(arr, da.Array)
    assert arr.chunksize == (1, 1, 1)
    assert arr.numblocks == (1, 2, 2)
    computed = arr.compute()
    assert isinstance(computed, np.ndarray)
    assert computed.shape == (1, 2, 2)
    for gbi in computed.ravel():
        assert isinstance(gbi, GeoBlockInfo)
        assert gbi.shape == (1, 50, 50)


def test_raster_geo_block_infos_property():
    raster = testdata.raster.dem_small.chunk((1, 50, 50))
    arr = raster.geo_block_infos
    assert isinstance(arr, da.Array)
    assert arr.numblocks == raster.data.numblocks
