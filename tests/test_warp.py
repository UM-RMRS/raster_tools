import dask.array
import numpy as np
import pytest
from affine import Affine
from odc.geo.geobox import GeoBox

import raster_tools as rts
from raster_tools import warp
from raster_tools.masking import get_default_null_value
from tests import testdata
from tests.utils import assert_valid_raster


def _footprint_grid(geobox, crs, resolution=None):
    # Smallest grid covering the footprint of `geobox` in `crs`, with its
    # origin at the footprint's top-left corner. This mirrors how reproject
    # builds its grid, so tests using it check the data, not the grid; the
    # tests with hard-coded shapes and origins pin the grid itself.
    bbox = geobox.footprint(crs, npoints=100).boundingbox
    if resolution is None:
        resolution = geobox.to_crs(crs).resolution
    return GeoBox.from_bbox(bbox, crs, resolution=resolution, tight=True)


def _repoject(raster, crs_or_geobox, method):
    nv = (
        raster.null_value
        if raster._masked
        else get_default_null_value(raster.dtype)
    )
    dst_geobox = crs_or_geobox
    if not isinstance(dst_geobox, GeoBox):
        dst_geobox = _footprint_grid(raster.geobox, crs_or_geobox)
    xreprojected = raster.xdata.odc.reproject(
        dst_geobox, resampling=method, dst_nodata=nv
    )
    xmask = xreprojected == nv
    return xreprojected.rio.write_nodata(nv), xmask


@pytest.mark.parametrize(
    "method",
    [
        "nearest",
        "bilinear",
        "cubic",
        "cubic_spline",
        "lanczos",
        "average",
        "mode",
        "max",
        "min",
        "med",
        "q1",
        "q3",
        "sum",
        "rms",
    ],
)
@pytest.mark.parametrize(
    "raster",
    [
        testdata.raster.dem_small.chunk((1, 20, 20)),
        testdata.raster.dem_small.set_null_value(None),
        testdata.raster.dem_small.chunk((1, 20, 20))
        .remap_range((0, 1100, -1))
        .set_null_value(-1),
    ],
)
def test_reproject(raster, method):
    crs = "EPSG:5070"
    truth_reprojected, truth_mask = _repoject(raster, crs, method)

    result = warp.reproject(raster, crs, method)
    assert_valid_raster(result)
    assert result.crs == crs
    assert result.null_value == truth_reprojected.rio.nodata
    assert result.data.chunksize == raster.data.chunksize
    assert np.allclose(result.xdata, truth_reprojected)
    assert np.allclose(result.xmask, truth_mask)

    result = raster.reproject(crs, method)
    assert_valid_raster(result)
    assert result.crs == crs
    assert result.null_value == truth_reprojected.rio.nodata
    assert result.data.chunksize == raster.data.chunksize
    assert np.allclose(result.xdata, truth_reprojected)
    assert np.allclose(result.xmask, truth_mask)


@pytest.mark.parametrize(
    "crs_or_geobox",
    [
        5070,
        "EPSG:5070",
        testdata.raster.dem_small.geobox.to_crs(4326),
        testdata.raster.dem_small.geobox.zoom_to(resolution=15),
    ],
)
def test_reproject_crs(crs_or_geobox):
    raster = testdata.raster.dem_small
    if isinstance(crs_or_geobox, GeoBox):
        dst_geobox = crs_or_geobox
    else:
        dst_geobox = _footprint_grid(raster.geobox, crs_or_geobox)
    truth_reprojected, truth_mask = _repoject(
        raster, crs_or_geobox, "bilinear"
    )
    crs = (
        crs_or_geobox.crs
        if isinstance(crs_or_geobox, GeoBox)
        else crs_or_geobox
    )

    result = warp.reproject(raster, crs_or_geobox, resample_method="bilinear")
    assert_valid_raster(result)
    if crs == 4326:
        # pixel -> world transforms are limited in their precision at fine
        # scales like 1m. The result of a reprojection from some CRS to
        # lat/lon will cause tiny differences in the resulting affine matrix.
        # see: https://github.com/opendatacube/odc-geo/issues/127
        assert result.geobox.crs == crs
        assert result.geobox.shape == dst_geobox.shape
        assert np.allclose(list(result.geobox.affine), list(dst_geobox.affine))
        assert np.allclose(
            list(result.xdata.rio.transform()), list(dst_geobox.affine)
        )
    else:
        assert result.geobox == dst_geobox
    assert result.null_value == truth_reprojected.rio.nodata
    assert np.allclose(result.to_numpy(), truth_reprojected.to_numpy())
    assert np.allclose(result.xmask, truth_mask)


@pytest.mark.parametrize(
    "resolution",
    [10, 30, 50, 55.5],
)
def test_reproject_resolution(resolution):
    raster = testdata.raster.dem_small
    gb = raster.geobox.zoom_to(resolution=resolution)
    truth_reprojected, truth_mask = _repoject(raster, gb, "bilinear")

    result = warp.reproject(
        raster, resolution=resolution, resample_method="bilinear"
    )
    assert_valid_raster(result)
    assert result.crs == raster.crs
    assert result.resolution == gb.resolution.xy
    assert result.null_value == truth_reprojected.rio.nodata
    assert np.allclose(result.to_numpy(), truth_reprojected.to_numpy())
    assert np.allclose(result.xmask, truth_mask)

    result = raster.reproject(
        resolution=resolution, resample_method="bilinear"
    )
    assert_valid_raster(result)
    assert result.crs == raster.crs
    assert result.resolution == gb.resolution.xy
    assert result.null_value == truth_reprojected.rio.nodata
    assert np.allclose(result.to_numpy(), truth_reprojected.to_numpy())
    assert np.allclose(result.xmask, truth_mask)


@pytest.mark.parametrize(
    "crs,resolution",
    [
        (5070, None),
        (None, 15),
        (5070, 15),
        (testdata.raster.dem_small.geobox.to_crs(5070), 15),
    ],
)
def test_reproject_crs_and_resolution(crs, resolution):
    raster = testdata.raster.dem_small

    if crs is None:
        target_grid = raster.geobox.zoom_to(resolution=resolution)
    elif isinstance(crs, GeoBox):
        target_grid = crs
        if resolution is not None:
            target_grid = target_grid.zoom_to(resolution=resolution)
    else:
        target_grid = _footprint_grid(raster.geobox, crs, resolution)

    truth_reprojected, truth_mask = _repoject(raster, target_grid, "bilinear")

    result = warp.reproject(
        raster, crs, resolution=resolution, resample_method="bilinear"
    )
    assert_valid_raster(result)
    assert result.crs == target_grid.crs
    assert result.resolution == target_grid.resolution.xy
    assert result.null_value == truth_reprojected.rio.nodata
    assert np.allclose(result.to_numpy(), truth_reprojected.to_numpy())
    assert np.allclose(result.xmask, truth_mask)

    result = raster.reproject(
        crs, resolution=resolution, resample_method="bilinear"
    )
    assert_valid_raster(result)
    assert result.crs == target_grid.crs
    assert result.resolution == target_grid.resolution.xy
    assert result.null_value == truth_reprojected.rio.nodata
    assert np.allclose(result.to_numpy(), truth_reprojected.to_numpy())
    assert np.allclose(result.xmask, truth_mask)


def test_reproject_nodata_duplication():
    raster = testdata.raster.dem_small
    assert raster.null_value == np.float32(-3.402823e38)
    assert "nodata" not in raster.xdata.attrs

    raster_rp = rts.warp.reproject(raster, 3857)
    assert raster_rp.null_value == np.float32(-3.402823e38)
    assert "nodata" not in raster_rp.xdata.attrs


def _unaligned_raster():
    # 30 m cells whose origin is not a multiple of the cell size
    return rts.data_to_raster(
        np.ones((1, 10, 10)),
        affine=Affine(30, 0, 1007, 0, -30, 2013),
        crs=5070,
    )


def test_reproject_to_crs_gives_tight_grid():
    raster = _unaligned_raster()

    result = warp.reproject(raster, "EPSG:32613")
    assert_valid_raster(result)
    # The footprint in EPSG:32613 spans about 10.5 x 10.9 cells from
    # (1425861.26, 2574214.66). Snapping the origin to a multiple of the cell
    # size, or padding the footprint, would add rows and columns of nulls.
    assert result.shape == (1, 11, 11)
    assert result.resolution == (30.0, -30.0)
    assert result.affine.c == pytest.approx(1425861.259, abs=1e-3)
    assert result.affine.f == pytest.approx(2574214.657, abs=1e-3)
    valid = ~result.mask.compute()[0]
    assert valid[0].any()
    assert valid[:, 0].any()


def test_reproject_to_crs_with_resolution_keeps_tight_origin():
    raster = _unaligned_raster()

    result = warp.reproject(raster, "EPSG:32613", resolution=20)
    assert_valid_raster(result)
    assert result.resolution == (20.0, -20.0)
    assert result.affine.c == pytest.approx(1425861.259, abs=1e-3)
    assert result.affine.f == pytest.approx(2574214.657, abs=1e-3)
    assert result.shape == (1, 17, 16)


def test_reproject_resolution_keeps_origin():
    raster = _unaligned_raster()

    result = warp.reproject(raster, resolution=20)
    assert_valid_raster(result)
    assert result.crs == raster.crs
    assert result.resolution == (20.0, -20.0)
    assert result.affine.c == 1007
    assert result.affine.f == 2013
    assert result.shape == (1, 15, 15)


def _bool_raster(mask=None, nv=None):
    data = np.array(
        [[[True, False], [True, True]], [[False, False], [True, False]]]
    )
    return rts.data_to_raster(
        data,
        mask=mask,
        x=np.array([0.5, 1.5]),
        y=np.array([1.5, 0.5]),
        crs=5070,
        nv=nv,
    )


# Covers the 2x2 test rasters and two columns past their right edge
_PAST_EDGE_DST = GeoBox((2, 4), Affine(1, 0, 0, 0, -1, 2), "EPSG:5070")


@pytest.mark.parametrize(
    "method", ["nearest", "bilinear", "average", "mode", "max"]
)
def test_reproject_unmasked_bool_masks_only_uncovered_cells(method):
    # https://github.com/UM-RMRS/raster_tools/issues/63
    raster = _bool_raster()
    assert raster.null_value is None

    result = warp.reproject(raster, _PAST_EDGE_DST, method)
    assert_valid_raster(result)
    assert result.dtype == np.dtype(bool)
    assert result.null_value == get_default_null_value(bool)
    expected_mask = np.zeros((2, 2, 4), dtype=bool)
    expected_mask[:, :, 2:] = True
    assert np.array_equal(result.mask.compute(), expected_mask)
    data = result.to_numpy()
    assert np.array_equal(data[:, :, :2], raster.to_numpy())
    assert (data[expected_mask] == result.null_value).all()


def test_reproject_masked_bool_keeps_valid_cells_holding_null_value():
    # Valid cells hold True, the null value, as well as False
    nv = True
    src_mask = np.zeros((2, 2, 2), dtype=bool)
    src_mask[0, 0, 1] = src_mask[1, 1, 0] = True
    raster = _bool_raster(mask=src_mask, nv=nv)
    assert raster.null_value == nv

    result = warp.reproject(raster, _PAST_EDGE_DST)
    assert_valid_raster(result)
    assert result.null_value == nv
    expected_mask = np.ones((2, 2, 4), dtype=bool)
    expected_mask[:, :, :2] = src_mask
    assert np.array_equal(result.mask.compute(), expected_mask)
    data = result.to_numpy()
    valid = ~src_mask
    assert np.array_equal(data[:, :, :2][valid], raster.to_numpy()[valid])
    assert (data[expected_mask] == nv).all()


def test_reproject_bool_matches_reprojected_integer_raster():
    # Across a CRS change, a bool raster must land where the same data as
    # 0/1 integers lands.
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    raster = (dem > np.median(dem.to_numpy())).set_null_value(None)
    assert raster.null_value is None
    ints = raster.astype("uint8")

    result = warp.reproject(raster, "EPSG:4326")
    expected = warp.reproject(ints, "EPSG:4326")
    assert_valid_raster(result)
    assert isinstance(result.data, dask.array.Array)
    assert result.data.chunksize == raster.data.chunksize
    assert result.geobox == expected.geobox
    mask = result.mask.compute()
    assert mask.any()
    assert not mask.all()
    assert np.array_equal(mask, expected.mask.compute())
    data = result.to_numpy()
    assert np.array_equal(data[~mask], expected.to_numpy()[~mask] == 1)
    assert data[~mask].any()
    assert not data[~mask].all()


def test_reproject_bool_sum_over_many_cells_stays_valid():
    # Summing 300 True cells into one must not land on the null value used
    # while warping
    raster = rts.data_to_raster(
        np.ones((1, 1, 300), dtype=bool),
        affine=Affine(1, 0, 0, 0, -1, 1),
        crs=5070,
    )
    dst = GeoBox((1, 1), Affine(300, 0, 0, 0, -1, 1), "EPSG:5070")

    result = warp.reproject(raster, dst, resample_method="sum")
    assert_valid_raster(result)
    assert not result.mask.compute().any()
    assert result.to_numpy().all()


@pytest.mark.parametrize(
    "method", ["nearest", "bilinear", "cubic", "average", "mode", "max"]
)
@pytest.mark.parametrize(
    "dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"]
)
def test_reproject_unmasked_keeps_valid_cells_holding_default_null(
    dtype, method
):
    # Valid cells that hold the default null value given to the output must
    # stay valid and unchanged, while only uncovered cells are masked.
    nv = get_default_null_value(dtype)
    data = np.array([[[nv, 1], [2, 3]], [[4, nv], [nv, 5]]], dtype=dtype)
    raster = rts.data_to_raster(
        data, x=np.array([0.5, 1.5]), y=np.array([1.5, 0.5]), crs=5070
    )
    assert raster.null_value is None

    result = warp.reproject(raster, _PAST_EDGE_DST, method)
    assert_valid_raster(result)
    assert result.dtype == np.dtype(dtype)
    assert result.null_value == nv
    expected_mask = np.zeros((2, 2, 4), dtype=bool)
    expected_mask[:, :, 2:] = True
    assert np.array_equal(result.mask.compute(), expected_mask)
    out = result.to_numpy()
    assert np.array_equal(out[:, :, :2], data)
    assert (out[expected_mask] == nv).all()


_ALL_METHODS = sorted(warp.SUPPORTED_RESAMPLE_METHODS)


# Mode is left out: GDAL breaks ties between equally common values
# differently for different dtypes.
@pytest.mark.parametrize("method", [m for m in _ALL_METHODS if m != "mode"])
def test_reproject_unmasked_int_matches_warp_in_own_dtype(method):
    # Unmasked ints are warped in a wider dtype. Away from the null value,
    # that must give what warping in the raster's own dtype gives, including
    # where a method overshoots the dtype's range and is clamped.
    dem = testdata.raster.dem_small
    data = np.where(dem.to_numpy() > 1167, 30_000, 0).astype("int16")
    raster = rts.data_to_raster(data, x=dem.x, y=dem.y, crs=dem.crs)
    raster = raster.chunk((1, 20, 20))
    assert raster.null_value is None
    truth, truth_mask = _repoject(raster, "EPSG:5070", method)

    result = warp.reproject(raster, "EPSG:5070", method)
    assert_valid_raster(result)
    assert result.dtype == raster.dtype
    assert result.null_value == truth.rio.nodata
    assert result.data.chunksize == raster.data.chunksize
    assert np.array_equal(result.xmask, truth_mask)
    assert np.array_equal(result.xdata, truth)
