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
