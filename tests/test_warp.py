import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from raster_tools import warp
from raster_tools.masking import get_default_null_value
from tests import testdata
from tests.utils import assert_valid_raster


def _repoject(raster, crs, method):
    nv = (
        raster.null_value
        if raster._masked
        else get_default_null_value(raster.dtype)
    )
    xreprojected = raster.xdata.odc.reproject(
        crs, resampling=method, dst_nodata=nv
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
        dst_geobox = raster.geobox.to_crs(crs_or_geobox)
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
        target_grid = raster.geobox
    elif isinstance(crs, GeoBox):
        target_grid = crs
    else:
        target_grid = raster.geobox.to_crs(crs)
    if resolution is not None:
        target_grid = target_grid.zoom_to(resolution=resolution)

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
