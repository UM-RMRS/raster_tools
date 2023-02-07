import numpy as np
import pytest

from raster_tools import focal, surface
from raster_tools.dtypes import I32
from raster_tools.general import band_concat
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster
from tests.utils import assert_rasters_similar, assert_valid_raster


@pytest.fixture
def dem():
    return Raster("tests/data/raster/elevation.tif")


# TODO: add test for surface_area_3d


@pytest.mark.parametrize(
    "degrees,truth",
    [
        (None, Raster("tests/data/raster/slope.tif")),
        (True, Raster("tests/data/raster/slope.tif")),
        (False, Raster("tests/data/raster/slope_percent.tif")),
    ],
)
def test_slope(dem, degrees, truth):
    if degrees is None:
        slope = surface.slope(dem)
    else:
        slope = surface.slope(dem, degrees=degrees)
    assert_valid_raster(slope)
    assert_rasters_similar(slope, dem)
    assert_rasters_similar(slope, truth)
    assert slope._masked
    assert slope.null_value == get_default_null_value(slope.dtype)
    assert np.allclose(slope, truth.set_null_value(slope.null_value))
    assert slope.dtype == np.dtype("float64")


def test_aspect(dem):
    aspect = surface.aspect(dem)
    truth = Raster("tests/data/raster/aspect.tif")

    assert_valid_raster(aspect)
    assert_rasters_similar(aspect, dem)
    assert_rasters_similar(aspect, truth)
    assert aspect._masked
    assert aspect.null_value == get_default_null_value(aspect.dtype)
    assert np.allclose(aspect, truth.set_null_value(aspect.null_value), 4e-5)
    assert aspect.dtype == np.dtype("float64")


def test_curvature(dem):
    curv = surface.curvature(dem)
    truth = Raster("tests/data/raster/curv.tif")

    assert_valid_raster(curv)
    assert_rasters_similar(curv, dem)
    assert_rasters_similar(curv, truth)
    assert curv._masked
    assert curv.null_value == get_default_null_value(curv.dtype)
    # ESRI treats the edges as valid even though they are not.
    # surface.curvature does not so we ignore the edges in the comparison.
    mask = truth._ds.mask
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    truth._ds["mask"] = mask
    assert np.allclose(curv, truth.set_null_value(curv.null_value))
    assert curv.dtype == np.dtype("float64")


def test_northing(dem):
    northing = surface.northing(dem)
    truth = Raster("tests/data/raster/northing.tif")

    assert_valid_raster(northing)
    assert_rasters_similar(northing, dem)
    assert_rasters_similar(northing, truth)
    assert dem._masked
    assert northing._masked
    assert northing.null_value == get_default_null_value(northing.dtype)
    assert np.allclose(
        northing, truth.set_null_value(northing.null_value), 1e-6, 1e-6
    )
    assert northing.dtype.kind == "f"

    northing = surface.northing(surface.aspect(dem), is_aspect=True)
    assert_valid_raster(northing)
    assert_rasters_similar(northing, dem)
    assert_rasters_similar(northing, truth)
    assert northing._masked
    assert northing.null_value == get_default_null_value(northing.dtype)
    assert np.allclose(
        northing, truth.set_null_value(northing.null_value), 1e-6, 1e-6
    )
    assert northing.dtype.kind == "f"


def test_easting(dem):
    easting = surface.easting(dem)
    truth = Raster("tests/data/raster/easting.tif")

    assert_valid_raster(easting)
    assert_rasters_similar(easting, dem)
    assert_rasters_similar(easting, truth)
    assert dem._masked
    assert easting._masked
    assert easting.null_value == get_default_null_value(easting.dtype)
    assert np.allclose(
        easting, truth.set_null_value(easting.null_value), 1e-6, 1e-6
    )
    assert easting.dtype.kind == "f"

    easting = surface.easting(surface.aspect(dem), is_aspect=True)
    assert_valid_raster(easting)
    assert_rasters_similar(easting, dem)
    assert_rasters_similar(easting, truth)
    assert easting._masked
    assert easting.null_value == get_default_null_value(easting.dtype)
    assert np.allclose(
        easting, truth.set_null_value(easting.null_value), 1e-6, 1e-6
    )
    assert easting.dtype.kind == "f"


def test_hillshade(dem):
    hill = surface.hillshade(dem)
    truth = Raster("tests/data/raster/hillshade.tif")

    assert_valid_raster(hill)
    assert_rasters_similar(hill, dem)
    assert_rasters_similar(hill, truth)
    assert hill._masked
    assert hill.null_value == 255
    assert np.allclose(hill, truth)
    assert hill.dtype == np.dtype("uint8")


def test_tpi():
    dem = Raster("tests/data/raster/elevation_small.tif")
    truth = ((dem - focal.focal(dem, "mean", (5, 11))) + 0.5).astype(
        I32, False
    )
    tpi = surface.tpi(dem, 5, 11)

    assert tpi.dtype == I32
    assert np.allclose(truth, tpi)
    assert tpi.null_value == get_default_null_value(I32)

    # Make sure that it works with multiple bands
    dem2 = dem + 100
    truth2 = ((dem2 - focal.focal(dem2, "mean", (5, 11))) + 0.5).astype(
        I32, False
    )
    truth = band_concat((truth, truth2))
    dem = band_concat((dem, dem2))

    tpi = surface.tpi(dem, 5, 11)

    assert np.allclose(truth, tpi)
