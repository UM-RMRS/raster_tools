import pytest

from tests import testdata


@pytest.fixture
def dem():
    return testdata.raster.dem


@pytest.fixture
def xdem():
    return testdata.raster.dem.xdata


@pytest.fixture
def dem_small():
    return testdata.raster.dem_small


@pytest.fixture
def xdem_small():
    return testdata.raster.dem_small.xdata


@pytest.fixture
def dem_clipped():
    return testdata.raster.dem_small


@pytest.fixture
def xdem_clipped():
    return testdata.raster.dem_small.xdata


@pytest.fixture
def dem_clipped_small():
    return testdata.raster.dem_small


@pytest.fixture
def xdem_clipped_small():
    return testdata.raster.dem_small.xdata


@pytest.fixture
def pods():
    return testdata.vector.pods


@pytest.fixture
def pods_small():
    return testdata.vector.pods_small


@pytest.fixture
def lmus():
    return testdata.vector.lmus


@pytest.fixture
def test_circles_small():
    return testdata.vector.test_circles_small
