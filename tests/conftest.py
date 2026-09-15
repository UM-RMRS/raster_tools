import pytest

from raster_tools import rasterize
from tests import testdata


@pytest.fixture(params=["rasterio", "numba"])
def _rasterize_backend(request, monkeypatch):
    # Runs the requesting test under both rasterization backends. Modules opt
    # in with ``pytestmark = pytest.mark.usefixtures("_rasterize_backend")``
    # so the parametrization stays confined to the rasterization tests.
    # Tests marked rasterio-only or backend-agnostic (see the decorators in
    # test_rasterize.py) skip the redundant numba run.
    func = getattr(request, "function", None)
    if request.param == "numba" and (
        getattr(func, "_rasterio_only", False)
        or getattr(func, "_backend_agnostic", False)
    ):
        pytest.skip("does not exercise the numba backend")
    monkeypatch.setattr(rasterize, "RASTERIZE_BACKEND", request.param)
    return request.param


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
