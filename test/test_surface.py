import unittest

import numpy as np

from raster_tools import Raster, surface


class TestSurface(unittest.TestCase):
    def setUp(self):
        self.dem = Raster("test/data/elevation.tif")

    # TODO: add test for surface_area_3d

    def test_slope(self):
        slope = surface.slope(self.dem)
        truth = Raster("test/data/raster/slope.tif")

        # Test default degrees
        self.assertTrue(slope._masked)
        self.assertTrue(slope.null_value == self.dem.null_value)
        self.assertTrue(np.allclose(slope, truth))
        self.assertTrue(slope.dtype == np.dtype("float64"))
        # Test degrees=True
        slope = surface.slope(self.dem, degrees=True)
        self.assertTrue(slope._masked)
        self.assertTrue(slope.null_value == self.dem.null_value)
        self.assertTrue(np.allclose(slope, truth))
        self.assertTrue(slope.dtype == np.dtype("float64"))
        # Test degrees=False
        truth = Raster("test/data/raster/slope_percent.tif")
        slope = surface.slope(self.dem, degrees=False)
        self.assertTrue(slope._masked)
        self.assertTrue(slope.null_value == self.dem.null_value)
        self.assertTrue(np.allclose(slope, truth))
        self.assertTrue(slope.dtype == np.dtype("float64"))

    def test_aspect(self):
        aspect = surface.aspect(self.dem)
        truth = Raster("test/data/raster/aspect.tif")

        self.assertTrue(aspect._masked)
        self.assertTrue(aspect.null_value == self.dem.null_value)
        self.assertTrue(np.allclose(aspect, truth, 4e-5))
        self.assertTrue(aspect.dtype == np.dtype("float64"))

    def test_curvature(self):
        curv = surface.curvature(self.dem)
        truth = Raster("test/data/raster/curv.tif")

        self.assertTrue(curv._masked)
        self.assertTrue(curv.null_value == self.dem.null_value)
        # ESRI treats the edges as valid even though they are not.
        # surface.curvature does not so we ignore the edges in the comparison.
        self.assertTrue(
            np.allclose(
                curv._rs.data[..., 1:-1, 1:-1], truth._rs.data[..., 1:-1, 1:-1]
            )
        )
        self.assertTrue(curv.dtype == np.dtype("float64"))

    def test_northing(self):
        northing = surface.northing(self.dem)
        truth = Raster("test/data/raster/northing.tif")

        self.assertTrue(self.dem._masked)
        self.assertTrue(northing._masked)
        self.assertTrue(northing.null_value == self.dem.null_value)
        self.assertTrue(np.allclose(northing, truth, 1e-6, 1e-6))
        self.assertTrue(northing.dtype.kind == "f")

        northing = surface.northing(surface.aspect(self.dem), is_aspect=True)
        self.assertTrue(northing._masked)
        self.assertTrue(northing.null_value == self.dem.null_value)
        self.assertTrue(np.allclose(northing, truth, 1e-6, 1e-6))
        self.assertTrue(northing.dtype.kind == "f")

    def test_easting(self):
        easting = surface.easting(self.dem)
        truth = Raster("test/data/raster/easting.tif")

        self.assertTrue(easting._masked)
        self.assertTrue(easting.null_value == self.dem.null_value)
        self.assertTrue(np.allclose(easting, truth, 1e-6, 1e-6))
        self.assertTrue(easting.dtype.kind == "f")

        easting = surface.easting(surface.aspect(self.dem), is_aspect=True)
        self.assertTrue(easting._masked)
        self.assertTrue(easting.null_value == self.dem.null_value)
        self.assertTrue(np.allclose(easting, truth, 1e-6, 1e-6))
        self.assertTrue(easting.dtype.kind == "f")

    def test_hillshade(self):
        hill = surface.hillshade(self.dem)
        truth = Raster("test/data/raster/hillshade.tif")

        self.assertTrue(hill._masked)
        self.assertTrue(hill.null_value == 255)
        self.assertTrue(np.allclose(hill, truth))
        self.assertTrue(hill.dtype == np.dtype("uint8"))
