import unittest

from raster_tools import Raster, general


class TestSurface(unittest.TestCase):
    def setUp(self):
        self.dem = Raster("./test/data/elevation.tif")
        self.multi = Raster("./test/data/multiband_small.tif")

    # TODO: finish setting up tests and building model module to test predict
    def test_random(self):
        general.random_raster(self.dem, bands=2)
        general.random_raster(
            self.dem,
            bands=3,
            rtype="poisson",
        )
        general.random_raster(
            self.dem,
            bands=3,
            rtype="weibull",
        )
        general.random_raster(
            self.dem,
            bands=3,
            rtype="binomial",
        )

    def test_constant(self):
        general.constant_raster(self.dem, 3, value=50)

    def test_regions(self):
        rs_pos = general.random_raster(
            self.dem, bands=1, rtype="poisson", vls=[7, 0.5]
        )
        general.regions(rs_pos)

    def test_local_stats(self):
        general.local_stats(self.multi, "mean")
        general.local_stats(self.multi, "median")
        general.local_stats(self.multi, "mode")
        general.local_stats(self.multi, "std")
        general.local_stats(self.multi, "var")
        general.local_stats(self.multi, "sum")
        general.local_stats(self.multi, "maxband")
        general.local_stats(self.multi, "minband")
        general.local_stats(self.multi, "min")
        general.local_stats(self.multi, "max")
        general.local_stats(self.multi, "entropy")
        general.local_stats(self.multi, "asm")
        general.local_stats(self.multi, "prod")

    def test_aggregate(self):
        general.aggregate(self.dem, (3, 3), "mean")
        general.aggregate(self.dem, (3, 3), "median")
        general.aggregate(self.dem, (3, 3), "mode")
        general.aggregate(self.dem, (3, 3), "std")
        general.aggregate(self.dem, (3, 3), "var")
        general.aggregate(self.dem, (3, 3), "max")
        general.aggregate(self.dem, (3, 3), "min")
        general.aggregate(self.dem, (3, 3), "prod")
        general.aggregate(self.dem, (3, 3), "sum")
        general.aggregate(self.dem, (3, 3), "entropy")
        general.aggregate(self.dem, (3, 3), "asm")
        general.aggregate(self.dem, (3, 3), "unique")
