import unittest

from raster_tools import Raster, creation, general


# TODO: fully test module
class TestSurface(unittest.TestCase):
    def setUp(self):
        self.dem = Raster("./test/data/elevation.tif")
        self.multi = Raster("./test/data/multiband_small.tif")

    def test_regions(self):
        rs_pos = creation.random_raster(
            self.dem, distribution="poisson", bands=1, params=[7, 0.5]
        )
        general.regions(rs_pos).eval()

    def test_local_stats(self):
        general.local_stats(self.multi, "mean").eval()
        general.local_stats(self.multi, "median").eval()
        general.local_stats(self.multi, "mode").eval()
        general.local_stats(self.multi, "std").eval()
        general.local_stats(self.multi, "var").eval()
        general.local_stats(self.multi, "sum").eval()
        general.local_stats(self.multi, "maxband").eval()
        general.local_stats(self.multi, "minband").eval()
        general.local_stats(self.multi, "min").eval()
        general.local_stats(self.multi, "max").eval()
        general.local_stats(self.multi, "entropy").eval()
        general.local_stats(self.multi, "asm").eval()
        general.local_stats(self.multi, "prod").eval()

    def test_aggregate(self):
        general.aggregate(self.dem, (3, 3), "mean").eval()
        general.aggregate(self.dem, (3, 3), "median").eval()
        general.aggregate(self.dem, (3, 3), "mode").eval()
        general.aggregate(self.dem, (3, 3), "std").eval()
        general.aggregate(self.dem, (3, 3), "var").eval()
        general.aggregate(self.dem, (3, 3), "max").eval()
        general.aggregate(self.dem, (3, 3), "min").eval()
        general.aggregate(self.dem, (3, 3), "prod").eval()
        general.aggregate(self.dem, (3, 3), "sum").eval()
        general.aggregate(self.dem, (3, 3), "entropy").eval()
        general.aggregate(self.dem, (3, 3), "asm").eval()
        general.aggregate(self.dem, (3, 3), "unique").eval()
