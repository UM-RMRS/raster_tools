import unittest

import numpy as np

from raster_tools import creation, general
from raster_tools.raster import Raster


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


class TestBandConcat(unittest.TestCase):
    def test_band_concat(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rs2 = Raster("test/data/elevation2_small.tif")
        rsnp1 = rs1._rs.values
        rsnp2 = rs2._rs.values
        truth = np.concatenate((rsnp1, rsnp2))
        test = general.band_concat([rs1, rs2])
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(np.allclose(test, truth))
        truth = np.concatenate((rsnp1, rsnp1, rsnp2, truth))
        test = general.band_concat([rs1, rs1, rs2, test])
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(np.allclose(test, truth))

    def test_band_concat_band_dim_values(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rs2 = Raster("test/data/elevation2_small.tif")
        test = general.band_concat([rs1, rs2])
        # Make sure that band is now an increaseing list starting at 1 and
        # incrementing by 1
        self.assertTrue(all(test._rs.band == [1, 2]))
        test = general.band_concat([rs1, test, rs2])
        self.assertTrue(all(test._rs.band == [1, 2, 3, 4]))

    def test_band_concat_path_inputs(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rs2 = Raster("test/data/elevation2_small.tif")
        rsnp1 = rs1._rs.values
        rsnp2 = rs2._rs.values
        truth = np.concatenate((rsnp1, rsnp2, rsnp1, rsnp2))
        test = general.band_concat(
            [
                rs1,
                rs2,
                "test/data/elevation_small.tif",
                "test/data/elevation2_small.tif",
            ]
        )
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(np.allclose(test, truth))

    def test_band_concat_bool_rasters(self):
        rs1 = Raster("test/data/elevation_small.tif") > -100
        rs2 = rs1.copy()
        result = general.band_concat((rs1, rs2))
        self.assertTrue(rs1.null_value == result.null_value)
        self.assertTrue(result.dtype == np.dtype(bool))
        self.assertTrue(np.all(result))

        # Force bool to be converted to int to accommodate the null value
        result = general.band_concat((rs1, rs2), -1)
        self.assertTrue(-1 == result.null_value)
        self.assertTrue(result.dtype.kind == "i")
        self.assertTrue(np.all(np.array(result) == 1))

    def test_band_concat_errors(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rs2 = Raster("test/data/elevation2_small.tif")
        rs3 = Raster("test/data/elevation.tif")
        with self.assertRaises(ValueError):
            general.band_concat([])
        with self.assertRaises(ValueError):
            general.band_concat([rs1, rs2, rs3])
        with self.assertRaises(ValueError):
            general.band_concat([rs3, rs2])
