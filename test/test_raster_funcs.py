import unittest

import numpy as np

from raster_tools import Raster, band_concat


def rs_eq_array(rs, ar):
    return (rs._rs.values == ar).all()


class TestBandConcat(unittest.TestCase):
    def test_band_concat(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rs2 = Raster("test/data/elevation2_small.tif")
        rsnp1 = rs1._rs.values
        rsnp2 = rs2._rs.values
        truth = np.concatenate((rsnp1, rsnp2))
        test = band_concat([rs1, rs2])
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(rs_eq_array(test, truth))
        truth = np.concatenate((rsnp1, rsnp1, rsnp2, truth))
        test = band_concat([rs1, rs1, rs2, test])
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(rs_eq_array(test, truth))

    def test_band_concat_band_dim_values(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rs2 = Raster("test/data/elevation2_small.tif")
        test = band_concat([rs1, rs2])
        # Make sure that band is now an increaseing list starting at 1 and
        # incrementing by 1
        self.assertTrue(all(test._rs.band == [1, 2]))
        test = band_concat([rs1, test, rs2])
        self.assertTrue(all(test._rs.band == [1, 2, 3, 4]))

    def test_band_concat_path_inputs(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rs2 = Raster("test/data/elevation2_small.tif")
        rsnp1 = rs1._rs.values
        rsnp2 = rs2._rs.values
        truth = np.concatenate((rsnp1, rsnp2, rsnp1, rsnp2))
        test = band_concat(
            [
                rs1,
                rs2,
                "test/data/elevation_small.tif",
                "test/data/elevation2_small.tif",
            ]
        )
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(rs_eq_array(test, truth))

    def test_band_concat_bool_rasters(self):
        rs1 = Raster("test/data/elevation_small.tif") > -100
        rs2 = rs1.copy()
        result = band_concat((rs1, rs2))
        self.assertTrue(rs1.null_value == result.null_value)
        self.assertTrue(result.dtype == np.dtype(bool))
        self.assertTrue(np.all(result))

        # Force bool to be converted to int to accommodate the null value
        result = band_concat((rs1, rs2), -1)
        self.assertTrue(-1 == result.null_value)
        self.assertTrue(result.dtype.kind == "i")
        self.assertTrue(np.all(np.array(result) == 1))

    def test_band_concat_errors(self):
        rs1 = Raster("test/data/elevation_small.tif")
        rs2 = Raster("test/data/elevation2_small.tif")
        rs3 = Raster("test/data/elevation.tif")
        with self.assertRaises(ValueError):
            band_concat([])
        with self.assertRaises(ValueError):
            band_concat([rs1, rs2, rs3])
        with self.assertRaises(ValueError):
            band_concat([rs3, rs2])
