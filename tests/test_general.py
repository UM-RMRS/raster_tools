import unittest

import numpy as np
import pytest

from raster_tools import creation, general
from raster_tools.raster import Raster


# TODO: fully test module
class TestSurface(unittest.TestCase):
    def setUp(self):
        self.dem = Raster("tests/data/elevation.tif")
        self.multi = Raster("tests/data/multiband_small.tif")

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
        rs1 = Raster("tests/data/elevation_small.tif")
        rs2 = Raster("tests/data/elevation2_small.tif")
        rsnp1 = rs1.xrs.values
        rsnp2 = rs2.xrs.values
        truth = np.concatenate((rsnp1, rsnp2))
        test = general.band_concat([rs1, rs2])
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(np.allclose(test, truth))
        truth = np.concatenate((rsnp1, rsnp1, rsnp2, truth))
        test = general.band_concat([rs1, rs1, rs2, test])
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(np.allclose(test, truth))

    def test_band_concat_band_dim_values(self):
        rs1 = Raster("tests/data/elevation_small.tif")
        rs2 = Raster("tests/data/elevation2_small.tif")
        test = general.band_concat([rs1, rs2])
        # Make sure that band is now an increaseing list starting at 1 and
        # incrementing by 1
        self.assertTrue(all(test.xrs.band == [1, 2]))
        test = general.band_concat([rs1, test, rs2])
        self.assertTrue(all(test.xrs.band == [1, 2, 3, 4]))

    def test_band_concat_path_inputs(self):
        rs1 = Raster("tests/data/elevation_small.tif")
        rs2 = Raster("tests/data/elevation2_small.tif")
        rsnp1 = rs1.xrs.values
        rsnp2 = rs2.xrs.values
        truth = np.concatenate((rsnp1, rsnp2, rsnp1, rsnp2))
        test = general.band_concat(
            [
                rs1,
                rs2,
                "tests/data/elevation_small.tif",
                "tests/data/elevation2_small.tif",
            ]
        )
        self.assertEqual(test.shape, truth.shape)
        self.assertTrue(np.allclose(test, truth))

    def test_band_concat_bool_rasters(self):
        rs1 = Raster("tests/data/elevation_small.tif") > -100
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
        rs1 = Raster("tests/data/elevation_small.tif")
        rs2 = Raster("tests/data/elevation2_small.tif")
        rs3 = Raster("tests/data/elevation.tif")
        with self.assertRaises(ValueError):
            general.band_concat([])
        with self.assertRaises(ValueError):
            general.band_concat([rs1, rs2, rs3])
        with self.assertRaises(ValueError):
            general.band_concat([rs3, rs2])


def test_remap_range():
    rs = Raster(np.arange(25).reshape((5, 5)))
    rsnp = rs.xrs.values

    mapping = (0, 5, 1)
    result = general.remap_range(rs, mapping)
    truth = rsnp.copy()
    truth[(rsnp >= mapping[0]) & (rsnp < mapping[1])] = mapping[2]
    assert np.allclose(result, truth)

    mappings = [mapping, (5, 15, -1)]
    result = general.remap_range(rs, mappings)
    truth[(rsnp >= mappings[1][0]) & (rsnp < mappings[1][1])] = mappings[1][2]
    assert np.allclose(result, truth)

    # Test multiple with potential conflict in last 2
    mappings = [(0, 1, 0), (1, 2, 1), (2, 3, 8), (8, 9, 2)]
    result = general.remap_range(rs, mappings)
    truth = rsnp.copy()
    for m in mappings:
        truth[(rsnp >= m[0]) & (rsnp < m[1])] = m[2]
    assert np.allclose(result.xrs.values, truth)

    # Test precedence
    mappings = [(0, 2, 0), (1, 2, 1)]
    result = general.remap_range(rs, mappings)
    truth = rsnp.copy()
    m = mappings[0]
    truth[(rsnp >= m[0]) & (rsnp < m[1])] = m[2]
    assert np.allclose(result.xrs.values, truth)


def test_remap_range_f16():
    rs = Raster(np.arange(25).reshape((5, 5))).astype("float16")
    rsnp = rs._values
    mapping = (0, 5, 1)
    result = general.remap_range(rs, mapping)
    truth = rsnp.copy()
    truth[(rsnp >= mapping[0]) & (rsnp < mapping[1])] = mapping[2]
    assert rs.dtype == np.dtype("float16")
    assert result.dtype == np.dtype("float16")
    assert np.allclose(result, truth)

    rs = Raster(np.arange(25).reshape((5, 5))).astype("int8")
    rsnp = rs._values
    mapping = (0, 5, 2.0)
    result = general.remap_range(rs, mapping)
    truth = rsnp.copy().astype("float16")
    truth[(rsnp >= mapping[0]) & (rsnp < mapping[1])] = mapping[2]
    assert rs.dtype == np.dtype("int8")
    assert result.dtype == np.dtype("float16")
    assert np.allclose(result, truth)


def test_remap_range_errors():
    rs = Raster("tests/data/elevation_small.tif")
    # TypeError if not scalars
    with pytest.raises(TypeError):
        general.remap_range(rs, (None, 2, 4))
    with pytest.raises(TypeError):
        general.remap_range(rs, (0, "2", 4))
    with pytest.raises(TypeError):
        general.remap_range(rs, (0, 2, None))
    with pytest.raises(TypeError):
        general.remap_range(rs, [(0, 2, 1), (2, 3, None)])
    # ValueError if nan
    with pytest.raises(ValueError):
        general.remap_range(rs, (np.nan, 2, 4))
    with pytest.raises(ValueError):
        general.remap_range(rs, (0, np.nan, 4))
    # ValueError if range reversed
    with pytest.raises(ValueError):
        general.remap_range(rs, (0, -1, 6))
    with pytest.raises(ValueError):
        general.remap_range(rs, (1, 1, 6))
    with pytest.raises(ValueError):
        general.remap_range(rs, (0, 1))
    with pytest.raises(ValueError):
        general.remap_range(rs, [(0, 1, 2), (0, 3)])
    with pytest.raises(ValueError):
        general.remap_range(rs, ())
