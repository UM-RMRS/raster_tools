import unittest
from functools import partial

import numpy as np
import pytest

from raster_tools import creation, general
from raster_tools.dtypes import F32, F64, I32, I64
from raster_tools.raster import Raster
from raster_tools.stat_common import (
    nan_unique_count_jit,
    nanargmax_jit,
    nanargmin_jit,
    nanasm_jit,
    nanentropy_jit,
    nanmode_jit,
)

stat_funcs = {
    "max": partial(np.nanmax, axis=0),
    "mean": partial(np.nanmean, axis=0),
    "median": partial(np.nanmedian, axis=0),
    "min": partial(np.nanmin, axis=0),
    "prod": partial(np.nanprod, axis=0),
    "std": partial(np.nanstd, axis=0),
    "sum": partial(np.nansum, axis=0),
    "var": partial(np.nanvar, axis=0),
}
custom_stat_funcs = {
    "asm": nanasm_jit,
    "entropy": nanentropy_jit,
    "maxband": nanargmax_jit,
    "minband": nanargmin_jit,
    "mode": nanmode_jit,
    "unique": nan_unique_count_jit,
}


def get_stat_dtype(stat, input_data):
    if stat == "unique":
        return np.min_scalar_type(input_data.shape[0])
    if stat in ("mode", "min", "max"):
        return input_data.dtype
    if stat in ("minband", "maxband"):
        return np.min_scalar_type(input_data.shape[0] - 1)
    if input_data.dtype == F32:
        return F32
    return F64


@pytest.mark.parametrize("chunk", [False, True])
@pytest.mark.parametrize("stat", list(custom_stat_funcs.keys()))
def test_local_stats(stat, chunk):
    for dt in (I32, I64, F32, F64):
        x = np.arange(5 * 4 * 4).reshape(5, 4, 4) - 20
        x[2, :, :-1] = 1
        x[:, 2, 2] = 1
        rs = Raster(x.astype(dt)).set_null_value(1)
        if chunk:
            rs._rs.data = rs._data.rechunk((1, 2, 2))
            rs._mask = rs._mask.rechunk((1, 2, 2))
            orig_chunks = rs._data.chunks
        xx = np.where(rs._mask.compute(), np.nan, rs._data.compute())

        if stat in stat_funcs:
            sfunc = stat_funcs[stat]
            truth = sfunc(xx)[None]
            if stat != "sum":
                truth = np.where(np.isnan(truth), 1, truth)
            else:
                truth = np.where(
                    np.all(rs._mask, axis=0).compute(), rs.null_value, truth
                )
            result = general.local_stats(rs, stat)
        else:
            sfunc = custom_stat_funcs[stat]
            truth = np.zeros(
                (1, *rs.shape[1:]), dtype=get_stat_dtype(stat, rs._data)
            )
            for i in range(rs.shape[1]):
                for j in range(rs.shape[2]):
                    v = sfunc(xx[:, i, j])
                    if np.isnan(v):
                        v = rs.null_value
                    truth[0, i, j] = v
            truth = np.where(
                np.all(rs._mask, axis=0).compute(), rs.null_value, truth
            )
            result = general.local_stats(rs, stat)
        assert result.shape[0] == 1
        assert result.shape == truth.shape
        assert np.allclose(result, truth, equal_nan=True)
        assert result._data.chunks == result._mask.chunks
        if chunk:
            assert result._data.chunks == ((1,), *orig_chunks[1:])
        else:
            assert result._data.chunks == ((1,), *rs._data.chunks[1:])
        assert result.dtype == get_stat_dtype(stat, rs._data)


def test_local_stats_reject_bad_stat():
    rs = Raster(np.arange(5 * 4 * 4).reshape(5, 4, 4))

    for stat in [0, np.nanvar, float]:
        with pytest.raises(TypeError):
            general.local_stats(rs, stat)
    for stat in ["", "nanstd", "minn"]:
        with pytest.raises(ValueError):
            general.local_stats(rs, stat)


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
        self.assertTrue(np.array(result).all())

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
