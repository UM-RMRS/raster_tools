import unittest
from functools import partial

import numpy as np
import pytest
import xarray as xr

from raster_tools import creation, general
from raster_tools.dtypes import (
    F32,
    F64,
    I16,
    I32,
    I64,
    U8,
    U16,
    get_default_null_value,
)
from raster_tools.raster import Raster, get_raster
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


def get_local_stats_dtype(stat, input_data):
    if stat == "unique":
        dt = np.min_scalar_type(input_data.shape[0])
        if dt == U8:
            return I16
        if dt == U16:
            return I32
    if stat in ("mode", "min", "max"):
        return input_data.dtype
    if stat in ("minband", "maxband"):
        dt = np.min_scalar_type(input_data.shape[0] - 1)
        if dt == U8:
            return I16
        if dt == U16:
            return I32
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
                (1, *rs.shape[1:]), dtype=get_local_stats_dtype(stat, rs._data)
            )
            for i in range(rs.shape[1]):
                for j in range(rs.shape[2]):
                    v = sfunc(xx[:, i, j])
                    if np.isnan(v):
                        v = rs.null_value
                    truth[0, i, j] = v
            if stat in ("unique", "minband", "maxband"):
                nv = get_default_null_value(I16)
                truth = np.where(
                    np.all(rs._mask, axis=0).compute(), nv, truth.astype(I16)
                )
            else:
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
        assert result.dtype == get_local_stats_dtype(stat, rs._data)
        assert result.crs == rs.crs
        assert result.affine == rs.affine


def test_local_stats_reject_bad_stat():
    rs = Raster(np.arange(5 * 4 * 4).reshape(5, 4, 4))

    for stat in [0, np.nanvar, float]:
        with pytest.raises(TypeError):
            general.local_stats(rs, stat)
    for stat in ["", "nanstd", "minn"]:
        with pytest.raises(ValueError):
            general.local_stats(rs, stat)


coarsen_stats = {
    "max": lambda x: x.max(),
    "mean": lambda x: x.mean(),
    "median": lambda x: x.median(),
    "min": lambda x: x.min(),
    "prod": lambda x: x.prod(),
    "std": lambda x: x.std(),
    "sum": lambda x: x.sum(),
    "var": lambda x: x.var(),
}
coarsen_custom_stats = {
    "asm": nanasm_jit,
    "entropy": nanentropy_jit,
    "mode": nanmode_jit,
    "unique": nan_unique_count_jit,
}


def get_coarsen_dtype(stat, window_size, input_data):
    if stat == "unique":
        return np.min_scalar_type(window_size)
    if stat in ("mode", "min", "max"):
        return input_data.dtype
    if input_data.dtype == F32:
        return F32
    return F64


def coarsen_block(x, axis, func, out_dtype):
    shape = tuple(d for i, d in enumerate(x.shape) if i not in axis)
    out = np.empty(shape, dtype=out_dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                v = func(x[i, j, :, k, :])
                out[i, j, k] = 0 if np.isnan(v) else v
    return out


def coarsen_reduce(x, axis, agg_func, out_dtype):
    dims = tuple(i for i in range(len(x.shape)) if i not in axis)
    chunks = tuple(x.chunks[d] for d in dims)
    return x.map_blocks(
        partial(coarsen_block, axis=axis, func=agg_func, out_dtype=out_dtype),
        chunks=chunks,
        drop_axis=axis,
        meta=np.array((), dtype=out_dtype),
    )


@pytest.mark.parametrize("chunk", [False, True])
@pytest.mark.parametrize("window_x", [1, 3])
@pytest.mark.parametrize("window_y", [2, 3, 4])
@pytest.mark.parametrize("stat", list(coarsen_custom_stats.keys()))
def test_aggregate(stat, window_y, window_x, chunk):
    window = (window_y, window_x)
    window_map = {"y": window_y, "x": window_x}
    x = np.arange(4 * 11 * 11).reshape((4, 11, 11)) - 20
    null_value = -999
    x[1, 6, :] = null_value
    x[2, 10, :] = null_value
    x[3, :, :] = null_value
    rs = Raster(x).set_null_value(null_value)
    rs._rs = rs._rs.rio.write_crs("EPSG:3857")
    if chunk:
        rs._rs.data = rs._data.rechunk((1, 3, 3))
        rs._mask = rs._mask.rechunk((1, 3, 3))
    xmask = xr.DataArray(rs._mask, coords=rs.xrs.coords, dims=rs.xrs.dims)
    xmask_truth = xmask.coarsen(dim=window_map, boundary="trim").all()

    if stat in coarsen_stats:
        stat_func = coarsen_stats[stat]
        xtruth = stat_func(
            get_raster(rs, null_to_nan=True).xrs.coarsen(
                dim=window_map, boundary="trim"
            )
        )
        xtruth = (
            xr.where(xmask_truth, null_value, xtruth)
            .astype(get_coarsen_dtype(stat, np.prod(window), rs.xrs))
            .rio.write_crs(rs.crs)
        )
        truth = Raster(xtruth).set_null_value(null_value)
    else:
        stat_func = coarsen_custom_stats[stat]
        out_dtype = get_coarsen_dtype(stat, np.prod(window), rs.xrs)
        xtruth = get_raster(rs, null_to_nan=True).xrs
        xtruth = (
            xtruth.coarsen(dim=window_map, boundary="trim")
            .reduce(
                partial(
                    coarsen_reduce,
                    agg_func=stat_func,
                    out_dtype=out_dtype,
                )
            )
            .compute()
        )
        if stat == "unique":
            nv = get_default_null_value(I16)
            xtruth = xr.where(
                xmask_truth, nv, xtruth.astype(I16)
            ).rio.write_crs(rs.crs)
            truth = Raster(xtruth).set_null_value(nv)
        else:
            xtruth = xr.where(xmask_truth, null_value, xtruth).rio.write_crs(
                rs.crs
            )
            assert xtruth.dtype == np.promote_types(
                np.min_scalar_type(null_value), out_dtype
            )
            truth = Raster(xtruth).set_null_value(null_value)
        truth._mask = xmask.data
    result = general.aggregate(rs, window, stat)

    assert xtruth.equals(result.xrs)
    assert truth.crs == result.crs
    assert truth.affine == result.affine
    assert all(
        result.resolution == np.atleast_1d(rs.resolution) * window[::-1]
    )
    assert result.null_value == nv if stat == "unique" else null_value


@pytest.mark.parametrize(
    "window,stat,error_type",
    [
        ((1, 1), "mean", ValueError),
        ((3, 0), "mean", ValueError),
        ((1, 2, 3), "mean", ValueError),
        ((1.0, 1), "mean", TypeError),
        ((3, 3), "other", ValueError),
        ((3, 3), np.sum, TypeError),
    ],
)
def test_aggregate_errors(window, stat, error_type):
    rs = Raster("tests/data/elevation.tif")
    with pytest.raises(error_type):
        general.aggregate(rs, window, stat)


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
