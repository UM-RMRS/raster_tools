import unittest
from functools import partial

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    grey_dilation,
    grey_erosion,
)

from raster_tools import creation, general
from raster_tools.dtypes import F32, F64, I16, I32, I64, U8, U16, is_scalar
from raster_tools.masking import get_default_null_value
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
            rs = rs._rechunk((1, 2, 2))
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
        rs = rs._rechunk((1, 3, 3))
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


@pytest.mark.parametrize("chunk", [False, True])
@pytest.mark.parametrize("null_value", [-99, 100, np.nan])
@pytest.mark.parametrize(
    "size", [2, 3, 4, 5, (3, 1), (1, 4), (2, 3), (3, 3), (3, 5), (4, 6)]
)
@pytest.mark.parametrize("name", ["erode", "dilate"])
def test_erode_dilate(name, size, null_value, chunk):
    erode_or_dilate = getattr(general, name)
    grey_erode_or_dilate = grey_erosion if name == "erode" else grey_dilation
    binary_erode_or_dilate = (
        binary_erosion if name == "erode" else binary_dilation
    )

    nan_null = np.isnan(null_value)
    x = np.full((15, 15), null_value)
    x[4:10, 4:11] = 2
    x[5:8, 6:10] = 3
    x[:, 3] = 1
    x[5] = 4
    x[12, 12] = 3
    x = np.stack((x, x))
    if nan_null:
        mask = np.isnan(x)
    else:
        mask = x == null_value
    x[1] += 1
    if not nan_null:
        x[mask] = null_value
    rs = Raster(x).set_null_value(null_value)
    if chunk:
        rs = rs._rechunk((1, 4, 4))
    rs._rs = rs._rs.rio.write_crs("EPSG:3857")
    tup_size = size if isinstance(size, tuple) else (size, size)

    ntruth = x.copy()
    mask_truth = mask.copy()
    fill = 0
    ntruth[mask] = fill
    for bnd in range(len(ntruth)):
        ntruth[bnd] = grey_erode_or_dilate(
            ntruth[bnd], size=size, mode="constant", cval=fill
        )
        mask_truth[bnd] = ~binary_erode_or_dilate(
            ~mask_truth[bnd], structure=np.ones(tup_size, dtype=bool)
        )
    ntruth[mask_truth] = null_value
    xtruth = xr.zeros_like(rs.xrs)
    xtruth.data = ntruth
    truth = rs._replace(xtruth, mask=da.from_array(mask_truth))

    result = erode_or_dilate(rs, size)

    assert result.dtype == rs.dtype
    assert xtruth.equals(result.xrs)
    assert np.allclose(result, truth, equal_nan=True)
    assert np.allclose(result._mask.compute(), mask_truth)
    assert rs.crs == result.crs
    if not nan_null:
        assert rs.null_value == result.null_value
    else:
        assert np.isnan(result.null_value)


@pytest.mark.parametrize("name", ["erode", "dilate"])
def test_erode_dilate_errors(name):
    func = getattr(general, name)
    rs = Raster("tests/data/elevation_small.tif")

    for size in [3.0, None]:
        with pytest.raises(TypeError):
            func(rs, size)
    for size in [0, 1, -1, (3,), (1, 1), (3, 3, 3), (-3, 3), ()]:
        with pytest.raises(ValueError):
            func(rs, size)


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

    @pytest.mark.filterwarnings("ignore:The null value")
    def test_band_concat_bool_rasters(self):
        rs1 = Raster("tests/data/elevation_small.tif") > -100
        rs2 = rs1.copy()
        result = general.band_concat((rs1, rs2))
        self.assertTrue(
            result.null_value == get_default_null_value(result.dtype)
        )
        self.assertTrue(result.dtype == np.dtype(bool))
        self.assertTrue(np.array(result).all())

        result = general.band_concat((rs1, rs2), -1)
        # The null value will be changed to match type
        self.assertTrue(result.null_value == get_default_null_value(bool))
        self.assertTrue(result.dtype == np.dtype("?"))
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
    assert np.allclose(rs.remap_range(mapping), truth)

    mappings = [mapping, (5, 15, -1)]
    result = general.remap_range(rs, mappings)
    truth[(rsnp >= mappings[1][0]) & (rsnp < mappings[1][1])] = mappings[1][2]
    assert np.allclose(result, truth)
    assert np.allclose(rs.remap_range(mappings), truth)

    # Test multiple with potential conflict in last 2
    mappings = [(0, 1, 0), (1, 2, 1), (2, 3, 8), (8, 9, 2)]
    result = general.remap_range(rs, mappings)
    truth = rsnp.copy()
    for m in mappings:
        truth[(rsnp >= m[0]) & (rsnp < m[1])] = m[2]
    assert np.allclose(result.xrs.values, truth)
    assert np.allclose(rs.remap_range(mappings), truth)

    # Test precedence
    mappings = [(0, 2, 0), (1, 2, 1)]
    result = general.remap_range(rs, mappings)
    truth = rsnp.copy()
    m = mappings[0]
    truth[(rsnp >= m[0]) & (rsnp < m[1])] = m[2]
    assert np.allclose(result.xrs.values, truth)
    assert np.allclose(rs.remap_range(mappings), truth)


@pytest.mark.parametrize(
    "rast,mapping",
    [
        (Raster(np.arange(16).reshape((4, 4))), [(0, 4, -1), (8, 12, 0)]),
        (
            Raster(np.arange(32).reshape((2, 4, 4))),
            [(0, 4, -1), (8, 12, 0), (15, 20, 0)],
        ),
    ],
)
@pytest.mark.parametrize("inc", ["left", "right", "both", "none"])
def test_remap_range_inclusivity(rast, mapping, inc):
    data = rast._values.copy()
    hist = np.zeros(data.shape, dtype=bool)
    for (left, right, new) in mapping:
        if inc == "left":
            mask = (left <= data) & (data < right)
        elif inc == "right":
            mask = (left < data) & (data <= right)
        elif inc == "both":
            mask = (left <= data) & (data <= right)
        else:
            mask = (left < data) & (data < right)
        mask[hist] = 0
        if mask.any():
            data[mask] = new
        hist |= mask

    result = general.remap_range(rast, mapping, inc)

    assert np.allclose(result, data)
    assert np.allclose(rast.remap_range(mapping, inc), data)


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


def get_where_truth(cond, x, y):
    if cond.dtype != np.dtype(bool):
        cond = cond > 0

    chunks = "auto"
    if not is_scalar(x):
        chunks = x._data.chunks
    elif not is_scalar(y):
        chunks = y._data.chunks
    cond = np.array(cond)
    nx = (
        np.ma.array(x._data.compute(), mask=x._mask.compute())
        if not is_scalar(x)
        else x
    )
    ny = (
        np.ma.array(y._data.compute(), mask=y._mask.compute())
        if not is_scalar(y)
        else y
    )
    masked = any(r._masked if isinstance(r, Raster) else False for r in (x, y))
    scalar_and_nan = (
        all(is_scalar(r) for r in (x, y)) and np.isnan((x, y)).any()
    )
    masked |= scalar_and_nan

    xmask = x._mask.compute() if not is_scalar(x) else np.zeros_like(cond)
    ymask = y._mask.compute() if not is_scalar(y) else np.zeros_like(cond)

    result = np.ma.where(cond, nx, ny)
    if scalar_and_nan:
        mask = np.isnan(result)
    else:
        mask = np.where(cond, xmask, ymask)
    result = Raster(np.array(result))
    if chunks != "auto":
        result = result._rechunk(chunks)
    if masked:
        result._mask = da.from_array(mask, chunks=chunks)
        result._null_value = get_default_null_value(result.dtype)
        result = result.burn_mask()
    return result


wshape = (10, 10)
nwhere = np.prod(wshape)


def create_rs(x):
    x = Raster(x)
    x._rs = x._rs.rio.write_crs("EPSG:3857")
    return x


@pytest.mark.parametrize(
    "x,y",
    [
        (0, 1),
        (0, np.nan),
        (0, create_rs(np.ones(wshape))),
        (0, create_rs(np.ones(wshape)).set_null_value(1)),
        (create_rs(np.ones(wshape)), 0),
        (create_rs(np.zeros(wshape)), create_rs(np.ones(wshape))),
        (
            create_rs(np.zeros(wshape)).set_null_value(0),
            create_rs(np.ones(wshape)).set_null_value(1),
        ),
        (
            create_rs(np.arange(nwhere).reshape(wshape)).set_null_value(10),
            create_rs(np.arange(nwhere).reshape(wshape) + 20).set_null_value(
                45
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "cond",
    [
        create_rs(np.zeros(wshape, dtype=bool)),
        create_rs(np.ones(wshape, dtype=bool)),
        create_rs(np.arange(nwhere).reshape(wshape) > 40),
        create_rs(np.arange(nwhere).reshape(wshape) % 10),
        create_rs((np.arange(nwhere).reshape(wshape) % 10) > 0),
        create_rs((np.arange(nwhere).reshape(wshape) % 4) > 0),
    ],
)
def test_where(cond, x, y):
    truth = get_where_truth(cond, x, y)
    result = general.where(cond, x, y)

    assert np.allclose(result, truth, equal_nan=True)
    assert result.dtype == result.dtype
    if result._masked:
        assert not np.isnan(result.null_value)
    assert result.null_value == truth.null_value
    assert np.allclose(result._mask.compute(), truth._mask.compute())
    assert result._data.chunks == truth._data.chunks
    assert result.crs is not None
    assert result.crs == "EPSG:3857"


def test_reclassify():
    data = np.arange(100).reshape((10, 10))
    rast = Raster(Raster(data).xrs.rio.write_crs("EPSG:3857"))
    mapping = {k: v for k, v in zip(np.arange(40), np.arange(20, 60))}
    included_mask = np.array([v in mapping for v in data.ravel()]).reshape(
        (10, 10)
    )

    truth = data.copy()
    truth[included_mask] = np.arange(20, 60)
    result = general.reclassify(rast, mapping)
    assert np.allclose(result, truth)
    assert result.crs == rast.crs

    truth = data.copy()
    truth[included_mask] = np.arange(20, 60)
    truth[~included_mask] = get_default_null_value(int)
    result = general.reclassify(rast, mapping, True)
    assert np.allclose(general.reclassify(rast, mapping, True), truth)
    assert result.null_value == get_default_null_value(int)
    assert result.crs == rast.crs

    rast = rast.set_null_value(-1)
    truth = data.copy()
    truth[included_mask] = np.arange(20, 60)
    truth[~included_mask] = -1
    result = general.reclassify(rast, mapping, True)
    assert np.allclose(result, truth)
    assert result.null_value == -1
    assert result.crs == rast.crs
