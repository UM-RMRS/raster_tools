# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools  # noqa: F401

# isort: on

import io
from functools import partial

import dask.array as da
import dask_geopandas as dgpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    grey_dilation,
    grey_erosion,
)

from raster_tools import general
from raster_tools.dtypes import (
    F32,
    F64,
    I16,
    I32,
    I64,
    U8,
    U16,
    get_common_dtype,
)
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
from raster_tools.vector import Vector
from tests import testdata
from tests.utils import (
    arange_raster,
    assert_rasters_similar,
    assert_valid_raster,
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


@pytest.mark.parametrize("dtype", [I32, I64, F32, F64])
@pytest.mark.parametrize("chunk", [False, True])
@pytest.mark.parametrize("stat", list(custom_stat_funcs.keys()))
def test_local_stats(stat, chunk, dtype):
    x = np.arange(5 * 4 * 4).reshape(5, 4, 4) - 20
    x[2, :, :-1] = 1
    x[:, 2, 2] = 1
    rs = Raster(x.astype(dtype)).set_null_value(1)
    if chunk:
        rs = rs.chunk((1, 2, 2))
        orig_chunks = rs.data.chunks
    xx = np.where(rs.mask.compute(), np.nan, rs.to_numpy())

    if stat in stat_funcs:
        sfunc = stat_funcs[stat]
        truth = sfunc(xx)[None]
        if stat != "sum":
            truth = np.where(np.isnan(truth), 1, truth)
        else:
            truth = np.where(
                np.all(rs.mask, axis=0).compute(), rs.null_value, truth
            )
        result = general.local_stats(rs, stat)
    else:
        sfunc = custom_stat_funcs[stat]
        truth = np.zeros(
            (1, *rs.shape[1:]), dtype=get_local_stats_dtype(stat, rs.data)
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
                np.all(rs.mask, axis=0).compute(), nv, truth.astype(I16)
            )
        else:
            truth = np.where(
                np.all(rs.mask, axis=0).compute(),
                get_default_null_value(truth.dtype),
                truth,
            )
        result = general.local_stats(rs, stat)
    assert_valid_raster(result)
    assert result.shape[0] == 1
    assert result.shape == truth.shape
    assert np.allclose(result, truth, equal_nan=True)
    if chunk:
        assert result.data.chunks == ((1,), *orig_chunks[1:])
    else:
        assert result.data.chunks == ((1,), *rs.data.chunks[1:])
    assert result.dtype == get_local_stats_dtype(stat, rs.data)
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
    rs._ds = rs._ds.rio.write_crs("EPSG:3857")
    if chunk:
        rs = rs.chunk((1, 3, 3))
    xmask = rs.xmask
    xmask_truth = xmask.coarsen(dim=window_map, boundary="trim").all()

    if stat in coarsen_stats:
        stat_func = coarsen_stats[stat]
        xtruth = stat_func(
            get_raster(rs, null_to_nan=True).xdata.coarsen(
                dim=window_map, boundary="trim"
            )
        )
        xtruth = (
            xr.where(xmask_truth, null_value, xtruth)
            .astype(get_coarsen_dtype(stat, np.prod(window), rs.xdata))
            .rio.write_crs(rs.crs)
        )
        truth = Raster(xtruth).set_null_value(null_value)
    else:
        stat_func = coarsen_custom_stats[stat]
        out_dtype = get_coarsen_dtype(stat, np.prod(window), rs.xdata)
        xtruth = get_raster(rs, null_to_nan=True).xdata
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
            xtruth = xr.where(
                xmask_truth, get_default_null_value(xtruth.dtype), xtruth
            ).rio.write_crs(rs.crs)
            assert xtruth.dtype == np.promote_types(
                np.min_scalar_type(null_value), out_dtype
            )
            truth = Raster(xtruth).set_null_value(null_value)
        truth.mask.data = xmask.data
    result = general.aggregate(rs, window, stat)

    assert_valid_raster(result)
    assert xtruth.equals(result.xdata)
    assert rs.crs == result.crs
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
    rs = testdata.raster.dem
    with pytest.raises(error_type):
        general.aggregate(rs, window, stat)


def mock_model_function(x):
    return x @ np.array([1] * x.shape[-1])


class MockModelClass:
    def __init__(self, func):
        self.func = func

    def apply(self, x):
        return self.func(x)


class ModelAdaptor:
    def __init__(self, func):
        self.func = func

    def predict(self, x):
        return self.func(x)


@pytest.mark.parametrize(
    "raster,model,nout",
    [
        (
            arange_raster((3, 4, 4)),
            ModelAdaptor(lambda x: np.sum(x, axis=-1, keepdims=True)),
            1,
        ),
        (
            arange_raster((3, 4, 4)).set_null_value(0).set_crs("5070"),
            ModelAdaptor(lambda x: np.sum(x, axis=-1, keepdims=True)),
            1,
        ),
        (
            arange_raster((3, 4, 4), dtype="int8"),
            ModelAdaptor(np.sin),
            3,
        ),
        (
            arange_raster((2, 4, 4)).chunk((1, 2, 2)),
            ModelAdaptor(lambda x: np.concatenate([x, x], axis=1)),
            4,
        ),
        # Params to test ModelPredictAdaptor
        (
            arange_raster((3, 4, 4)).chunk((1, 2, 2)),
            general.ModelPredictAdaptor(mock_model_function),
            1,
        ),
        (
            arange_raster((3, 4, 4)),
            general.ModelPredictAdaptor(
                MockModelClass(mock_model_function), "apply"
            ),
            1,
        ),
    ],
)
def test_model_predict_raster(raster, model, nout):
    data = np.nan_to_num(raster.data.compute())
    mask = raster.mask.any(axis=0).compute()
    test_input = np.zeros((1, raster.nbands), dtype=data.dtype)
    r = model.predict(test_input)
    assert r.size == nout
    assert r.shape == (1, nout) or r.shape == (nout,)
    truth = np.empty((nout, *raster.shape[1:]), dtype=r.dtype)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            truth[:, i, j] = model.predict(data[:, i, j][None])
    if raster._masked:
        truth[:, mask] = get_default_null_value(F64)

    for result in [
        general.model_predict_raster(raster, model, nout),
        raster.model_predict(model, nout),
    ]:
        assert_valid_raster(result)
        assert_rasters_similar(
            raster, result, check_chunks=True, check_nbands=False
        )
        assert result.nbands == nout
        assert np.allclose(result, truth)
        assert result.dtype == F64
        if raster._masked:
            assert result.null_value == get_default_null_value(F64)
        assert np.allclose(result.mask, np.stack(mask, axis=0))


@pytest.mark.parametrize(
    "features,model,columns,nout,prefix",
    [
        (
            dgpd.read_file("tests/data/vector/pods.shp", chunksize=20),
            ModelAdaptor(lambda x: np.sum(x, axis=-1, keepdims=True)),
            ["Shape_Leng", "Shape_Area"],
            1,
            "pred_",
        ),
        (
            dgpd.read_file("tests/data/vector/pods.shp", chunksize=20).where(
                dgpd.read_file(
                    "tests/data/vector/pods.shp", chunksize=20
                ).Shape_Leng
                > 7e4,
                np.nan,
            ),
            ModelAdaptor(lambda x: np.sum(x, axis=-1, keepdims=True)),
            ["Shape_Leng", "Shape_Area"],
            1,
            "result",
        ),
        (
            dgpd.read_file("tests/data/vector/pods.shp", chunksize=20),
            ModelAdaptor(np.sin),
            ["Shape_Leng", "POD_Acres", "Shape_Area"],
            3,
            "pred_",
        ),
        (
            dgpd.read_file("tests/data/vector/pods.shp", chunksize=20),
            ModelAdaptor(lambda x: np.concatenate([x, x], axis=1)),
            ["Shape_Leng", "Shape_Area"],
            4,
            "others",
        ),
        # Params to test ModelPredictAdaptor
        (
            dgpd.read_file("tests/data/vector/pods.shp", chunksize=20),
            general.ModelPredictAdaptor(mock_model_function),
            ["Shape_Leng", "Shape_Area"],
            1,
            "p",
        ),
        (
            dgpd.read_file("tests/data/vector/pods.shp", chunksize=20),
            general.ModelPredictAdaptor(
                MockModelClass(mock_model_function), "apply"
            ),
            ["Shape_Leng", "Shape_Area"],
            1,
            "p",
        ),
    ],
)
def test_model_predict_vector(features, model, columns, nout, prefix):
    feats_df = features.compute()
    new_columns = [f"{prefix}{i + 1}" for i in range(nout)]
    data = feats_df[columns].to_numpy()
    valid = ~np.isnan(data).any(axis=1)
    truth = np.full((len(feats_df), nout), np.nan, dtype=F64)
    r = model.predict(data[valid])
    if r.ndim == 1:
        r = r[:, None]
    truth[valid] = r
    new_df = pd.DataFrame(dict(zip(new_columns, truth.T)))
    truth_df = pd.concat([feats_df, new_df], axis=1)

    for result in [
        general.model_predict_vector(features, model, columns, nout, prefix),
        Vector(features).model_predict(model, columns, nout, prefix),
    ]:
        assert isinstance(result, Vector)
        for c in new_columns:
            assert c in result._geo
        rdf = result._geo.compute()
        assert truth_df.equals(rdf)


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
    mask = np.isnan(x) if nan_null else x == null_value
    x[1] += 1
    if not nan_null:
        x[mask] = null_value
    rs = Raster(x).set_null_value(null_value)
    if chunk:
        rs = rs.chunk((1, 4, 4))
    rs._ds = rs._ds.rio.write_crs("EPSG:3857")
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
    xtruth = xr.zeros_like(rs.xdata)
    xtruth.data = ntruth
    truth = Raster(xtruth).set_null_value(null_value)
    truth._ds.mask.data = da.from_array(mask_truth)

    result = erode_or_dilate(rs, size)

    assert result.dtype == rs.dtype
    assert xtruth.equals(result.xdata)
    assert np.allclose(result, truth, equal_nan=True)
    assert np.allclose(result.mask.compute(), mask_truth)
    assert rs.crs == result.crs
    if not nan_null:
        assert rs.null_value == result.null_value
    else:
        assert np.isnan(result.null_value)


@pytest.mark.parametrize("name", ["erode", "dilate"])
def test_erode_dilate_errors(name):
    func = getattr(general, name)
    rs = testdata.raster.dem_small

    for size in [3.0, None]:
        with pytest.raises(TypeError):
            func(rs, size)
    for size in [0, 1, -1, (3,), (1, 1), (3, 3, 3), (-3, 3), ()]:
        with pytest.raises(ValueError):
            func(rs, size)


@pytest.mark.parametrize(
    "raster,expected,neighbors,unique_values",
    [
        (
            Raster(
                np.array(
                    [
                        [7, 7, 1, 1],
                        [9, 7, 1, 3],
                        [0, 5, 7, 3],
                        [5, 5, 4, 4],
                    ]
                )
            )
            .chunk((1, 2, 2))
            .set_null_value(9)
            .set_crs("5070"),
            Raster(
                np.array(
                    [
                        [5, 5, 1, 1],
                        [9, 5, 1, 2],
                        [0, 4, 6, 2],
                        [4, 4, 3, 3],
                    ]
                )
            )
            .chunk((1, 2, 2))
            .set_null_value(9)
            .set_crs("5070")
            .astype("uint64"),
            4,
            None,
        ),
        (
            Raster(
                np.array(
                    [
                        [7, 7, 1, 1],
                        [9, 7, 1, 3],
                        [0, 5, 7, 3],
                        [5, 5, 4, 4],
                    ]
                )
            )
            .chunk((1, 2, 2))
            .set_null_value(9)
            .set_crs("5070"),
            Raster(
                np.array(
                    [
                        [5, 5, 1, 1],
                        [9, 5, 1, 2],
                        [0, 4, 5, 2],
                        [4, 4, 3, 3],
                    ]
                )
            )
            .chunk((1, 2, 2))
            .set_null_value(9)
            .set_crs("5070")
            .astype("uint64"),
            8,
            None,
        ),
        (
            Raster(
                np.array(
                    [
                        [7, 7, 1, 1],
                        [9, 7, 1, 3],
                        [0, 5, 7, 3],
                        [5, 5, 4, 4],
                    ]
                )
            )
            .chunk((1, 2, 2))
            .set_null_value(9)
            .set_crs("5070"),
            Raster(
                np.array(
                    [
                        [1, 1, 3, 3],
                        [9, 1, 3, 4],
                        [0, 6, 2, 4],
                        [6, 6, 5, 5],
                    ]
                )
            )
            .chunk((1, 2, 2))
            .set_null_value(9)
            .set_crs("5070")
            .astype("uint64"),
            4,
            [7, 1, 3, 4, 5],
        ),
        (
            Raster(
                np.array(
                    [
                        [7, 7, 1, 1],
                        [9, 7, 1, 3],
                        [0, 5, 7, 3],
                        [5, 5, 4, 4],
                    ]
                )
            )
            .chunk((1, 2, 2))
            .set_null_value(9)
            .set_crs("5070"),
            Raster(
                np.array(
                    [
                        [1, 1, 2, 2],
                        [9, 1, 2, 3],
                        [0, 5, 1, 3],
                        [5, 5, 4, 4],
                    ]
                )
            )
            .chunk((1, 2, 2))
            .set_null_value(9)
            .set_crs("5070")
            .astype("uint64"),
            8,
            [7, 1, 3, 4, 5],
        ),
    ],
)
def test_regions(raster, expected, neighbors, unique_values):
    result = general.regions(
        raster, neighbors=neighbors, unique_values=unique_values
    )
    assert_valid_raster(result)
    assert_rasters_similar(raster, result)
    assert np.allclose(result, expected)
    assert result.dtype == expected.dtype


def test_band_concat():
    rs1 = testdata.raster.dem_clipped_small
    rs2 = rs1 + 30
    rsnp1 = rs1.to_numpy()
    rsnp2 = rs2.to_numpy()
    mask1 = rs1.xmask.to_numpy()
    mask2 = rs2.xmask.to_numpy()
    truth = np.concatenate((rsnp1, rsnp2))
    mask = np.concatenate((mask1, mask2))
    truth[mask] = get_default_null_value(truth.dtype)
    assert truth.ndim == 3
    test = general.band_concat([rs1, rs2])
    assert_valid_raster(test)
    assert test.crs == rs1.crs
    assert test.dtype == truth.dtype
    assert test.null_value == get_default_null_value(test.dtype)
    assert test.shape == truth.shape
    assert np.allclose(test, truth)
    truth = np.concatenate((rsnp1, rsnp1, rsnp2, truth))
    mask = np.concatenate((mask1, mask1, mask2, mask))
    truth[mask] = get_default_null_value(truth.dtype)
    test = general.band_concat([rs1, rs1, rs2, test])
    assert_valid_raster(test)
    assert test.crs == rs1.crs
    assert test.dtype == truth.dtype
    assert test.null_value == get_default_null_value(test.dtype)
    assert test.shape == truth.shape
    assert np.allclose(test, truth)


@pytest.mark.filterwarnings("ignore:The null value")
def test_band_concat_bool_rasters():
    rs1 = testdata.raster.dem_small > -100
    rs2 = rs1.copy()
    result = general.band_concat((rs1, rs2))
    assert result.null_value == get_default_null_value(result.dtype)
    assert result.dtype == np.dtype(bool)
    assert np.array(result).all()

    result = general.band_concat((rs1, rs2))
    # The null value will be changed to match type
    assert result.null_value == get_default_null_value(bool)
    assert result.dtype == np.dtype("?")
    assert np.all(np.array(result) == 1)


def test_band_concat_errors():
    with pytest.raises(ValueError):
        general.band_concat([])


@pytest.mark.parametrize("inc", ["left", "right", "both", "none"])
@pytest.mark.parametrize(
    "raster,mapping",
    [
        (arange_raster((1, 5, 5)), (0, 5, 1)),
        (arange_raster((1, 5, 5)), [(0, 5, 1), (5, 15, -1)]),
        (
            arange_raster((1, 5, 5)),
            # Test multiple with potential conflict in last 2
            [(0, 1, 0), (1, 2, 1), (2, 3, 8), (8, 9, 2)],
        ),
        # Test precedence
        (arange_raster((1, 5, 5)), [(0, 2, 0), (1, 2, 1)]),
        (arange_raster((1, 5, 5)) + 0.1, [(0, 2, 0), (1, 2, 1)]),
    ],
)
def test_remap_range(raster, mapping, inc):
    truth = raster.to_numpy()
    if np.asarray(mapping).shape == (3,):
        mapping = [mapping]
    out_dtype = get_common_dtype([m[-1] for m in mapping])
    out_dtype = np.promote_types(raster.dtype, out_dtype)
    truth = truth.astype(out_dtype)
    remapped_mask = np.zeros_like(truth, dtype=bool)
    for m in mapping:
        if inc == "left":
            match = (truth >= m[0]) & (truth < m[1])
        elif inc == "right":
            match = (truth > m[0]) & (truth <= m[1])
        elif inc == "both":
            match = (truth >= m[0]) & (truth <= m[1])
        elif inc == "none":
            match = (truth > m[0]) & (truth < m[1])
        truth[(~remapped_mask) & match] = m[2]
        remapped_mask |= match

    result1 = general.remap_range(raster, mapping, inc)
    result2 = raster.remap_range(mapping, inc)
    assert_valid_raster(result1)
    assert_valid_raster(result2)
    assert result1.dtype == out_dtype
    assert result2.dtype == out_dtype
    assert np.allclose(result1, result2)
    assert np.allclose(result1, truth)
    assert np.allclose(result2, truth)


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
    data = rast.to_numpy().copy()
    hist = np.zeros(data.shape, dtype=bool)
    for left, right, new in mapping:
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
    rsnp = rs.to_numpy()
    mapping = (0, 5, 1)
    result = general.remap_range(rs, mapping)
    truth = rsnp.copy()
    truth[(rsnp >= mapping[0]) & (rsnp < mapping[1])] = mapping[2]
    assert rs.dtype == np.dtype("float16")
    assert result.dtype == np.dtype("float16")
    assert np.allclose(result, truth)

    rs = Raster(np.arange(25).reshape((5, 5))).astype("int8")
    rsnp = rs.to_numpy()
    mapping = (0, 5, 2.0)
    result = general.remap_range(rs, mapping)
    truth = rsnp.copy().astype("float16")
    truth[(rsnp >= mapping[0]) & (rsnp < mapping[1])] = mapping[2]
    assert rs.dtype == np.dtype("int8")
    assert result.dtype == np.dtype("float16")
    assert np.allclose(result, truth)


def test_remap_range_null_mapping():
    raster = testdata.raster.dem_small.set_null_value(None)
    assert raster.null_value is None
    nv = get_default_null_value(raster.dtype)
    mapping = (0, 1500, None)
    truth_rast = xr.where(raster.xdata > 1500, raster.xdata, nv)
    truth_mask = truth_rast == nv

    result = general.remap_range(raster, mapping)
    assert_valid_raster(result)
    assert result.dtype == raster.dtype
    assert result.null_value == get_default_null_value(result.dtype)
    assert np.allclose(result.xdata, truth_rast)
    assert np.allclose(result.xmask, truth_mask)

    raster = raster.set_null_value(-999)
    mapping = [(0, 1500, None), (1500, 2000, 1)]
    truth_rast = xr.where(raster.xdata > 1500, raster.xdata, raster.null_value)
    truth_rast = xr.where(
        (truth_rast >= 1500) & (truth_rast < 2000), 1, truth_rast
    )
    truth_mask = truth_rast == raster.null_value

    result = general.remap_range(raster, mapping)
    assert_valid_raster(result)
    assert result.dtype == raster.dtype
    assert result.null_value == raster.null_value
    assert np.allclose(result.xdata, truth_rast)
    assert np.allclose(result.xmask, truth_mask)


def test_remap_range_errors():
    rs = testdata.raster.dem_small
    # TypeError if not scalars
    with pytest.raises(TypeError):
        general.remap_range(rs, (None, 2, 4))
    with pytest.raises(TypeError):
        general.remap_range(rs, (0, "2", 4))
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


def _make_raster(data, mask, dtype=None):
    data = np.asarray(data)
    mask = np.asarray(mask).astype(bool)
    # Allow nan values as valid data. Raster() automatically converts nans to
    # null.
    nan_mask = np.isnan(data)
    if mask.any():
        nan_mask[mask] = False
    sent = -99
    if nan_mask.any():
        data[nan_mask] = sent
    r = Raster(data).set_crs("EPSG:3857")
    if nan_mask.any():
        r._ds.raster.data = da.where(nan_mask, np.nan, r._ds.raster.data)
    mask = Raster(mask)
    if mask is not None:
        r = r.set_null(mask)
    if dtype is not None:
        r = r.astype(dtype, False)
    return r


@pytest.mark.parametrize(
    "condition,x,y,expected",
    [
        # 0
        (
            _make_raster([[0, 0], [1, 1]], np.zeros((2, 2)), bool),
            _make_raster(np.ones((2, 2)), np.zeros((2, 2))),
            _make_raster(np.full((2, 2), 2), np.zeros((2, 2))),
            _make_raster([[2, 2], [1, 1]], np.zeros((2, 2))),
        ),
        # 1
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            _make_raster(np.ones((2, 2)), np.zeros((2, 2))),
            _make_raster(np.full((2, 2), 2), np.zeros((2, 2))),
            _make_raster([[0, 2], [1, 1]], [[1, 0], [0, 0]]),
        ),
        # 2
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            _make_raster(np.ones((2, 2)), [[0, 1], [0, 0]]),
            _make_raster(np.full((2, 2), 2), [[0, 0], [1, 0]]),
            _make_raster([[0, 2], [1, 1]], [[1, 0], [0, 0]]),
        ),
        # 3
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            _make_raster(np.ones((2, 2)), [[0, 0], [0, 1]]),
            _make_raster(np.full((2, 2), 2), [[0, 1], [0, 0]]),
            _make_raster([[2, 2], [1, 1]], [[1, 1], [0, 1]]),
        ),
        # 4
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            _make_raster(np.ones((2, 2)), [[0, 0], [0, 1]]),
            2,
            _make_raster([[2, 2], [1, 1]], [[1, 0], [0, 1]]),
        ),
        # 5
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            1,
            _make_raster(np.full((2, 2), 2), [[0, 1], [0, 0]]),
            _make_raster([[2, 2], [1, 1]], [[1, 1], [0, 0]]),
        ),
        # 6
        (
            _make_raster([[0, 0], [1, 1]], [[0, 0], [0, 0]], bool),
            1,
            2,
            _make_raster([[2, 2], [1, 1]], [[0, 0], [0, 0]]),
        ),
        # 7
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            1,
            2,
            _make_raster([[2, 2], [1, 1]], [[1, 0], [0, 0]], np.uint8),
        ),
        # 8
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            1,
            np.nan,
            _make_raster(
                [[np.nan, np.nan], [1, 1]], [[1, 0], [0, 0]], "float16"
            ),
        ),
        # 9
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            None,
            _make_raster(np.full((2, 2), 2), [[0, 1], [0, 0]]),
            _make_raster([[2, 2], [1, 1]], [[1, 1], [1, 1]]),
        ),
        # 10
        (
            _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool),
            _make_raster(np.ones((2, 2)), [[0, 0], [0, 1]]),
            None,
            _make_raster([[2, 2], [1, 1]], [[1, 1], [0, 1]]),
        ),
    ],
)
def test_where(condition, x, y, expected):
    masked = any(
        v._masked for v in (condition, x, y) if isinstance(v, Raster)
    ) or any(v is None for v in (x, y))

    result = general.where(condition, x, y)
    assert_valid_raster(result)
    assert_rasters_similar(result, condition)
    assert result._masked == masked
    assert np.allclose(result, expected, equal_nan=True)
    assert np.allclose(result.mask.compute(), expected.mask.compute())

    if isinstance(x, Raster):
        result = x.where(condition, y)
        assert_valid_raster(result)
        assert_rasters_similar(result, condition)
        assert result._masked == masked
        assert np.allclose(result, expected, equal_nan=True)
        assert np.allclose(result.mask.compute(), expected.mask.compute())


def test_where_both_none():
    with pytest.raises(ValueError):
        cond = _make_raster([[0, 0], [1, 1]], [[1, 0], [0, 0]], bool)
        general.where(cond, None, None)


@pytest.mark.parametrize("unmapped_to_null", [True, False])
@pytest.mark.parametrize(
    "raster,mapping,expected_out_dtype",
    [
        (
            arange_raster((2, 10, 10))
            .set_crs("EPSG:3857")
            .set_null_value(99)
            .set_null_value(100)
            .astype("int16"),
            {0: -1, 1: -2, 2: -3, 120: -120, 150: None},
            np.dtype("int16"),
        ),
        # Check promotion from uint16 to int32
        (
            arange_raster((2, 10, 10))
            .set_crs("EPSG:3857")
            .set_null_value(99)
            .set_null_value(100)
            .astype("uint16"),
            {0: -1, 1: -2, 2: -3, 120: -120, 150: -150},
            np.dtype("int32"),
        ),
        # Make sure that float values in the mapping cause promotion (i2 -> f4)
        (
            arange_raster((2, 10, 10)).astype("int16"),
            {0: -1.0, 1: -2, 2: -3, 120: -120, 150: -150},
            np.dtype("float32"),
        ),
        (
            arange_raster((2, 10, 10)).astype("int16"),
            {
                0: -1.0,
                1: -2,
                2: -3,
                120: -120,
                150: np.finfo("float32").max + 1e35,
            },
            np.dtype("float64"),
        ),
        # make sure that float input rasters are accepted
        (
            arange_raster((2, 10, 10)).astype("float32"),
            {0.0: -1.0, 1: -2, 2: -3, 120: -120, 150: 1e23},
            np.dtype("float32"),
        ),
    ],
)
def test_reclassify(raster, mapping, unmapped_to_null, expected_out_dtype):
    tdata = raster.data.compute().astype(expected_out_dtype)
    mapped = np.zeros_like(tdata, dtype=bool)
    nv = (
        raster.null_value
        if raster._masked
        else get_default_null_value(expected_out_dtype)
    )
    for f, t in mapping.items():
        mapped |= tdata == f
        if t is None:
            t = nv
        tdata[tdata == f] = t
    if unmapped_to_null:
        tdata[~mapped] = nv
    tmask = tdata == nv
    result = general.reclassify(raster, mapping, unmapped_to_null)

    assert_valid_raster(result)
    assert_rasters_similar(raster, result)
    assert result.dtype == expected_out_dtype
    if (
        raster._masked
        or unmapped_to_null
        or any(v is None for v in mapping.values())
    ):
        assert result._masked
        assert result.null_value == nv
    assert np.allclose(tdata, result.data.compute())
    assert np.allclose(tmask, result.mask.compute())


@pytest.mark.parametrize(
    "fd,expected_mapping",
    [
        (io.StringIO(""), {}),
        (
            io.StringIO("0:3\n1.:2\n.2:2\n1.2:3.\n5:.0"),
            {0: 3, 1.0: 2, 0.2: 2, 1.2: 3.0, 5: 0.0},
        ),
        (
            io.StringIO("1e23:2\n-1e-23 :2\n1.2e+23:3"),
            {1e23: 2, -1e-23: 2, 1.2e23: 3},
        ),
        (io.StringIO("1:\t 2\n 3 : 4 \n"), {1: 2, 3: 4}),
        (io.StringIO("1:NoData"), {1: None}),
    ],
)
def test_reclassify_mapping_file_parsing(fd, expected_mapping):
    assert general._parse_ascii_remap_file(fd) == expected_mapping


@pytest.mark.parametrize(
    "fd,error",
    [
        (io.StringIO("1.e:2"), general.RemapFileParseError),
        (io.StringIO(":2"), general.RemapFileParseError),
        (io.StringIO("1:"), general.RemapFileParseError),
        (io.StringIO("12"), general.RemapFileParseError),
        (io.StringIO("12 34"), general.RemapFileParseError),
        (io.StringIO("1:2\n1:3"), ValueError),
        (io.StringIO("1:ND"), general.RemapFileParseError),
    ],
)
def test_reclassify__parse_ascii_remap_file_errors(fd, error):
    with pytest.raises(error):
        assert general._parse_ascii_remap_file(fd)
