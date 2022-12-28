import dask.array as da
import numba as nb
import numpy as np
import pytest
import xarray as xr

from raster_tools.distance import proximity as prx
from raster_tools.dtypes import F32
from raster_tools.masking import get_default_null_value
from raster_tools.raster import Raster


@pytest.mark.parametrize(
    "x1,x2,truth",
    [
        ([0, 0], [0, 0], 0),
        ([0, 0], [0, 1], 360),
        ([0, 0], [1, 1], 45),
        ([0, 0], [2, 0], 90),
        ([0, 0], [1, -1], 135),
        ([0, 0], [0, -1], 180),
        ([0, 0], [-1, -1], 225),
        ([0, 0], [-1, 0], 270),
        ([0, 0], [-1, 1], 315),
    ],
)
def test_proximity_direction(x1, x2, truth):
    assert prx._direction(*x1, *x2) == truth


esri_src = np.array(
    [
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0],
    ]
)
esri_euc_prox = np.array(
    [
        [1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        [1.4142135, 1.0, 0.0, 1.0, 2.0, 3.0],
        [2.236068, 1.4142135, 1.0, 1.4142135, 2.236068, 3.1622777],
        [2.0, 2.236068, 2.0, 2.236068, 2.828427, 3.6055512],
        [1.0, 1.4142135, 2.236068, 3.1622777, 3.6055512, 4.2426405],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    ],
    dtype=np.float32,
)
esri_euc_alloc = np.array(
    [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [2, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1],
        [2, 2, 2, 2, 2, 1],
    ]
)
esri_euc_dir = np.array(
    [
        [90.0, 0.0, 0.0, 270.0, 270.0, 270.0],
        [45.0, 360.0, 0.0, 270.0, 270.0, 270.0],
        [26.565052, 45.0, 360.0, 315.0, 296.56506, 288.43494],
        [180.0, 26.565052, 360.0, 333.43494, 315.0, 303.69006],
        [180.0, 225.0, 243.43495, 341.56506, 326.30994, 315.0],
        [0.0, 270.0, 270.0, 270.0, 270.0, 323.1301],
    ],
    dtype=np.float32,
)


def test_proximity_analysis_esri_example():
    src = Raster(esri_src).set_null_value(0)

    prox, alloc, dirn = prx.proximity_analysis(src)

    assert np.allclose(prox, esri_euc_prox)
    assert np.allclose(alloc, esri_euc_alloc)
    assert np.allclose(dirn, esri_euc_dir)
    assert prox.dtype == F32
    assert prox.xdata.compute().dtype == F32
    assert alloc.dtype == src.dtype
    assert alloc.xdata.compute().dtype == src.dtype
    assert dirn.dtype == F32
    assert dirn.xdata.compute().dtype == F32

    mask = esri_euc_prox > 3
    prox_truth = esri_euc_prox.copy()
    prox_truth[mask] = prox.null_value
    alloc_truth = esri_euc_alloc.copy()
    alloc_truth[mask] = alloc.null_value
    dir_truth = esri_euc_dir.copy()
    dir_truth[mask] = dirn.null_value

    prox, alloc, dirn = prx.proximity_analysis(src, max_distance=3)

    assert np.allclose(prox, prox_truth)
    assert np.allclose(prox.mask.compute(), mask)
    assert np.allclose(alloc, alloc_truth)
    assert np.allclose(alloc.mask.compute(), mask)
    assert np.allclose(dirn, dir_truth)
    assert np.allclose(dirn.mask.compute(), mask)


@nb.jit(nopython=True, nogil=True)
def euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


@nb.jit(nopython=True, nogil=True)
def taxi(x1, x2):
    return np.sum(np.abs(x1 - x2))


@nb.jit(nopython=True, nogil=True)
def chessboard(x1, x2):
    return np.max(np.abs(x1 - x2))


@nb.jit(nopython=True, nogil=True)
def haversine(x1, x2):
    x1 = np.radians(x1)
    x2 = np.radians(x2)
    dx = x2 - x1
    sindx = np.sin(dx / 2)
    h = sindx[1] ** 2 + (np.cos(x1[1]) * np.cos(x2[1]) * sindx[0] ** 2)
    return 2 * 6371009.0 * np.arcsin(np.sqrt(h))


@nb.jit(nopython=True, nogil=True)
def apply_metric(x, xc, yc, func, max_distance=np.inf, is_cb=False):
    prox = np.full_like(x, np.nan, dtype=F32)
    alloc = np.full_like(x, -99, dtype=x.dtype)
    dirn = np.full_like(x, np.nan, dtype=F32)
    ny, nx = x.shape

    target_values = np.array([v for v in x.ravel() if v != 0])
    target_locs = np.array(
        [
            (i % x.shape[0], i // x.shape[1])
            for i, v in enumerate(x.ravel())
            if v != 0
        ]
    )
    nt = len(target_values)

    for i in range(ny):
        for j in range(nx):
            x1 = np.array([xc[j], yc[i]])
            x2s = np.zeros((nt, 2))
            for k in range(nt):
                ti, tj = target_locs[k]
                x2s[k] = (xc[ti], yc[tj])
            is_target = False
            for k, x2 in enumerate(x2s):
                is_target = (x1[0] == x2[0]) and (x1[1] == x2[1])
                if is_target:
                    prox[i, j] = 0
                    alloc[i, j] = target_values[k]
                    dirn[i, j] = 0
                    break
            if is_target:
                continue
            dists = np.zeros(nt, dtype=np.float32)
            for k in range(nt):
                x2 = x2s[k]
                dists[k] = func(x1, x2)
            if is_cb:
                imin = 0
                for k in range(nt):
                    x2 = x2s[k]
                    if x2[0] == x1[0] or x2[1] == x1[1]:
                        imin = k
            else:
                imin = np.argmin(dists)
            if dists[imin] > max_distance:
                continue
            x2 = x2s[imin]
            prox[i, j] = dists[imin]
            alloc[i, j] = target_values[imin]
            dirn[i, j] = prx._direction(x1[0], x1[1], x2[0], x2[1])
    return prox, alloc, dirn


def build_raster(src):
    x = np.arange(src.shape[1]) * 30
    y = np.arange(src.shape[0] - 1, -1, -1) * 30
    xrs = xr.DataArray(src, dims=("y", "x"), coords=(y, x))
    rs = Raster(xrs).set_null_value(0)
    return rs


src1 = np.zeros((100, 100))
src1[20] = np.arange(1, 101)
src2 = np.zeros((100, 100))
src2[100 // 2, 100 // 2] = 1
src3 = np.zeros((100, 100))
src3[10:15, 20] = 1
src3[80, 60:65] = np.arange(2, 7)
src3[-1, 0] = 7
srcs = [src1, src2, src2]
srcs = [build_raster(s) for s in srcs]


@pytest.mark.parametrize("max_distance", [np.inf, 20 * 30])
@pytest.mark.parametrize(
    "metric,truth_func",
    [
        ("euclidean", euclidean),
        (None, euclidean),
        ("taxi", taxi),
        ("taxicab", taxi),
        ("manhatten", taxi),
        ("city_block", taxi),
        ("chebyshev", chessboard),
        ("chessboard", chessboard),
    ],
)
@pytest.mark.parametrize("src", srcs)
def test_proximity_metric(src, metric, truth_func, max_distance):
    data = src.values[0]
    x = src.x
    y = src.y

    rprox, ralloc, rdirn = prx.proximity_analysis(
        src, distance_metric=metric, max_distance=max_distance
    )
    tprox, talloc, tdirn = apply_metric(
        data,
        x,
        y,
        truth_func,
        max_distance,
        is_cb=metric in ("chebyshev", "chessboard"),
    )
    mask = np.isnan(tprox)
    tprox[mask] = get_default_null_value(F32)
    talloc[mask] = get_default_null_value(src.dtype)
    tdirn[mask] = get_default_null_value(F32)

    assert np.allclose(rprox, tprox)
    assert rprox.dtype == F32
    assert np.allclose(ralloc, talloc)
    assert ralloc.dtype == src.dtype
    assert np.allclose(rdirn, tdirn)
    assert rdirn.dtype == F32


def test_proximity_great_circle():
    tprox = Raster("tests/data/raster/prox_haversine_proximity.tif")
    talloc = Raster("tests/data/raster/prox_haversine_allocation.tif")
    tdirn = Raster("tests/data/raster/prox_haversine_direction.tif")

    src = Raster("tests/data/raster/prox_src.tif")

    rprox, ralloc, rdirn = prx.proximity_analysis(
        src, distance_metric="haversine"
    )
    assert np.allclose(rprox, tprox)
    assert np.allclose(ralloc, talloc)
    assert np.allclose(rdirn, tdirn)

    max_dist = 1.5e6
    mask = tprox.values > max_dist
    tprox.xdata.data = da.where(mask, tprox.null_value, tprox.data)
    talloc.xdata.data = da.where(mask, talloc.null_value, talloc.data)
    tdirn.xdata.data = da.where(mask, tdirn.null_value, tdirn.data)

    rprox, ralloc, rdirn = prx.proximity_analysis(
        src, distance_metric="haversine", max_distance=max_dist
    )
    assert np.allclose(rprox, tprox)
    assert np.allclose(ralloc, talloc)
    assert np.allclose(rdirn, tdirn)
