from functools import partial

import dask
import numpy as np
import pytest
from scipy import ndimage, stats

from raster_tools import focal
from tests.utils import assert_valid_raster, make_raster


def test_get_focal_window_circle_rect():
    truths = [
        np.array([[1.0]]),
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ),
    ]
    truths = [t.astype(bool) for t in truths]
    for r, truth in zip(range(1, len(truths) + 1), truths, strict=True):
        window = focal.get_focal_window(r)
        assert np.allclose(window, truth)
        assert window.dtype == bool
    for w in range(1, 6):
        for h in range(1, 6):
            window = focal.get_focal_window(w, h)
            assert np.allclose(window, np.ones((w, h)))
            assert window.dtype == bool


def test_get_focal_window_annulus():
    truths = [
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [1, 1, 0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 1, 0, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ),
        np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ),
    ]
    truths = [t.astype(bool) for t in truths]
    i = 0
    for r2 in range(1, 7):
        for r1 in range(1, r2):
            if r1 >= r2:
                continue
            w = focal.get_focal_window((r1, r2))
            t = truths[i]
            assert np.allclose(w, t)
            i += 1


@pytest.mark.parametrize("r", [-2, -1, 0])
def test_get_focal_window_invalid_radius(r):
    with pytest.raises(ValueError):
        focal.get_focal_window(r)


@pytest.mark.parametrize("r", [2.3, 4.999, 3.0, None, "4"])
def test_get_focal_window_non_int_radius(r):
    with pytest.raises(TypeError):
        focal.get_focal_window(r)


@pytest.mark.parametrize(
    "rvalues", [(0, 3), (3, 3), (3, 1), (-1, 3), (-3, -1)]
)
def test_get_focal_window_invalid_annulus(rvalues):
    with pytest.raises(ValueError):
        focal.get_focal_window(rvalues)


@pytest.mark.parametrize("rvalues", [(3.0, 4), (2.1, 3.0), (3, 5.0)])
def test_get_focal_window_non_int_annulus(rvalues):
    with pytest.raises(TypeError):
        focal.get_focal_window(rvalues)


@pytest.mark.parametrize("w,h", [(-1, 4), (0, 3), (3, -3)])
def test_get_focal_window_invalid_rect(w, h):
    with pytest.raises(ValueError):
        focal.get_focal_window(w, h)


@pytest.mark.parametrize("w,h", [(3.0, 3.0), (4, 3.0)])
def test_get_focal_window_non_int_rect(w, h):
    with pytest.raises(TypeError):
        focal.get_focal_window(w, h)


def test_get_focal_window_annulus_with_height():
    with pytest.raises(ValueError):
        focal.get_focal_window((2, 4), 5)


def test_focal_return_dask():
    x = np.arange(16.0).reshape(1, 4, 4)
    kern = focal.get_focal_window(2)
    assert dask.is_dask_collection(focal._focal(x, kern, "max", True))
    assert dask.is_dask_collection(focal._focal(x, kern, "max", False))
    xd = dask.array.from_array(x)
    assert dask.is_dask_collection(focal._focal(xd, kern, "max", True))
    assert dask.is_dask_collection(focal._focal(xd, kern, "max", False))


def asm(x):
    c = {}
    n = 0
    for v in x.ravel():
        if not np.isnan(v):
            if v in c:
                c[v] += 1
            else:
                c[v] = 1
            n += 1
    if n == 0:
        return 0.0
    p = np.array([cnt / n for cnt in c.values()])
    return np.sum(p * p)


def entropy(x):
    c = {}
    xn = x[~np.isnan(x)]
    for v in xn:
        if v in c:
            c[v] += 1
        else:
            c[v] = 1
    n = sum(c.values())
    if not n:
        return 0.0
    p = np.array([cnt / n for cnt in c.values()])
    return np.nansum(-p * np.log(p))


def mode(x):
    if x[~np.isnan(x)].size == 0:
        return np.nan
    m = stats.mode(x, axis=None, nan_policy="omit", keepdims=True)
    return m.mode[0]


def unique(x):
    u = np.unique(x[~np.isnan(x)])
    if len(u):
        return len(u)
    return 0


def correlate(x, kern):
    return np.nansum(x * kern.ravel())


@pytest.mark.parametrize(
    "kernel,kernel_params",
    [
        (focal.get_focal_window(2), (2,)),
        (focal.get_focal_window(3, 3), (3, 3)),
        (focal.get_focal_window((2, 3)), ((2, 3),)),
    ],
)
@pytest.mark.parametrize(
    "filter_name,filter_func",
    [
        ("asm", asm),
        ("entropy", entropy),
        ("max", np.nanmax),
        ("mean", np.nanmean),
        ("median", np.nanmedian),
        ("mode", mode),
        ("min", np.nanmin),
        ("std", np.nanstd),
        ("sum", np.nansum),
        ("var", np.nanvar),
        ("unique", unique),
    ],
)
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
@pytest.mark.filterwarnings("ignore:Mean of empty slice")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0 for slice")
def test_focal(filter_name, filter_func, kernel, kernel_params):
    rx = make_raster(
        "arange",
        dtype="float64",
        shape=(1, 8, 8),
        null=np.nan,
        null_pattern=np.s_[:, :3, :3],
    )
    x = rx.to_numpy()
    mask = rx.mask.compute()
    truth = ndimage.generic_filter(
        x[0],
        filter_func,
        footprint=kernel,
        mode="constant",
        cval=np.nan,
    )
    res = focal._focal(x, kernel, filter_name, True).compute()
    assert np.allclose(truth, res, equal_nan=True)

    result_raster = focal.focal(rx, filter_name, *kernel_params)
    assert_valid_raster(result_raster)
    res_full = result_raster.to_numpy()[0]
    # Fill with null value
    assert np.allclose(
        np.where(mask, result_raster.null_value, truth),
        res_full,
        equal_nan=True,
    )


def test_correlate_return_dask():
    x = np.arange(16.0).reshape(1, 4, 4)
    kern = focal.get_focal_window(2)
    assert dask.is_dask_collection(focal._correlate(x, kern))


@pytest.mark.parametrize("kern", [np.ones((5, 5)), np.ones((4, 4))])
@pytest.mark.parametrize("nan_aware", [False, True])
@pytest.mark.parametrize("mode", ["reflect", "nearest", "wrap", "constant"])
def test_focal_correlate(kern, nan_aware, mode):
    func = partial(correlate, kern=kern)
    data = np.arange(64.0).reshape(8, 8)
    if nan_aware:
        data[:3, :3] = np.nan
    origin = [-1 if d % 2 == 0 else 0 for d in kern.shape]
    truth = ndimage.generic_filter(
        data, func, size=kern.shape, mode=mode, origin=origin
    )
    test = focal._correlate(
        data[None], kern, mode=mode, nan_aware=nan_aware
    ).compute()
    assert np.allclose(truth, test, equal_nan=nan_aware)


def test_focal_integration():
    rs = make_raster("arange", dtype="float32", shape=(4, 16, 16), null=np.nan)
    rsnp = rs.to_numpy()
    truth = rsnp.astype(float)
    for bnd in range(truth.shape[0]):
        truth[bnd] = ndimage.generic_filter(
            truth[bnd], np.nanmean, size=3, mode="constant", cval=np.nan
        )
    res_raster = focal.focal(rs, "mean", 3, 3)
    assert_valid_raster(res_raster)
    assert res_raster.crs == rs.crs
    res = res_raster.to_numpy()
    assert np.allclose(truth, res, equal_nan=True)
    truth = rsnp.astype(float)
    kern = focal.get_focal_window(3)
    for bnd in range(truth.shape[0]):
        truth[bnd] = ndimage.generic_filter(
            truth[bnd],
            np.nanmedian,
            footprint=kern,
            mode="constant",
            cval=np.nan,
        )
    res_raster = focal.focal(rs, "median", 3)
    assert_valid_raster(res_raster)
    assert res_raster.crs == rs.crs
    res = res_raster.to_numpy()
    assert np.allclose(truth, res, equal_nan=True)


def test_focal_integration_raster_input():
    rs = make_raster("arange", dtype="float32", shape=(4, 16, 16), null=np.nan)
    rsnp = rs.to_numpy()
    with pytest.raises(TypeError):
        focal.focal(rsnp, "median", 3)
    res_raster = focal.focal(rs, "mean", 1)
    assert_valid_raster(res_raster)
    assert res_raster.crs == rs.crs
    res = res_raster.to_numpy()
    assert np.allclose(rsnp, res, equal_nan=True)


def test_focal_output_type():
    rs = (
        make_raster("arange", dtype="float32", shape=(4, 16, 16), null=np.nan)
        * 100
    )
    rs_masked = rs.set_null_value(-1).astype(int)
    rsi = rs.set_null_value(None).astype(int)

    # Masked
    assert rs_masked._masked
    assert rs_masked.dtype.kind == "i"
    res = focal.focal(rs_masked, "mode", 3)
    assert_valid_raster(res)
    res = res.eval()
    assert res.dtype.kind == "f"
    res = focal.focal(rs_masked, "unique", 3)
    assert_valid_raster(res)
    res = res.eval()
    assert res.dtype.kind == "f"
    res = focal.focal(rs_masked, "mean", 3)
    assert_valid_raster(res)
    res = res.eval()
    assert res.dtype.kind == "f"

    # Unmasked
    assert not rsi._masked
    assert rsi.dtype.kind == "i"
    res = focal.focal(rsi, "mode", 3).eval()
    assert res.dtype.kind == "i"
    res = focal.focal(rsi, "unique", 3).eval()
    assert res.dtype.kind == "u"
    res = focal.focal(rsi, "mean", 3).eval()
    assert res.dtype.kind == "f"


def test_correlate_integration():
    rs = make_raster("arange", dtype="float64", shape=(4, 16, 16), null=np.nan)
    rsnp = rs.to_numpy()
    truth = rsnp.astype(float)
    kernel = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]).astype(float)
    for bnd in range(truth.shape[0]):
        truth[bnd] = ndimage.generic_filter(
            truth[bnd], np.sum, footprint=kernel, mode="constant"
        )
    res_raster = focal.correlate(rs, kernel)
    assert_valid_raster(res_raster)
    assert res_raster.crs == rs.crs
    res = res_raster.to_numpy()
    assert np.allclose(truth, res, equal_nan=False)

    rs_masked = make_raster(
        "arange",
        dtype="float64",
        shape=(4, 16, 16),
        null=np.nan,
        null_pattern=np.s_[:, :3, :3],
    )
    rsnp_masked = rs_masked.to_numpy()
    truth = rsnp_masked.astype(float)
    kern = focal.get_focal_window(3)
    for bnd in range(truth.shape[0]):
        truth[bnd] = ndimage.generic_filter(
            truth[bnd],
            np.nansum,
            footprint=kern,
            mode="constant",
        )
    res_raster = focal.correlate(rs_masked, kern)
    assert_valid_raster(res_raster)
    assert res_raster.crs == rs_masked.crs
    res = res_raster.to_numpy()
    truth[res_raster.xmask.to_numpy()] = res_raster.null_value
    assert np.allclose(truth, res, equal_nan=True)


def test_convolve_integration():
    rs = make_raster("arange", dtype="float64", shape=(4, 16, 16), null=np.nan)
    rsnp = rs.to_numpy()
    truth = rsnp.astype(float)
    kernel = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]).astype(float)
    for bnd in range(truth.shape[0]):
        truth[bnd] = ndimage.generic_filter(
            truth[bnd],
            np.sum,
            footprint=kernel[::-1, ::-1],
            mode="constant",
        )
    res_raster = focal.convolve(rs, kernel)
    assert_valid_raster(res_raster)
    assert res_raster.crs == rs.crs
    res = res_raster.to_numpy()
    assert np.allclose(truth, res, equal_nan=False)

    rs_masked = make_raster(
        "arange",
        dtype="float64",
        shape=(4, 16, 16),
        null=np.nan,
        null_pattern=np.s_[:, :3, :3],
    )
    rsnp_masked = rs_masked.to_numpy()
    truth = rsnp_masked.astype(float)
    for bnd in range(truth.shape[0]):
        truth[bnd] = ndimage.generic_filter(
            truth[bnd],
            np.nansum,
            footprint=kernel[::-1, ::-1],
            mode="constant",
        )
    res_raster = focal.convolve(rs_masked, kernel)
    assert_valid_raster(res_raster)
    assert res_raster.crs == rs_masked.crs
    res = res_raster.to_numpy()
    truth[res_raster.xmask.to_numpy()] = res_raster.null_value
    assert np.allclose(truth, res, equal_nan=True)


def test_correlate_integration_raster_input():
    rs = make_raster("arange", dtype="float32", shape=(4, 16, 16), null=np.nan)
    rsnp = rs.to_numpy()
    with pytest.raises(TypeError):
        focal.correlate(rsnp, 3)
    res_raster = focal.correlate(rs, np.ones((1, 1)))
    assert_valid_raster(res_raster)
    assert res_raster.crs == rs.crs
    res = res_raster
    assert np.allclose(rsnp, res, equal_nan=True)


def test_correlate_output_type():
    rs = (
        make_raster("arange", dtype="float32", shape=(4, 16, 16), null=np.nan)
        * 100
    )
    rs = rs.set_null_value(-1)
    rs = rs.astype(int)

    assert rs._masked
    assert rs.dtype.kind == "i"
    res = focal.correlate(rs, np.ones((3, 3), dtype=int)).eval()
    assert res.dtype == rs.dtype

    res = focal.correlate(rs, np.ones((3, 3), dtype=float)).eval()
    assert res.dtype.kind == "f"
