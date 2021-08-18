import dask
import numpy as np
import unittest
import warnings
from scipy import ndimage, stats

from raster_tools import focal


def array_eq_all(ar1, ar2):
    return (ar1 == ar2).all()


class TestFocalWindow(unittest.TestCase):
    def test_get_focal_window_circle_rect(self):
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
        for r, truth in zip(range(1, len(truths) + 1), truths):
            window = focal.get_focal_window(r)
            self.assertTrue(array_eq_all(window, truth))
            self.assertEqual(window.dtype, bool)
        for w in range(1, 6):
            for h in range(1, 6):
                window = focal.get_focal_window(w, h)
                self.assertTrue(array_eq_all(window, np.ones((w, h))))
                self.assertEqual(window.dtype, bool)

    def test_get_focal_window_annulus(self):
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
                self.assertTrue(array_eq_all(w, t))
                i += 1

    def test_get_focal_window_errors(self):
        for r in [-2, -1, 0]:
            with self.assertRaises(ValueError):
                focal.get_focal_window(r)
        for r in [2.3, 4.999, 3.0, None, "4"]:
            with self.assertRaises(TypeError):
                focal.get_focal_window(r)
        for rvalues in [(0, 3), (3, 3), (3, 1), (-1, 3), (-3, -1)]:
            with self.assertRaises(ValueError):
                focal.get_focal_window(rvalues)
        for rvalues in [(3.0, 4), (2.1, 3.0), (3, 5.0)]:
            with self.assertRaises(TypeError):
                focal.get_focal_window(rvalues)
        for args in [(-1, 4), (0, 3), (3, -3)]:
            with self.assertRaises(ValueError):
                focal.get_focal_window(*args)
        for args in [(3.0, 3.0), (4, 3.0)]:
            with self.assertRaises(TypeError):
                focal.get_focal_window(*args)
        with self.assertRaises(ValueError):
            focal.get_focal_window((2, 4), 5)

    def test_focal_return_dask(self):
        x = np.arange(16.0).reshape(4, 4)
        kern = focal.get_focal_window(2)
        self.assertTrue(
            dask.is_dask_collection(focal.focal(x, kern, "max", True))
        )
        self.assertTrue(
            dask.is_dask_collection(focal.focal(x, kern, "max", False))
        )
        xd = dask.array.from_array(x)
        self.assertTrue(
            dask.is_dask_collection(focal.focal(xd, kern, "max", True))
        )
        self.assertTrue(
            dask.is_dask_collection(focal.focal(xd, kern, "max", False))
        )

    def test_focal(self):
        filters = [
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
        ]
        x = np.arange(64.0).reshape(8, 8)
        x[:3, :3] = np.nan
        kernels = [
            focal.get_focal_window(2),
            focal.get_focal_window(3, 3),
            focal.get_focal_window((2, 3)),
        ]
        for kern in kernels:
            for stat, func in filters:
                with warnings.catch_warnings():
                    # Ignore nan related warnings
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    truth = ndimage.generic_filter(
                        x, func, footprint=kern, mode="constant", cval=np.nan
                    )
                res = focal.focal(x, kern, stat, True).compute()
                self.assertTrue(np.allclose(truth, res, equal_nan=True))


def asm(x):
    c = {}
    xn = x[~np.isnan(x)]
    for v in xn:
        if v in c:
            c[v] += 1
        else:
            c[v] = 1
    n = sum(c.values())
    if not n:
        return np.nan
    p = np.array([cnt / n for cnt in c.values()])
    return np.nansum(p * p)


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
        return np.nan
    p = np.array([cnt / n for cnt in c.values()])
    return np.nansum(-p * np.log(p))


def mode(x):
    if x[~np.isnan(x)].size == 0:
        return np.nan
    m = stats.mode(x, axis=None, nan_policy="omit")
    return m.mode[0]


def unique(x):
    u = np.unique(x[~np.isnan(x)])
    if len(u):
        return len(u)
    return np.nan
