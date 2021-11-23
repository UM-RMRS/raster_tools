import unittest
import warnings
from functools import partial

import dask
import numpy as np
from scipy import ndimage, stats

from raster_tools import Raster, focal


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
            dask.is_dask_collection(focal._focal(x, kern, "max", True))
        )
        self.assertTrue(
            dask.is_dask_collection(focal._focal(x, kern, "max", False))
        )
        xd = dask.array.from_array(x)
        self.assertTrue(
            dask.is_dask_collection(focal._focal(xd, kern, "max", True))
        )
        self.assertTrue(
            dask.is_dask_collection(focal._focal(xd, kern, "max", False))
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
                res = focal._focal(x, kern, stat, True).compute()
                self.assertTrue(np.allclose(truth, res, equal_nan=True))


class TestCorrelate(unittest.TestCase):
    def test_correlate_return_dask(self):
        x = np.arange(16.0).reshape(4, 4)
        kern = focal.get_focal_window(2)
        self.assertTrue(dask.is_dask_collection(focal._correlate(x, kern)))

    def test_focal_correlate(self):
        for kern in [np.ones((5, 5)), np.ones((4, 4))]:
            func = partial(correlate, kern=kern)
            for nan_aware in [False, True]:
                data = np.arange(64.0).reshape(8, 8)
                if nan_aware:
                    data[:3, :3] = np.nan
                for mode in ["reflect", "nearest", "wrap", "constant"]:
                    origin = [-1 if d % 2 == 0 else 0 for d in kern.shape]
                    truth = ndimage.generic_filter(
                        data, func, size=kern.shape, mode=mode, origin=origin
                    )
                    test = focal._correlate(
                        data, kern, mode=mode, nan_aware=nan_aware
                    ).compute()
                    self.assertTrue(
                        np.allclose(truth, test, equal_nan=nan_aware)
                    )


class TestFocalIntegration(unittest.TestCase):
    def test_focal_integration(self):
        rs = Raster("test/data/multiband_small.tif")
        rsnp = rs._rs.values
        truth = rsnp.astype(float)
        for bnd in range(truth.shape[0]):
            truth[bnd] = ndimage.generic_filter(
                truth[bnd], np.nanmean, size=3, mode="constant", cval=np.nan
            )
        res = focal.focal(rs, "mean", 3, 3).eval()._rs.values
        self.assertTrue(np.allclose(truth, res, equal_nan=True))
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
        res = focal.focal(rs, "median", 3).eval()._rs.values
        self.assertTrue(np.allclose(truth, res, equal_nan=True))

    def test_focal_integration_raster_input(self):
        rs = Raster("test/data/multiband_small.tif")
        rsnp = rs._rs.values
        with self.assertRaises(TypeError):
            focal.focal(rsnp, "median", 3)
        res = focal.focal(rs, "mean", 1).eval()._rs.values
        self.assertTrue(np.allclose(rsnp, res, equal_nan=True))

    def test_focal_output_type(self):
        rs = Raster("test/data/multiband_small.tif") * 100
        rs = rs.set_null_value(-1)
        rs = rs.astype(int)

        self.assertTrue(rs._masked)
        self.assertTrue(rs.dtype.kind == "i")
        res = focal.focal(rs, "mode", 3).eval()
        self.assertTrue(res.dtype == rs.dtype)

        res = focal.focal(rs, "mean", 3).eval()
        self.assertTrue(res.dtype.kind == "f")


class TestCorrelateConvolveIntegration(unittest.TestCase):
    def test_correlate_integration(self):
        rs = Raster("test/data/multiband_small.tif").astype(float)
        rsnp = rs._rs.values
        truth = rsnp.astype(float)
        kernel = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]).astype(float)
        for bnd in range(truth.shape[0]):
            truth[bnd] = ndimage.generic_filter(
                truth[bnd], np.sum, footprint=kernel, mode="constant"
            )
        res = focal.correlate(rs, kernel).eval()._rs.values
        self.assertTrue(np.allclose(truth, res, equal_nan=False))

        truth = rsnp.astype(float)
        truth[:, :3, :3] = np.nan
        kern = focal.get_focal_window(3)
        for bnd in range(truth.shape[0]):
            truth[bnd] = ndimage.generic_filter(
                truth[bnd],
                np.nansum,
                footprint=kern,
                mode="constant",
            )
        rs._rs.data[:, :3, :3] = -1
        rs = rs.set_null_value(-1)
        res = focal.correlate(rs, kern).eval()._rs.values
        self.assertTrue(np.allclose(truth, res, equal_nan=True))

    def test_convolve_integration(self):
        rs = Raster("test/data/multiband_small.tif").astype(float)
        rsnp = rs._rs.values
        truth = rsnp.astype(float)
        kernel = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]).astype(float)
        for bnd in range(truth.shape[0]):
            truth[bnd] = ndimage.generic_filter(
                truth[bnd],
                np.sum,
                footprint=kernel[::-1, ::-1],
                mode="constant",
            )
        res = focal.convolve(rs, kernel).eval()._rs.values
        self.assertTrue(np.allclose(truth, res, equal_nan=False))

        truth = rsnp.astype(float)
        truth[:, :3, :3] = np.nan
        for bnd in range(truth.shape[0]):
            truth[bnd] = ndimage.generic_filter(
                truth[bnd],
                np.nansum,
                footprint=kernel[::-1, ::-1],
                mode="constant",
            )
        rs._rs.data[:, :3, :3] = -1
        rs = rs.set_null_value(-1)
        res = focal.convolve(rs, kernel).eval()._rs.values
        self.assertTrue(np.allclose(truth, res, equal_nan=True))

    def test_correlate_integration_raster_input(self):
        rs = Raster("test/data/multiband_small.tif")
        rsnp = rs._rs.values
        with self.assertRaises(TypeError):
            focal.correlate(rsnp, 3)
        res = focal.correlate(rs, np.ones((1, 1))).eval()._rs.values
        self.assertTrue(np.allclose(rsnp, res, equal_nan=True))

    def test_correlate_output_type(self):
        rs = Raster("test/data/multiband_small.tif") * 100
        rs = rs.set_null_value(-1)
        rs = rs.astype(int)

        self.assertTrue(rs._masked)
        self.assertTrue(rs.dtype.kind == "i")
        res = focal.correlate(rs, np.ones((3, 3), dtype=int)).eval()
        self.assertTrue(res.dtype == rs.dtype)

        res = focal.correlate(rs, np.ones((3, 3), dtype=float)).eval()
        self.assertTrue(res.dtype.kind == "f")


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


def correlate(x, kern):
    return np.nansum(x * kern.ravel())
