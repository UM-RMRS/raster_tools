import sys
import warnings
from unittest import TestCase

import numpy as np
from scipy import stats

import raster_tools.stat_common as stc
from raster_tools.dtypes import is_scalar

PY_VER_37 = sys.version_info[0] == 3 and sys.version_info[1] == 7

INPUTS = [
    0,
    4.9,
    -1.0,
    np.array([]),
    np.array([0]),
    np.array([0, -1, 9]),
    np.arange(100, dtype=float),
    np.arange(600, dtype=float),
    np.arange(33_000, 0, -1, dtype=float),
    np.arange(66_000, 0, -1, dtype=float),
    np.array([np.nan, np.nan, np.nan]),
    np.array([np.nan, 0.0, np.nan]),
    np.array([np.nan, 1.0, 500.0]),
    np.array([np.nan, 1.0, 500.0]),
]


class TestJitStats(TestCase):
    def _test(self, func, arg, truth):
        res = func(arg)
        self.assertTrue(is_scalar(res))
        self.assertTrue(np.allclose(res, truth, equal_nan=True))

    def _test_func(self, func, truth_func):
        truth = []
        with warnings.catch_warnings():
            # Ignore nan related warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in INPUTS:
                try:
                    truth.append(truth_func(i))
                except ValueError:
                    truth.append(None)
        for i, t in zip(INPUTS, truth):
            if t is not None:
                self._test(func, i, t)
            else:
                with self.assertRaises(ValueError):
                    self._test(func, i, t)

    def test_nanmin_jit(self):
        self._test_func(stc.nanmin_jit, np.nanmin)

    def test_nanmax_jit(self):
        self._test_func(stc.nanmax_jit, np.nanmax)

    def test_nanmean_jit(self):
        self._test_func(stc.nanmean_jit, np.nanmean)

    def test_nanmedian_jit(self):
        self._test_func(stc.nanmedian_jit, np.nanmedian)

    def test_nansum_jit(self):
        self._test_func(stc.nansum_jit, np.nansum)

    def test_nanvar_jit(self):
        self._test_func(stc.nanvar_jit, np.nanvar)

    def test_nanstd_jit(self):
        self._test_func(stc.nanstd_jit, np.nanstd)

    def test_nan_unique_count_jit(self):
        def unique(x):
            uniq = np.unique(x)
            nans = np.isnan(uniq)
            uniq = uniq[~nans]
            return len(uniq)

        self._test_func(stc.nan_unique_count_jit, unique)

    def test_nanmode_jit(self):
        def mode(x):
            x = np.atleast_1d(x)
            x = x[~np.isnan(x)]
            if x.size == 0:
                return np.nan
            kwargs = {"nan_policy": "omit"}
            if not PY_VER_37:
                kwargs["keepdims"] = True
            m = stats.mode(x, **kwargs)
            return m[0][0]

        self._test_func(stc.nanmode_jit, mode)

    def test_nanentropy_jit(self):
        def entropy(x):
            x = np.atleast_1d(x)
            x = x[~np.isnan(x)]
            ent = 0.0
            _, cnts = np.unique(x, return_counts=True)
            n = len(cnts)
            if n:
                for c in cnts:
                    p = c / n
                    ent -= p * np.log(p)
            return ent

        self._test_func(stc.nanentropy_jit, entropy)

    def test_nanasm_jit(self):
        def asm(x):
            x = np.atleast_1d(x)
            x = x[~np.isnan(x)]
            ent = 0.0
            _, cnts = np.unique(x, return_counts=True)
            n = len(cnts)
            if n:
                for c in cnts:
                    p = c / n
                    ent += p * p
            return ent

        self._test_func(stc.nanasm_jit, asm)

    def test_nanargmin_jit(self):
        def argmin(x):
            x = np.atleast_1d(x)
            if x.size == 0:
                return -1
            if np.isnan(x).all():
                return -2
            return np.nanargmin(x)

        self._test_func(stc.nanargmin_jit, argmin)

    def test_nanargmax_jit(self):
        def argmax(x):
            x = np.atleast_1d(x)
            if x.size == 0:
                return -1
            if np.isnan(x).all():
                return -2
            return np.nanargmax(x)

        self._test_func(stc.nanargmax_jit, argmax)
