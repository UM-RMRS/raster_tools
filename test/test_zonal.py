from unittest import TestCase

import dask
import numpy as np

from raster_tools import Raster, band_concat, open_vectors
from raster_tools.zonal import _ZONAL_STAT_FUNCS, ZONAL_STAT_FUNCS, zonal_stats
from test.test_focal import asm, entropy, mode, unique


def _apply_stat(dem, vrs, func):
    rsnp = np.array(vrs)
    mask = rsnp > 0
    return func(dem[mask])


def _apply_stat_all(dem, vrss, func):
    return [_apply_stat(dem, vrs, func) for vrs in vrss]


def count(x):
    return np.count_nonzero(~np.isnan(x))


all_stats = {
    "asm": asm,
    "count": count,
    "entropy": entropy,
    "max": np.nanmax,
    "mean": np.nanmean,
    "median": np.nanmedian,
    "min": np.nanmin,
    "mode": mode,
    "std": np.nanstd,
    "sum": np.nansum,
    "unique": unique,
    "var": np.nanvar,
}


class TestZonalStats(TestCase):
    def setUp(self):
        self.dem = Raster("test/data/elevation.tif")
        self.dem_np = np.array(self.dem)
        self.vc = open_vectors("test/data/vector/pods_first_10.shp")
        self.vc_rasters = [v.to_raster(self.dem).eval() for v in self.vc]

    def test_stat_func_set(self):
        # Make sure public set of funcs matches private look up table
        self.assertTrue(ZONAL_STAT_FUNCS == set(_ZONAL_STAT_FUNCS))

    def test_zonal_stats(self):
        res = zonal_stats(self.vc, self.dem, "mode")
        resc = res.compute()
        # Check that result is lazy
        self.assertTrue(dask.is_dask_collection(res))

        mode_truth = _apply_stat_all(self.dem_np, self.vc_rasters, mode)
        self.assertTrue(all(resc.columns == ["band", "mode"]))
        self.assertTrue(len(resc["mode"]) == 10)
        self.assertTrue(all(resc.index == list(range(10))))
        self.assertTrue(np.allclose(resc["mode"], mode_truth))

        stats = {"std": np.nanstd, "count": count, "asm": asm}
        resc = zonal_stats(self.vc, self.dem, list(stats)).compute()

        truth = {
            k: _apply_stat_all(self.dem_np, self.vc_rasters, func)
            for k, func in stats.items()
        }
        self.assertTrue(all(resc.columns == ["band", *stats]))
        for k in stats:
            self.assertTrue(np.allclose(resc[k], truth[k], 1e5))

    def test_lazy(self):
        res = zonal_stats(self.vc.to_lazy(), self.dem, "mode")
        # Check that result is lazy
        self.assertTrue(dask.is_dask_collection(res))
        mode_truth = _apply_stat_all(self.dem_np, self.vc_rasters, mode)
        self.assertTrue(np.allclose(res["mode"], mode_truth))

    def test_all_stats(self):
        self.assertTrue(set(all_stats) == set(ZONAL_STAT_FUNCS))
        res = zonal_stats(self.vc, self.dem, list(all_stats)).compute()

        truth = {
            k: _apply_stat_all(self.dem_np, self.vc_rasters, func)
            for k, func in all_stats.items()
        }
        self.assertTrue(all(res.columns == ["band", *all_stats]))
        for k in all_stats:
            self.assertTrue(np.allclose(res[k], truth[k], 1e5))

    def test_multiband_all_stats(self):
        dem = band_concat((self.dem, self.dem * 105))
        dem_np = np.array(dem)
        res = zonal_stats(self.vc, dem, list(all_stats)).compute()

        truth = {
            k: _apply_stat_all(dem_np[0:1], self.vc_rasters, func)
            for k, func in all_stats.items()
        }
        for k in truth:
            truth[k] = np.concatenate(
                [
                    truth[k],
                    _apply_stat_all(dem_np[1:], self.vc_rasters, all_stats[k]),
                ]
            )
        self.assertTrue(all(res.columns == ["band", *all_stats]))
        band = np.ones(2 * len(self.vc))
        band[len(self.vc) :] = 2
        self.assertTrue(len(res) == 2 * len(self.vc))
        self.assertTrue(np.allclose(res.band, band))
        for k in all_stats:
            self.assertTrue(np.allclose(res[k], truth[k], 1e5))
