# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools  # noqa: F401

# isort: on

from unittest import TestCase

import dask
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from raster_tools import Raster, band_concat, open_vectors
from raster_tools.zonal import (
    _ZONAL_STAT_FUNCS,
    ZONAL_STAT_FUNCS,
    extract_points_eager,
    zonal_stats,
)
from tests.test_focal import asm, entropy, mode, unique
from tests.utils import arange_raster


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
        self.dem = Raster("tests/data/raster/elevation.tif")
        self.dem_np = np.array(self.dem)
        self.vc = open_vectors("tests/data/vector/pods_first_10.shp")
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


def get_random_points(n, nparts, dem):
    xmin, ymin, xmax, ymax = dem.bounds
    xspan = xmax - xmin
    yspan = ymax - ymin
    x = xmin + (xspan * 1.1 * np.random.random(n)) - (0.05 * xspan)
    y = ymin + (yspan * 1.1 * np.random.random(n)) - (0.05 * yspan)
    df = gpd.GeoSeries.from_xy(x, y, crs=dem.crs).to_frame("geometry")
    return dgpd.from_geopandas(df, npartitions=nparts)


def dem_clipped_small():
    return Raster("tests/data/raster/elevation_clipped_small.tif")


@pytest.mark.parametrize(
    "dem,n,nparts,name",
    [
        (dem_clipped_small(), 1_000, 1, None),
        (dem_clipped_small(), 1_000, 4, "values"),
        (dem_clipped_small().chunk((1, 10, 10)), 1_000, 4, "points"),
        (
            band_concat(
                [
                    dem_clipped_small(),
                    dem_clipped_small() + 1,
                    dem_clipped_small() + 10,
                ]
            ).chunk((1, 10, 10)),
            1_000,
            4,
            None,
        ),
        (arange_raster((700, 6, 6)), 10, 1, None),
        (arange_raster((700, 6, 6)).set_crs("EPSG:3857"), 10, 1, None),
    ],
)
def test_extract_points_eager(dem, n, nparts, name):
    if name is None:
        name = "extracted"
    df = get_random_points(n, nparts, dem)
    points = df.geometry.compute().to_list()
    x, y = dask.compute(df.geometry.x, df.geometry.y)
    r, c = dem.index(x, y)
    data = dem.values
    mask = dem.mask.compute()
    nb, nr, nc = data.shape
    valid = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
    n = len(x)
    bbox = shapely.geometry.box(*dem.bounds)
    point_check = []
    for p in points:
        point_check.append(bbox.contains(p))
    assert np.allclose(valid, point_check)
    dfs = []
    for bnd in range(nb):
        d = {
            "band": np.full(n, bnd + 1, dtype=np.min_scalar_type(bnd + 1)),
            name: np.full(n, np.nan),
        }
        extracted = data[bnd, r[valid], c[valid]].astype(float)
        masked = mask[bnd, r[valid], c[valid]]
        extracted[masked] = np.nan
        d[name][valid] = extracted
        dfs.append(
            pd.DataFrame(d, index=pd.RangeIndex(n * bnd, n * (bnd + 1)))
        )
    truth = pd.concat(dfs)
    result = extract_points_eager(df, dem, name)
    resultc = result.compute()
    assert isinstance(result, dask.dataframe.DataFrame)
    assert isinstance(resultc, pd.DataFrame)
    assert result.band.dtype == np.min_scalar_type(nb)
    assert result.known_divisions
    assert result.npartitions == nb
    assert all(result.columns == ["band", name])
    assert truth.equals(resultc)
