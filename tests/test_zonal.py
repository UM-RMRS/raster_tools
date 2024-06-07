# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts  # noqa: F401

# isort: on

import dask
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy
import shapely

from raster_tools import band_concat
from raster_tools.zonal import (
    _asm_agg,
    _entropy_agg,
    _mode_agg,
    _nunique_agg,
    _raster_to_series,
    extract_points_eager,
    zonal_stats,
)
from tests import testdata
from tests.utils import arange_raster


def asm(x):
    return rts.stat_common.nanasm_jit(x.to_numpy())


def entropy(x):
    return rts.stat_common.nanentropy_jit(x.to_numpy())


def mode(x):
    m = scipy.stats.mode(x.to_numpy()).mode
    if np.isscalar(m):
        return m
    return m[0]


def nunique(x):
    return rts.stat_common.nan_unique_count_jit(x.to_numpy())


group_df1 = pd.DataFrame(
    {"zone": np.arange(5).repeat(4), "band_1": np.arange(20).astype("float64")}
)
group_df2 = pd.DataFrame(
    {
        "zone": [8, 0, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 6, 7, 7, 7, 7, 7, 8, 8],
        "band_1": np.arange(20).astype("float64"),
    }
)
group_df3 = pd.DataFrame(
    {"zone": np.arange(5).repeat(4), "band_1": np.ones(20).astype("float64")}
)


@pytest.mark.parametrize(
    "stat,stat_truth",
    [
        (_asm_agg, asm),
        (_entropy_agg, entropy),
        (_mode_agg, mode),
        (_nunique_agg, nunique),
    ],
)
@pytest.mark.parametrize(
    "frame",
    [
        dd.from_pandas(group_df1, npartitions=1),
        dd.from_pandas(group_df1, npartitions=3),
        dd.from_pandas(group_df1, npartitions=10),
        dd.from_pandas(group_df2, npartitions=3),
        dd.from_pandas(group_df2, npartitions=7),
        dd.from_pandas(group_df3, npartitions=7),
    ],
)
def test_custom_stats(stat, stat_truth, frame):
    gdfc = frame.compute().groupby("zone")
    gdf = frame.groupby("zone")
    truth = gdfc.agg([stat_truth]).sort_index()
    result = gdf.agg([stat]).compute().sort_index()

    assert truth.equals(result)


def rasters_to_zonal_df(feat_raster, data_raster):
    feat_raster = feat_raster.chunk(data_raster.data.chunks)
    dfs = [_raster_to_series(feat_raster).rename("zone").compute()]
    for b in range(1, data_raster.nbands + 1):
        nv = data_raster.null_value
        band = (
            _raster_to_series(data_raster.get_bands(b))
            .rename(f"band_{b}")
            .compute()
        )
        if nv is not None:
            band = band.replace(data_raster.null_value, np.nan)
        # Cast up to avoid floating point issues in sums
        band = band.astype("float64")
        dfs.append(band)
    return pd.concat(dfs, axis=1)


@pytest.mark.parametrize(
    "stats",
    [
        ["max", "mean", "median", "min", "size", "std", "sum", "var"],
        ["asm", "entropy", "mode", "nunique", "mean", "var", "count"],
        ["asm", "median"],
        "median",
        ["mean", "median"],
    ],
)
@pytest.mark.parametrize(
    "features,raster",
    [
        # No null data in data raster
        (
            testdata.vector.pods_small.data.repartition(npartitions=3),
            testdata.raster.dem_small.chunk((1, 20, 20)),
        ),
        # Some null data in data raster outside of features
        (
            testdata.vector.pods_small,
            rts.clipping.clip(
                testdata.vector.pods_small.buffer(100),
                testdata.raster.dem_small.chunk((1, 20, 20)),
            ),
        ),
        # All null data in data raster outside of features and some inside
        (
            testdata.vector.pods_small,
            rts.clipping.clip(
                testdata.vector.pods_small.data.compute()
                .dissolve()
                .buffer(-100),
                testdata.raster.dem_small.chunk((1, 20, 20)),
            ),
        ),
        # Raster features instead of vector
        (
            testdata.vector.pods_small.to_raster(testdata.raster.dem_small),
            testdata.raster.dem_small.chunk((1, 20, 20)),
        ),
        # Raster features instead of vector and mismatched chunksize
        (
            testdata.vector.pods_small.to_raster(
                testdata.raster.dem_small.chunk((1, 25, 25))
            ),
            testdata.raster.dem_small.chunk((1, 20, 20)),
        ),
        # No null value set
        (
            testdata.vector.pods_small,
            testdata.raster.dem_small.set_null_value(None).chunk((1, 20, 20)),
        ),
    ],
)
def test_zonal_stats(features, raster, stats):
    stats_working = [stats] if isinstance(stats, str) else stats.copy()
    for i, s in enumerate(stats_working):
        if s == "asm":
            stats_working[i] = asm
        elif s == "entropy":
            stats_working[i] = entropy
        elif s == "mode":
            stats_working[i] = mode
        elif s == "nunique":
            stats_working[i] = nunique

    if not isinstance(features, rts.Raster):
        feat_raster = rts.rasterize.rasterize(features, raster)
    else:
        feat_raster = features
    tdf = rasters_to_zonal_df(feat_raster, raster)
    # Trim off areas outside features
    tdf = tdf[tdf.zone != feat_raster.null_value]
    truth_df = tdf.groupby("zone").agg(stats_working).sort_index()

    result = zonal_stats(features, raster, stats)
    assert isinstance(result, dd.DataFrame)
    assert result.columns.equals(truth_df.columns)
    assert result.dtypes.equals(truth_df.dtypes)
    resultc = result.compute().sort_index()
    # Use this because .equals() is too sensitive
    assert (truth_df - resultc).abs().max().max() < 1e-8


@pytest.mark.parametrize(
    "raster",
    [
        testdata.raster.dem_small,
        rts.band_concat(
            [
                testdata.raster.dem_small,
                testdata.raster.dem_small,
                testdata.raster.dem_small,
            ]
        ).chunk((1, 20, 20)),
    ],
)
def test_zonal_stats_long_format(raster):
    zdf = zonal_stats(
        testdata.vector.pods_small, raster, ["mean", "min", "max"]
    )
    truth = (
        zdf.compute()  # noqa: PD013  warning is wrong. .melt cannot do this
        .stack(0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "band"})
        .sort_values(["band", "zone"])
        .sort_index()
    )
    lookup = {f"band_{i + 1}": i + 1 for i in range(raster.nbands)}
    truth["band"] = truth.band.apply(lambda x: lookup[x])
    zdf = zonal_stats(
        testdata.vector.pods_small,
        raster,
        ["mean", "min", "max"],
        wide_format=False,
    )

    assert truth.columns.equals(zdf.columns)
    zdf = zdf.compute().sort_index()
    assert truth.equals(zdf)


def test_zonal_stats_handle_overlap():
    # 4 rectangles that overlap to form 4x4 square
    features = gpd.GeoSeries(
        [
            shapely.geometry.box(0, 2, 4, 4),
            shapely.geometry.box(0, 0, 2, 4),
            shapely.geometry.box(0, 0, 4, 2),
            shapely.geometry.box(2, 0, 4, 4),
        ],
        crs="EPSG:3857",
    )
    data_raster = arange_raster((4, 4)).set_crs("EPSG:3857")
    # Confirm that first zone is dropped due to overlap and second &
    # third zones are halved in area.
    expected = pd.DataFrame(
        {("band_1", "mean"): [2.5, 10.5, 8.5]},
        index=pd.RangeIndex(2, 5, name="zone"),
    )
    result = zonal_stats(features, data_raster, "mean").compute()
    assert result.equals(expected)

    expected = pd.DataFrame(
        {("band_1", "mean"): [3.5, 6.5, 11.5, 8.5]},
        index=pd.RangeIndex(1, 5, name="zone"),
    )
    for feats in [
        features,
        features.to_crs("EPSG:4326"),
        dgpd.from_geopandas(features, npartitions=1),
        dgpd.from_geopandas(features, npartitions=4),
        dgpd.from_geopandas(features, npartitions=4).to_crs("EPSG:4326"),
        rts.Vector(features),
    ]:
        result = zonal_stats(feats, data_raster, "mean", handle_overlap=True)
        assert isinstance(result, dd.DataFrame)
        assert result.npartitions == 1
        result = result.compute()
        assert result.equals(expected)


def get_random_points(n, nparts, dem):
    xmin, ymin, xmax, ymax = dem.bounds
    xspan = xmax - xmin
    yspan = ymax - ymin
    x = xmin + (xspan * 1.1 * np.random.random(n)) - (0.05 * xspan)
    y = ymin + (yspan * 1.1 * np.random.random(n)) - (0.05 * yspan)
    points = gpd.GeoSeries.from_xy(x, y, crs=dem.crs).to_frame("geometry")
    return dgpd.from_geopandas(points, npartitions=nparts)


def dem_clipped_small():
    return testdata.raster.dem_clipped_small


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
    points_df = get_random_points(n, nparts, dem)
    points = points_df.geometry.compute().to_list()
    x, y = dask.compute(points_df.geometry.x, points_df.geometry.y)
    r, c = dem.index(x, y)
    data = dem.to_numpy()
    mask = dem.mask.compute()
    nb, nr, nc = data.shape
    valid = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
    n = len(x)
    bbox = shapely.geometry.box(*dem.bounds)
    point_check = [bbox.contains(p) for p in points]
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
    result = extract_points_eager(points_df, dem, name)
    resultc = result.compute()
    assert isinstance(result, dask.dataframe.DataFrame)
    assert isinstance(resultc, pd.DataFrame)
    assert result.band.dtype == np.min_scalar_type(nb)
    assert result.known_divisions
    assert result.npartitions == nb
    assert all(result.columns == ["band", name])
    assert truth.equals(resultc)
