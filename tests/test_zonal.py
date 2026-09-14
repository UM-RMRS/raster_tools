# isort: off
# TODO(pygeos): remove this once shapely is the default backend for geopandas.
# Force raster_tools._compat to be loaded before geopandas when running tests
import raster_tools as rts  # noqa: F401

# isort: on

import threading

import dask
import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy
import shapely

from raster_tools import stack_bands
from raster_tools.dtypes import F64, I64
from raster_tools.zonal import (
    _MAX_DIRECT_ZONE_ID,
    _block_partials,
    _merge_moments,
    _raster_to_series,
    _zone_codes,
    extract_points_eager,
    zonal_stats,
)
from tests import testdata
from tests.utils import make_raster


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


def test_raster_to_series_partitions_and_order():
    raster = testdata.raster.dem_small.chunk((1, 20, 20))
    data = raster.data
    series = _raster_to_series(raster)

    assert series.npartitions == int(np.prod(data.numblocks))

    nbands, ny, nx = data.numblocks
    # Partitioning guard: expected is rebuilt with the same block-iteration
    # order as _raster_to_series, so this pins the partition count and the
    # band-major, x, y-innermost block ordering. It is not an independent
    # correctness check (that is test_zonal_stats_matches_numpy_oracle).
    expected = np.concatenate(
        [
            data.blocks[b, i, j].compute().reshape(-1)
            for b in range(nbands)
            for j in range(nx)
            for i in range(ny)
        ]
    )
    np.testing.assert_array_equal(series.compute().to_numpy(), expected)


def test_raster_to_series_no_strip_rechunk():
    # Lazy (no compute) guard against the old to_dask_dataframe path, which
    # rechunked a 2D-chunked array into full-height column strips before
    # ravelling. The block-aligned path keeps one lazy partition per array
    # block, so the underlying dask array has one chunk per block. A strip
    # rechunk would instead collapse each column of blocks into a single chunk.
    raster = testdata.raster.dem_small.chunk((1, 20, 20))
    data = raster.data
    series = _raster_to_series(raster)
    nblocks = int(np.prod(data.numblocks))
    assert series.npartitions == nblocks
    assert len(series.to_dask_array().chunks[0]) == nblocks


# A shuffled stat order covering all fourteen stat functions, so the tests
# exercise column ordering, the moment path, the value-count path, and the
# median join together.
SHUFFLED_STATS = [
    "prod",
    "median",
    "mode",
    "std",
    "count",
    "asm",
    "size",
    "sum",
    "var",
    "nunique",
    "min",
    "entropy",
    "max",
    "mean",
]


def _oracle_stat(s, v, valid):
    ne = valid.size > 0
    if s == "size":
        return len(v)
    if s == "count":
        return valid.size
    if s == "sum":
        return valid.sum() if ne else 0.0
    if s == "prod":
        return valid.prod() if ne else 1.0
    if s == "min":
        return valid.min() if ne else np.nan
    if s == "max":
        return valid.max() if ne else np.nan
    if s == "mean":
        return valid.mean() if ne else np.nan
    if s == "median":
        return np.median(valid) if ne else np.nan
    if s == "var":
        return np.var(valid, ddof=1) if valid.size > 1 else np.nan
    if s == "std":
        return np.std(valid, ddof=1) if valid.size > 1 else np.nan
    if s == "mode":
        return np.ravel(scipy.stats.mode(valid).mode)[0] if ne else np.nan
    if s == "asm":
        return rts.stat_common.nanasm_jit(valid) if ne else np.nan
    if s == "entropy":
        return rts.stat_common.nanentropy_jit(valid) if ne else np.nan
    if s == "nunique":
        return rts.stat_common.nan_unique_count_jit(valid)
    raise ValueError(f"Unknown stat: {s!r}")


def numpy_zonal_oracle(feat_raster, data_raster, stats):
    # Independent reference: loop the zones with numpy. Zone-null cells are
    # dropped, data nulls become NaN, and each stat is computed on the valid
    # (non-NaN) values of the zone. Returns a wide (band_b, stat) frame with
    # int64 for the count-like stats and float64 for the rest.
    zones = feat_raster.to_numpy().ravel()
    znull = feat_raster.null_value
    if znull is not None:
        keepz = zones != znull
    else:
        keepz = np.ones(zones.shape, dtype=bool)
    zone_ids = np.unique(zones[keepz])
    nv = data_raster.null_value
    cols = {}
    for b in range(1, data_raster.nbands + 1):
        vals = data_raster.get_bands(b).to_numpy().ravel().astype(np.float64)
        if nv is not None and not np.isnan(nv):
            vals = np.where(vals == nv, np.nan, vals)
        per = {s: [] for s in stats}
        for z in zone_ids:
            v = vals[keepz & (zones == z)]
            valid = v[~np.isnan(v)]
            for s in stats:
                per[s].append(_oracle_stat(s, v, valid))
        for s in stats:
            cols[(f"band_{b}", s)] = np.array(per[s], dtype=np.float64)
    out = pd.DataFrame(cols, index=pd.Index(zone_ids, name="zone"))
    out.columns = pd.MultiIndex.from_tuples(list(cols))
    for key in list(out.columns):
        if key[1] in ("count", "size", "nunique"):
            out[key] = out[key].astype(np.int64)
    return out


def assert_zonal_matches_oracle(features, data, stats):
    if isinstance(features, rts.Raster):
        feat_raster = features
    else:
        feat_raster = rts.rasterize.rasterize(features, data)
    oracle = numpy_zonal_oracle(feat_raster, data, stats).sort_index()

    result = zonal_stats(features, data, stats)
    # Lazy metadata must line up before computing.
    assert result.columns.equals(oracle.columns)
    assert result.dtypes.equals(oracle.dtypes)

    resultc = result.compute().sort_index()
    assert resultc.columns.equals(oracle.columns)
    assert resultc.dtypes.equals(oracle.dtypes)
    assert resultc.index.name == "zone"
    assert resultc.index.dtype == oracle.index.dtype
    np.testing.assert_array_equal(
        resultc.index.to_numpy(), oracle.index.to_numpy()
    )
    # NaN pattern must be identical; the value comparison below is NaN-blind.
    assert (resultc.isna() == oracle.isna()).all().all()
    for c in oracle.columns:
        np.testing.assert_allclose(
            resultc[c].to_numpy(dtype=np.float64),
            oracle[c].to_numpy(dtype=np.float64),
            rtol=1e-9,
            atol=1e-8,
        )


def _case_single_band():
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    return testdata.vector.pods_small, dem


def _case_three_band():
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    clipped = rts.clipping.clip(
        testdata.vector.pods_small.data.compute().dissolve().buffer(-100),
        dem,
    )
    # The third band disagrees with the others on which cells are null.
    data = stack_bands([dem, dem * 2 + 1, clipped]).chunk((1, 20, 20))
    return testdata.vector.pods_small, data


def _case_all_null_block():
    # A data raster with one full 20x20 block forced to null, so some zones
    # have blocks with size > 0 but count == 0 that the merge must combine.
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    feat = rts.rasterize.rasterize(testdata.vector.pods_small, dem)
    feat_r = (
        rts.Raster(feat.to_numpy())
        .set_null_value(feat.null_value)
        .chunk((1, 20, 20))
    )
    data_arr = dem.to_numpy().astype(np.float64)
    data_arr[:, 0:20, 0:20] = -9999.0
    data_r = rts.Raster(data_arr).set_null_value(-9999.0).chunk((1, 20, 20))
    return feat_r, data_r


def _tiny_probe_rasters(zone_null=0):
    z = np.array(
        [[1, 1, 1, 2], [2, 2, 3, 3], [3, 3, 0, 0], [0, 1, 1, 2]],
        dtype="uint8",
    )
    d = np.array(
        [
            [1.0, 2.0, np.nan, np.nan],
            [np.nan, np.nan, 5.0, 5.0],
            [6.0, 7.0, 1.0, 1.0],
            [1.0, 2.0, 2.0, np.nan],
        ],
        dtype="float32",
    )
    feat = rts.Raster(np.expand_dims(z, 0)).set_null_value(zone_null)
    data = rts.Raster(np.expand_dims(d, 0)).set_null_value(-999.0)
    return feat, data


def _case_tiny():
    return _tiny_probe_rasters(zone_null=0)


def _case_no_zone_null():
    feat, data = _tiny_probe_rasters(zone_null=None)
    return feat, data


@pytest.mark.parametrize(
    "build_case",
    [
        _case_single_band,
        _case_three_band,
        _case_all_null_block,
        _case_tiny,
        _case_no_zone_null,
    ],
    ids=[
        "single_band",
        "three_band",
        "all_null_block",
        "tiny",
        "no_zone_null",
    ],
)
def test_zonal_stats_matches_numpy_oracle(build_case):
    features, data = build_case()
    assert_zonal_matches_oracle(features, data, SHUFFLED_STATS)


def test_zonal_stats_oracle_multiblock_merge(monkeypatch):
    # zonal_stats rechunks the data raster to the auto chunk size before the
    # fast path runs, so a small chunk size is needed to make the block-partial
    # merge fire. A 10KiB chunk splits dem_small into a 3x3 grid. The
    # median join at finer grids is covered by its own tests below.
    import raster_tools.zonal as zonal_mod

    seen = {}
    real = zonal_mod._block_zonal_stats

    def record_numblocks(features_raster, data_raster, stats):
        # numblocks is read at graph-build time, not during threaded compute.
        seen["numblocks"] = data_raster.data.numblocks
        return real(features_raster, data_raster, stats)

    monkeypatch.setattr(zonal_mod, "_block_zonal_stats", record_numblocks)

    with dask.config.set({"array.chunk-size": "10KiB"}):
        dem = testdata.raster.dem_small.chunk((1, 20, 20))
        assert_zonal_matches_oracle(
            testdata.vector.pods_small, dem, SHUFFLED_STATS
        )

    # The fast path saw a multi-block grid, so the merge was exercised.
    _, ny, nx = seen["numblocks"]
    assert ny * nx > 1


def test_zonal_stats_sparse_int64_ids():
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    feat = rts.rasterize.rasterize(testdata.vector.pods_small, dem)
    znull = feat.null_value
    # Spread the ids out and make the null id negative to force the np.unique
    # code path in _zone_codes.
    transformed = feat.astype(I64) * 1_000_003 + 7
    ids = rts.general.where(feat != znull, transformed, -5).set_null_value(-5)
    ids = ids.chunk((1, 20, 20))

    valid_zones = np.unique(feat.to_numpy()[feat.to_numpy() != znull])
    expected_index = np.sort(valid_zones.astype(np.int64) * 1_000_003 + 7)

    result = zonal_stats(ids, dem, SHUFFLED_STATS).compute().sort_index()
    assert result.index.dtype == np.int64
    np.testing.assert_array_equal(result.index.to_numpy(), expected_index)

    assert_zonal_matches_oracle(ids, dem, SHUFFLED_STATS)


def test_zonal_all_null_zone_values():
    # Zone 2 in the tiny probe has only null data cells.
    feat, data = _tiny_probe_rasters(zone_null=0)
    stats = ["count", "size", "sum", "prod", "nunique", "mean", "min", "std"]
    result = zonal_stats(feat, data, stats).compute().sort_index()
    row = result.loc[2, "band_1"]
    assert row["count"] == 0
    assert row["size"] == 4
    assert row["sum"] == 0.0
    assert row["prod"] == 1.0
    assert row["nunique"] == 0
    assert np.isnan(row["mean"])
    assert np.isnan(row["min"])
    assert np.isnan(row["std"])


def test_zonal_stats_duplicate_stats_raises():
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    with pytest.raises(ValueError, match="Duplicate stats function: 'mean'"):
        zonal_stats(testdata.vector.pods_small, dem, ["mean", "mean"])


def test_zonal_stats_multiband_mismatched_features_chunks():
    # A multi-band data raster ends up with band chunks after the internal
    # rechunk, so a single-band features raster with different spatial chunks
    # must be rechunked to match without inheriting those band chunks.
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    data = stack_bands([dem, dem * 2 + 1]).chunk((1, 20, 20))
    assert data.nbands == 2
    feat = rts.rasterize.rasterize(testdata.vector.pods_small, dem).chunk(
        (1, 25, 25)
    )
    assert feat.data.chunks[1:] != data.data.chunks[1:]
    assert_zonal_matches_oracle(feat, data, SHUFFLED_STATS)


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
        rts.stack_bands(
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


def test_zonal_float_feature_raster_error_message():
    features = make_raster("arange", dtype="float32", shape=(1, 6, 6))
    data_raster = make_raster("arange", dtype="float32", shape=(1, 6, 6))
    with pytest.raises(TypeError) as exc:
        zonal_stats(features, data_raster, "mean")
    msg = str(exc.value)
    assert "Feature raster must be an integer type" in msg
    assert "float32" in msg
    assert "GeoTIFF" in msg
    assert "astype" in msg


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
    data_raster = make_raster("arange", shape=(4, 4))
    # Confirm that first zone is dropped due to overlap and second &
    # third zones are halved in area.
    expected = pd.DataFrame(
        {("band_1", "mean"): [2.5, 10.5, 8.5]},
        index=pd.RangeIndex(2, 5, name="zone"),
    )
    # sort_index because the grouped output row order is not guaranteed; the
    # per-zone values are what matter here.
    result = zonal_stats(features, data_raster, "mean").compute().sort_index()
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
    rng = np.random.default_rng()
    x = xmin + (xspan * 1.1 * rng.random(n)) - (0.05 * xspan)
    y = ymin + (yspan * 1.1 * rng.random(n)) - (0.05 * yspan)
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
            stack_bands(
                [
                    dem_clipped_small(),
                    dem_clipped_small() + 1,
                    dem_clipped_small() + 10,
                ],
                null_value="default",
            ).chunk((1, 10, 10)),
            1_000,
            4,
            None,
        ),
        (make_raster("arange", shape=(700, 6, 6), crs=None), 10, 1, None),
        (make_raster("arange", shape=(700, 6, 6)), 10, 1, None),
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


LONG_STATS = ["mean", "median", "min", "count", "mode", "nunique"]


@pytest.mark.parametrize(
    "build_case",
    [_case_single_band, _case_three_band],
    ids=["single_band", "three_band"],
)
def test_zonal_stats_long_format_matches_oracle(build_case):
    features, data = build_case()
    if isinstance(features, rts.Raster):
        feat_raster = features
    else:
        feat_raster = rts.rasterize.rasterize(features, data)
    oracle = numpy_zonal_oracle(feat_raster, data, LONG_STATS)
    lookup = {f"band_{i + 1}": i + 1 for i in range(data.nbands)}
    expected = (
        oracle.stack(0, future_stack=True)  # noqa: PD013
        .reset_index()
        .rename(columns={"level_1": "band"})
    )
    expected["band"] = expected.band.apply(lambda x: lookup[x])
    expected = expected.sort_values(["band", "zone"]).reset_index(drop=True)

    result = zonal_stats(features, data, LONG_STATS, wide_format=False)
    resultc = (
        result.compute().sort_values(["band", "zone"]).reset_index(drop=True)
    )

    assert result.columns.equals(expected.columns)
    assert resultc.columns.equals(expected.columns)
    assert resultc.dtypes.equals(expected.dtypes)
    assert (resultc.isna() == expected.isna()).all().all()
    np.testing.assert_array_equal(
        resultc["zone"].to_numpy(), expected["zone"].to_numpy()
    )
    np.testing.assert_array_equal(
        resultc["band"].to_numpy(), expected["band"].to_numpy()
    )
    for s in LONG_STATS:
        np.testing.assert_allclose(
            resultc[s].to_numpy(dtype=np.float64),
            expected[s].to_numpy(dtype=np.float64),
            rtol=1e-9,
            atol=1e-8,
        )


MEDIAN_JOIN_STATS = [
    ["mean", "median"],
    ["median", "mean", "count"],
    ["mean", "median", "count"],
]


def _record_block_numblocks(monkeypatch):
    # Capture the data raster's block grid at graph-build time, the same way
    # test_zonal_stats_oracle_multiblock_merge observes it, so the median join
    # can be exercised above the fifteen-block split_out threshold.
    import raster_tools.zonal as zonal_mod

    seen = {}
    real = zonal_mod._block_zonal_stats

    def record_numblocks(features_raster, data_raster, stats):
        seen["numblocks"] = data_raster.data.numblocks
        return real(features_raster, data_raster, stats)

    monkeypatch.setattr(zonal_mod, "_block_zonal_stats", record_numblocks)
    return seen


@pytest.mark.parametrize("stats", MEDIAN_JOIN_STATS)
def test_zonal_median_join_many_blocks_wide(stats, monkeypatch):
    # A 3KiB chunk size splits dem_small into a 6x6 (36-block) grid. Above
    # fifteen blocks the median group-by emits split_out > 1, the case that
    # used to break the median join onto the block stats.
    seen = _record_block_numblocks(monkeypatch)
    # Rasterize once outside the chunk-size context; rasterizing the vector on
    # the finer auto chunking diverges from the oracle on a few boundary cells.
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    feat = rts.rasterize.rasterize(testdata.vector.pods_small, dem)
    with dask.config.set({"array.chunk-size": "3KiB"}):
        assert_zonal_matches_oracle(feat, dem, stats)
    _, ny, nx = seen["numblocks"]
    assert ny * nx > 15


@pytest.mark.parametrize("stats", MEDIAN_JOIN_STATS)
def test_zonal_median_join_many_blocks_long(stats, monkeypatch):
    seen = _record_block_numblocks(monkeypatch)
    dem = testdata.raster.dem_small.chunk((1, 20, 20))
    feat = rts.rasterize.rasterize(testdata.vector.pods_small, dem)
    oracle = numpy_zonal_oracle(feat, dem, stats)
    expected = (
        oracle.stack(0, future_stack=True)  # noqa: PD013
        .reset_index()
        .rename(columns={"level_1": "band"})
    )
    expected["band"] = expected.band.apply(lambda x: int(x.split("_")[-1]))
    expected = expected.sort_values(["band", "zone"]).reset_index(drop=True)
    with dask.config.set({"array.chunk-size": "3KiB"}):
        result = zonal_stats(feat, dem, stats, wide_format=False)
        resultc = (
            result.compute()
            .sort_values(["band", "zone"])
            .reset_index(drop=True)
        )
    assert result.columns.equals(expected.columns)
    assert resultc.dtypes.equals(expected.dtypes)
    np.testing.assert_array_equal(
        resultc["zone"].to_numpy(), expected["zone"].to_numpy()
    )
    for s in stats:
        np.testing.assert_allclose(
            resultc[s].to_numpy(dtype=np.float64),
            expected[s].to_numpy(dtype=np.float64),
            rtol=1e-9,
            atol=1e-8,
        )
    _, ny, nx = seen["numblocks"]
    assert ny * nx > 15


def test_merge_moments_matches_direct():
    rng = np.random.default_rng(0)
    # Three block partials over separate cell runs. Chunk A lacks zone 4;
    # chunk B has zone 3 present but with all-null data (count 0).
    za = np.array([1, 1, 2, 2, 3, 3, 3, 3, 0, 0], dtype="uint16")
    zb = np.array([1, 2, 3, 3, 4, 4, 4, 0, 1, 2], dtype="uint16")
    zc = np.array([4, 4, 3, 3, 2, 2, 1, 1, 0, 0], dtype="uint16")
    da_ = rng.random(10) + 1.0
    db = rng.random(10) + 1.0
    dc = rng.random(10) + 1.0
    db[zb == 3] = np.nan

    parts = [
        _block_partials(
            z.reshape(1, 1, -1), [d.reshape(1, 1, -1)], None, 0, False
        )
        for z, d in [(za, da_), (zb, db), (zc, dc)]
    ]
    assert 4 not in parts[0].index.get_level_values("zone")
    assert (parts[1]["count"] == 0).any()

    merged = _merge_moments(pd.concat(parts)).sort_index()
    zfull = np.concatenate([za, zb, zc])
    dfull = np.concatenate([da_, db, dc])
    whole = _block_partials(
        zfull.reshape(1, 1, -1), [dfull.reshape(1, 1, -1)], None, 0, False
    ).sort_index()

    assert merged.index.equals(whole.index)
    for col in ["size", "count", "sum", "prod", "min", "max", "m2"]:
        np.testing.assert_allclose(
            merged[col].to_numpy(),
            whole[col].to_numpy(),
            rtol=1e-12,
        )


def test_zone_codes_paths():
    # Direct path: small non-negative uint16 ids with the null id present.
    z = np.array([[0, 1, 2], [2, 3, 0]], dtype="uint16")
    ids, codes, nzones, null_code = _zone_codes(z, 0)
    assert nzones == 4
    assert null_code == 0
    assert ids.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(ids, np.arange(4))
    np.testing.assert_array_equal(codes, z.ravel())

    # Unique path: int64 ids with a negative null id.
    zi = np.array([-5, 10, 10, 2_000_000, -5], dtype="int64")
    ids, codes, nzones, null_code = _zone_codes(zi, -5)
    np.testing.assert_array_equal(ids, np.array([-5, 10, 2_000_000]))
    assert nzones == 3
    assert null_code == 0
    np.testing.assert_array_equal(ids[codes], zi)

    # Direct path with the null id outside the id range -> null_code == -1.
    z2 = np.array([1, 2, 3], dtype="uint16")
    _, _, _, nc_none = _zone_codes(z2, None)
    assert nc_none == -1
    _, _, _, nc_out = _zone_codes(z2, 9)
    assert nc_out == -1

    # Ids above the direct-index threshold -> unique path.
    big = np.array([_MAX_DIRECT_ZONE_ID + 5, 3, 3], dtype="int64")
    ids, codes, nzones, null_code = _zone_codes(big, None)
    np.testing.assert_array_equal(ids, np.array([3, _MAX_DIRECT_ZONE_ID + 5]))
    assert nzones == 2
    assert null_code == -1
    np.testing.assert_array_equal(ids[codes], big)


_read_lock = threading.Lock()
_read_calls = {"n": 0}


def _counting_loader(block):
    # Module-level so dask can tokenize it deterministically. Counts how many
    # times a data block is materialized.
    with _read_lock:
        _read_calls["n"] += 1
    return block


@pytest.mark.parametrize("stats", [["mean", "mode"], ["mean", "median"]])
def test_zonal_reads_each_block_once(stats):
    # Guards the optimize_graph=False decision: the fast path and the median
    # path must share block-read keys so each data block is read exactly once.
    with dask.config.set(
        {
            "array.chunk-size": "32KiB",
            "scheduler": "threads",
            "num_workers": 2,
        }
    ):
        ncs = da.empty((1, 128, 128), dtype=F64).chunksize
        base = np.arange(1.0 * 128 * 128).reshape(1, 128, 128) % 37 + 1.0
        arr = da.from_array(base, chunks=ncs)
        wrapped = da.map_blocks(
            _counting_loader, arr, dtype=arr.dtype, name="zonal-read-probe"
        )
        data = rts.Raster(wrapped).set_null_value(-1.0)
        nblocks = int(np.prod(data.data.numblocks))
        assert nblocks > 1
        zones = (np.arange(128 * 128).reshape(1, 128, 128) % 5 + 1).astype(
            "uint16"
        )
        feat = rts.Raster(zones).set_null_value(0)

        _read_calls["n"] = 0
        zonal_stats(feat, data, stats).compute()
        assert _read_calls["n"] == nblocks
