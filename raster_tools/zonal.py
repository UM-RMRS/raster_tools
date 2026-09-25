from collections.abc import Sequence

import dask
import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numba as nb
import numpy as np
import pandas as pd

from raster_tools.dtypes import F64, I64, is_float, is_int, is_str
from raster_tools.raster import Raster, get_raster
from raster_tools.vector import Vector, get_vector

__all__ = ["ZONAL_STAT_FUNCS", "extract_points_eager", "zonal_stats"]


# Stats computed from per-block moments (size, count, sum, product, min, max,
# and squared deviations).
_MOMENT_STATS = frozenset(
    ("count", "max", "mean", "min", "prod", "size", "std", "sum", "var")
)
# Stats computed from per-block (zone, value) counts.
_COUNT_STATS = frozenset(("asm", "entropy", "mode", "nunique"))
# Stats returned as int64 rather than float64.
_INT_STATS = frozenset(("count", "size", "nunique"))
# Largest zone id used directly as an accumulator index. Larger ids, or any
# negative id, are factorized with np.unique before accumulation.
_MAX_DIRECT_ZONE_ID = 2**20

ZONAL_STAT_FUNCS = frozenset(sorted(_MOMENT_STATS | _COUNT_STATS | {"median"}))


def _build_long_format_meta(df):
    return pd.DataFrame(
        {
            "zone": np.array((), dtype=df.index.dtype),
            "band": np.array((), dtype=I64),
            **{
                s: np.array((), dtype=dtype)
                for s, dtype in df.dtypes["band_1"].items()
            },
        },
        index=np.array((), dtype=I64),
    )


def _melt_part(part):
    # Convert a dataframe from wide format to long format
    #
    # DataFrame structure, the columns are a MultiIndex and zone ids
    # are the index:
    #       band_1             band_2             ...
    #       stat1  stat2  ...  stat1  stat2  ...  ...
    # zone
    #    1     --     --  ...
    part = (
        # Unpivot band_# column labels into the index. The index is now a
        # MultiIndex of (zone, level_1) where level_1 has the band labels
        # The PD013 warning is silenced below because I could not get melt to
        # reproduce the desired result (I tried A LOT).
        part.stack(0, future_stack=True)  # noqa: PD013
        # Move the zone and level_1 indices to columns
        .reset_index()
        .rename(columns={"level_1": "band"})
        .sort_values(["band", "zone"])
    )
    # Convert band labels to 1-based integers
    part["band"] = part.band.apply(lambda x: int(x.split("_")[-1]))
    # New DataFrame structure:
    #        zone  band  stat1  stat2  ...
    # index
    #     0     1     1     --     --  ...
    return part


def _find_problem_stats(stats):
    has_median = False
    out = []
    for s in stats:
        if s != "median":
            out.append(s)
        else:
            has_median = True
    return out, has_median


def _raster_to_series(raster):
    # Build one dataframe partition per array block instead of ravelling the
    # whole 2D block grid with xarray's to_dask_dataframe. Ravelling a grid of
    # blocks into one long axis is not expressible as a blockwise reshape, so
    # dask inserts rechunk-split/merge steps that require whole column-bands of
    # blocks to be resident at once. Reshaping a single block to 1-D is always
    # a single task with no data movement, so this keeps peak memory to roughly
    # one block per running task. Two rasters built this way from identical
    # chunks put the same cell at the same partition and position, so
    # dd.concat(axis=1) lines them up without an index-based join.
    data = raster.data
    if any(np.isnan(size) for axis in data.chunks for size in axis):
        raise ValueError(
            "Raster has unknown chunk sizes. Call .chunk(...) or "
            "compute_chunk_sizes() on it before converting to a series."
        )
    nbands, ny, nx = data.numblocks
    blocks = [
        data.blocks[b, i, j].reshape(-1)
        for b in range(nbands)
        for j in range(nx)
        for i in range(ny)
    ]
    return dd.from_dask_array(da.concatenate(blocks), columns="raster")


@nb.jit(nopython=True, nogil=True)
def _zone_moments_kernel(codes, values, valid, nzones, null_code):
    # First pass builds size, count, a compensated sum, product, min, and max
    # for each zone code. The second pass sums squared deviations from each
    # zone mean so variance can be recombined across blocks.
    size = np.zeros(nzones, dtype=np.int64)
    count = np.zeros(nzones, dtype=np.int64)
    total = np.zeros(nzones, dtype=np.float64)
    comp = np.zeros(nzones, dtype=np.float64)
    prod = np.ones(nzones, dtype=np.float64)
    vmin = np.full(nzones, np.inf)
    vmax = np.full(nzones, -np.inf)
    for i in range(codes.size):
        c = codes[i]
        if c == null_code:
            continue
        size[c] += 1
        if not valid[i]:
            continue
        v = values[i]
        count[c] += 1
        # Kahan compensated summation, mirroring pandas' group sums.
        y = v - comp[c]
        t = total[c] + y
        comp[c] = (t - total[c]) - y
        total[c] = t
        prod[c] *= v
        if v < vmin[c]:
            vmin[c] = v
        if v > vmax[c]:
            vmax[c] = v
    m2 = np.zeros(nzones, dtype=np.float64)
    for i in range(codes.size):
        c = codes[i]
        if c == null_code or not valid[i]:
            continue
        d = values[i] - total[c] / count[c]
        m2[c] += d * d
    return size, count, total, prod, vmin, vmax, m2


def _zone_codes(zones, zone_null):
    # Map zone ids to dense codes for accumulation. Small non-negative ids are
    # used directly; wider or negative ids are factorized. Returns ids, codes,
    # the code count, and the code standing in for the null zone (-1 when no
    # zone is null). ids[codes] reproduces the input zones.
    zones = zones.ravel()
    zmin = int(zones.min())
    zmax = int(zones.max())
    if zmin >= 0 and zmax < _MAX_DIRECT_ZONE_ID:
        nzones = zmax + 1
        ids = np.arange(nzones, dtype=zones.dtype)
        codes = zones
        null_code = -1
        if zone_null is not None and 0 <= zone_null < nzones:
            null_code = int(zone_null)
    else:
        ids, codes = np.unique(zones, return_inverse=True)
        codes = codes.ravel()
        nzones = len(ids)
        null_code = -1
        if zone_null is not None:
            pos = int(np.searchsorted(ids, zone_null))
            if pos < nzones and ids[pos] == zone_null:
                null_code = pos
    return ids, codes, nzones, null_code


def _empty_counts(zone_dtype):
    return pd.DataFrame(
        {
            "zone": np.array((), zone_dtype),
            "band": np.array((), I64),
            "value": np.array((), F64),
            "n": np.array((), I64),
        }
    )


def _block_value_counts(ids, codes, values, valid, null_code, band):
    # One row per (zone, value) pair present in the block, with its count.
    # values is already float64 and nulls are already excluded by valid.
    keep = valid & (codes != null_code)
    vals_u, vinv = np.unique(values[keep], return_inverse=True)
    nvals = len(vals_u)
    if nvals == 0:
        return _empty_counts(ids.dtype)
    key = codes[keep].astype(I64) * nvals + vinv.ravel()
    key_u, n = np.unique(key, return_counts=True)
    return pd.DataFrame(
        {
            "zone": ids[key_u // nvals],
            "band": np.full(len(key_u), band, dtype=I64),
            "value": vals_u[key_u % nvals],
            "n": n,
        }
    )


def _block_partials(zones, bands, data_null, zone_null, want_counts):
    # zones is a (1, ny, nx) block; bands is a list of (1, ny, nx) blocks, one
    # per data band. Returns the moment frame, or (moment frame, count frame)
    # when value-count stats are requested. All bands share the zone set taken
    # from band 1's size, which counts cells regardless of data validity.
    ids, codes, nzones, null_code = _zone_codes(zones, zone_null)
    moment_parts = []
    count_parts = []
    present = None
    for b, band in enumerate(bands, 1):
        values = band.ravel().astype(F64, copy=False)
        valid = ~np.isnan(values)
        if data_null is not None and not np.isnan(data_null):
            valid &= values != data_null
        size, count, total, prod, vmin, vmax, m2 = _zone_moments_kernel(
            codes, values, valid, nzones, null_code
        )
        if present is None:
            present = np.nonzero(size)[0]
        index = pd.MultiIndex.from_arrays(
            [ids[present], np.full(len(present), b, dtype=I64)],
            names=["zone", "band"],
        )
        moment_parts.append(
            pd.DataFrame(
                {
                    "size": size[present],
                    "count": count[present],
                    "sum": total[present],
                    "prod": prod[present],
                    "min": vmin[present],
                    "max": vmax[present],
                    "m2": m2[present],
                },
                index=index,
            )
        )
        if want_counts:
            count_parts.append(
                _block_value_counts(ids, codes, values, valid, null_code, b)
            )
    moments = pd.concat(moment_parts)
    if want_counts:
        return moments, pd.concat(count_parts)
    return moments


def _merge_moments(part):
    # part is indexed by (zone, band) and may hold one row per block for a key.
    # Combine the rows, recombining squared deviations with the parallel
    # variance formula M2 = sum(M2_i) + sum(n_i * (mean_i - mean) ** 2).
    if part.index.is_unique:
        return part
    g = part.groupby(level=[0, 1], sort=True)
    count = g["count"].sum()
    total = g["sum"].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = total / count
        delta = (
            part["count"]
            * (part["sum"] / part["count"] - mean.reindex(part.index)) ** 2
        )
    delta = delta.fillna(0.0)
    return pd.DataFrame(
        {
            "size": g["size"].sum(),
            "count": count,
            "sum": total,
            "prod": g["prod"].prod(),
            "min": g["min"].min(),
            "max": g["max"].max(),
            "m2": g["m2"].sum() + delta.groupby(level=[0, 1]).sum(),
        }
    )


def _merge_counts(part):
    return part.groupby(["zone", "band", "value"], sort=True, as_index=False)[
        "n"
    ].sum()


def _finalize_moments(part, stats):
    count = part["count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = part["sum"] / count
        var = (part["m2"] / (count - 1)).where(count > 1)
    cols = {}
    for s in stats:
        if s in ("count", "size", "sum", "prod"):
            cols[s] = part[s]
        elif s in ("min", "max"):
            cols[s] = part[s].where(count > 0)
        elif s == "mean":
            cols[s] = mean
        elif s == "var":
            cols[s] = var
        elif s == "std":
            cols[s] = np.sqrt(var)
    return pd.DataFrame(cols, index=part.index)


def _finalize_counts(counts, stats):
    # counts holds merged (zone, band, value, n) rows. Each stat is computed
    # vectorized over the groups, with no per-zone Python loop.
    keys = [counts["zone"], counts["band"]]
    g = counts.groupby(keys, sort=True)
    cols = {}
    if "nunique" in stats:
        cols["nunique"] = g.size()
    if "mode" in stats:
        # Highest count wins; ties break to the lowest value, matching scipy.
        first = counts.sort_values(
            ["zone", "band", "n", "value"],
            ascending=[True, True, False, True],
        ).drop_duplicates(["zone", "band"])
        cols["mode"] = pd.Series(
            first["value"].to_numpy(dtype=F64),
            index=pd.MultiIndex.from_frame(first[["zone", "band"]]),
        )
    if "asm" in stats or "entropy" in stats:
        p = counts["n"] / g["n"].transform("sum")
        if "asm" in stats:
            cols["asm"] = (p * p).groupby(keys).sum()
        if "entropy" in stats:
            cols["entropy"] = -(p * np.log(p)).groupby(keys).sum()
    out = pd.DataFrame(cols)
    out.index.names = ["zone", "band"]
    return out


def _to_wide(df, nbands, stats):
    # df is indexed by (zone, band) with one column per stat. Produce a frame
    # whose columns are (band_b, stat) in the caller's stat order and whose
    # index is the zone id.
    wide = df.unstack("band")  # noqa: PD010
    cols = {}
    for b in range(1, nbands + 1):
        for s in stats:
            if (s, b) in wide.columns:
                col = wide[(s, b)]
            else:
                col = pd.Series(np.nan, index=wide.index)
            if s in _INT_STATS:
                col = col.fillna(0).astype(I64)
            cols[(f"band_{b}", s)] = col
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(list(cols))
    out.index.name = "zone"
    return out


def _finalize_block_stats(moments, counts, mstats, cstats, nbands, stats):
    res = _finalize_moments(moments, mstats)
    if counts is not None:
        res = res.join(_finalize_counts(counts, cstats), how="left")
    return _to_wide(res, nbands, stats)


def _moments_meta(zone_dtype):
    return pd.DataFrame(
        {
            "size": np.array((), I64),
            "count": np.array((), I64),
            "sum": np.array((), F64),
            "prod": np.array((), F64),
            "min": np.array((), F64),
            "max": np.array((), F64),
            "m2": np.array((), F64),
        },
        index=pd.MultiIndex.from_arrays(
            [np.array((), zone_dtype), np.array((), I64)],
            names=["zone", "band"],
        ),
    )


def _wide_meta(zone_dtype, nbands, stats):
    cols = {
        (f"band_{b}", s): np.array((), I64 if s in _INT_STATS else F64)
        for b in range(1, nbands + 1)
        for s in stats
    }
    meta = pd.DataFrame(cols, index=np.array((), dtype=zone_dtype))
    meta.columns = pd.MultiIndex.from_tuples(list(cols))
    meta.index.name = "zone"
    return meta


def _block_zonal_stats(features_raster, data_raster, stats):
    # Non-median stats are computed by reducing small per-block partial frames.
    # The delayed block reads use optimize_graph=False so that they share keys
    # with the median path's block reads and each block is read only once.
    zone_null = features_raster.null_value
    zone_dtype = features_raster.dtype
    nbands, ny, nx = data_raster.data.numblocks
    # Later code indexes bands by block; that holds only because zonal_stats
    # rechunks the data raster to one band per block before this runs.
    assert nbands == data_raster.shape[0], "one band per block required"
    mstats = [s for s in stats if s in _MOMENT_STATS]
    cstats = [s for s in stats if s in _COUNT_STATS]
    want_counts = bool(cstats)
    zone_blocks = features_raster.data.to_delayed(optimize_graph=False)
    data_blocks = data_raster.data.to_delayed(optimize_graph=False)
    func = dask.delayed(_block_partials, nout=2 if want_counts else None)
    parts = [
        func(
            zone_blocks[0, i, j],
            [data_blocks[b, i, j] for b in range(nbands)],
            data_raster.null_value,
            zone_null,
            want_counts,
        )
        for i in range(ny)
        for j in range(nx)
    ]
    mmeta = _moments_meta(zone_dtype)
    moments = dd.from_delayed(
        [p[0] for p in parts] if want_counts else parts,
        meta=mmeta,
        verify_meta=False,
    ).reduction(
        chunk=_merge_moments,
        combine=_merge_moments,
        aggregate=_merge_moments,
        meta=mmeta,
    )
    wide_meta = _wide_meta(zone_dtype, nbands, stats)
    if want_counts:
        cmeta = _empty_counts(zone_dtype)
        # The value-count stats (asm, entropy, mode, nunique) are intended for
        # integer or categorical data. On continuous float data each block's
        # (zone, band, value, count) table is nearly cell-sized, so a smaller
        # fan-in bounds the peak memory of each merge task.
        counts = dd.from_delayed(
            [p[1] for p in parts], meta=cmeta, verify_meta=False
        ).reduction(
            chunk=_merge_counts,
            combine=_merge_counts,
            aggregate=_merge_counts,
            meta=cmeta,
            split_every=4,
        )
        return dd.map_partitions(
            _finalize_block_stats,
            moments,
            counts,
            mstats,
            cstats,
            nbands,
            stats,
            align_dataframes=False,
            meta=wide_meta,
        )
    return moments.map_partitions(
        _finalize_block_stats,
        None,
        mstats,
        cstats,
        nbands,
        stats,
        meta=wide_meta,
    )


def _median_stats(features_raster, data_raster):
    # Median needs the full per-cell frame, so it stays on the group-by path.
    # It also needs the shuffle argument and cannot share the custom aggs.
    # ref: https://github.com/dask/dask/issues/10517
    raster_dfs = [_raster_to_series(features_raster).rename("zone")]
    for b in range(1, data_raster.nbands + 1):
        band = _raster_to_series(data_raster.get_bands(b)).rename(f"band_{b}")
        if data_raster.null_value is not None and not np.isnan(
            data_raster.null_value
        ):
            # Replace null values with NA values
            band = band.replace(data_raster.null_value, np.nan)
        # Cast up to avoid floating point issues in sums. F32 sums loose
        # precision quickly even for smaller rasters. This affects any stat
        # that involves a sum such as std, var, and mean.
        # ref: https://web.archive.org/web/20230329091023/https://pythonspeed.com/articles/float64-float32-precision/  # noqa
        if band.dtype != F64:
            band = band.astype(F64)
        raster_dfs.append(band)
    combined_raster_df = dd.concat(raster_dfs, axis=1)
    # Filter out non-feature areas
    combined_raster_df = combined_raster_df[
        combined_raster_df.zone != features_raster.null_value
    ]
    grouped = combined_raster_df.groupby("zone")
    median = grouped.agg(["median"], shuffle_method="tasks")
    # Dask's group-by median splits its output across multiple partitions once
    # there are more than fifteen input blocks (split_out is ceil(nblocks /
    # 15)). Joining a multi-partition frame that carries MultiIndex columns
    # trips a dask-expr merge error on its "_partitions" shuffle helper column,
    # so collapse the medians to one partition before returning. The per-zone
    # medians are tiny, so a single partition costs nothing.
    return median.repartition(npartitions=1)


def _zonal_stats(features_raster, data_raster, stats):
    """
    Compute zonal statistics per band, grouped by the feature zone ids.

    Non-median stats use a per-block partial-aggregation path. Median stays on
    the group-by path and is joined onto the other stats afterward.
    """
    stat_names = stats
    stats, has_median = _find_problem_stats(stats)
    if features_raster.data.chunks[1:] != data_raster.data.chunks[1:]:
        # Force rasters to have matching chunksizes. The features raster is
        # single-band, so keep its band axis at one chunk and match only the
        # spatial chunks; passing the data raster's band chunks would break
        # when the data raster has multiple bands.
        features_raster = features_raster.chunk(
            (1, *data_raster.data.chunks[1:])
        )
    agg_result_df = None
    if len(stats):
        agg_result_df = _block_zonal_stats(features_raster, data_raster, stats)
    if has_median:
        median_df = _median_stats(features_raster, data_raster)
        if agg_result_df is not None:
            agg_result_df = agg_result_df.join(median_df)
        else:
            # Only median was provided
            agg_result_df = median_df
        # Shuffle columns to be in order of the original stats
        col_tuples = []
        for b in range(1, data_raster.nbands + 1):
            band = f"band_{b}"
            col_tuples.extend((band, stat) for stat in stat_names)
        target_columns = pd.MultiIndex.from_tuples(col_tuples)
        if not agg_result_df.columns.equals(target_columns):
            agg_result_df = agg_result_df[target_columns]
    # DataFrame structure, the columns are a MultiIndex and zone ids are the
    # index:
    #       band_1             band_2             ...
    #       stat1  stat2  ...  stat1  stat2  ...  ...
    # zone
    #    1     --     --  ...
    return agg_result_df


def zonal_stats(
    features,
    data_raster,
    stats,
    features_field=None,
    wide_format=True,
    handle_overlap=False,
):
    """Apply stat functions to a raster based on a set of features.

    Parameters
    ----------
    features : str, Vector, Raster
        A `Vector` or path string pointing to a vector file or a categorical
        Raster. The vector features are used like cookie cutters to pull data
        from the `data_raster` bands. If `features` is a Raster, it must be an
        int dtype and have only one band.
    data_raster : Raster, str
        A `Raster` or path string pointing to a raster file. The data raster
        to pull data from and apply the stat functions to.
    stats : str, list of str
        A single string or list of strings corresponding to stat functions.
        These functions will be applied to the raster data for each of the
        features in `features`. Valid string values:

        'asm'
            Angular second moment. Applies sum(P(g)**2) where P(g) gives the
            probability of g within the zone.
        'count'
            Count valid cells.
        'entropy'
            Calculates the entropy. Applies -sum(P(g) * log(P(g))). See 'asm'
            above.
        'max'
            Find the maximum value.
        'mean'
            Calculate the mean.
        'median'
            Calculate the median value.
        'min'
            Find the minimum value.
        'mode'
            Compute the statistical mode of the data. In the case of a tie, the
            lowest value is returned.
        'nunique'
            Count unique values.
        'prod'
            Calculate the product.
        'size'
            Calculate zone size.
        'std'
            Calculate the standard deviation.
        'sum'
            Calculate the sum.
        'var'
            Calculate the variance.
    features_field : str, optional
        If the `features` argument is a vector, this determines which field to
        use when rasterizing the features. It must match one of the fields in
        `features`. The default is to use `features`' index.
    wide_format : bool, optional
        If ``True``, the resulting dataframe is returned in wide format where
        the columns are a cartesian product of the `data_raster` bands and the
        specified stats and the index contains the feature zone IDs.

        .. code-block::

            pandas.MultiIndex(
              [
                ('band_1', 'stat1'),
                ('band_1', 'stat2'),
                ...
                ('band_2', 'stat1'),
                ('band_2', 'stat2'),
                ...
              ],
            )

        If ``False``, the resulting dataframe has columns `'zone', 'band',
        'stat1', 'stat2', ...` and an integer index. In this case, the zone
        column contains the feature zone IDs and band contains the one-base
        integer band number. The rest of the columns correspond to the
        specified stats.

        The default is wide format.
    handle_overlap: bool, optional
        Normally, polygon inputs for `features` are converted to a raster. This
        means that a cell can have only one value. In the case of overlapping
        polygons, one polygon will trump the others and the resulting
        statistics for all of the incident polygons may be affected. If
        ``True``, overlapping polygons are accounted for and zonal statistics
        will be calculated independent of overlap. Currently this will trigger
        computation of `features`. The default is ``False``.

    Returns
    -------
    dask.dataframe.DataFrame
        A delayed dask DataFrame where the specified stats have been applied to
        the bands in `data_raster`. See the `wide_format` option for a
        description of the dataframe's structure.

    """
    in_memory = False
    if isinstance(
        features,
        (
            str,
            Vector,
            dgpd.GeoDataFrame,
            dgpd.GeoSeries,
            gpd.GeoDataFrame,
            gpd.GeoSeries,
        ),
    ):
        in_memory = isinstance(features, (gpd.GeoDataFrame, gpd.GeoSeries))
        features = get_vector(features)
        if in_memory:
            features = features.calculate_spatial_partitions()
    elif isinstance(features, Raster):
        if not is_int(features.dtype):
            msg = (
                "Feature raster must be an integer type, got "
                f"{features.dtype}."
            )
            if is_float(features.dtype):
                msg += (
                    " Saving integer rasters to GeoTIFF on older GDAL"
                    " casts them to float64; use .astype(...) to convert"
                    " back to an integer type."
                )
            raise TypeError(msg)
        if features.shape[0] > 1:
            raise ValueError("Feature raster must have only 1 band.")
    else:
        raise TypeError(
            "Could not understand features arg. Must be Vector, str or Raster"
        )
    data_raster = get_raster(data_raster)
    if is_str(stats):
        stats = [stats]
    elif isinstance(stats, Sequence):
        stats = list(stats)
        if not stats:
            raise ValueError("No stat functions provide")
    else:
        raise ValueError(f"Could not understand stats arg: {repr(stats)}")
    for stat in stats:
        if stat not in ZONAL_STAT_FUNCS:
            raise ValueError(f"Invalid stats function: {repr(stat)}")
    duplicate = next((s for s in stats if stats.count(s) > 1), None)
    if duplicate is not None:
        raise ValueError(f"Duplicate stats function: {repr(duplicate)}")

    if handle_overlap:
        if isinstance(features, Raster):
            raise ValueError(
                "'features' cannont be a raster when 'handle_overlap' is True"
            )
        features = features.data.compute()
        features = [features.iloc[[i]] for i in range(len(features))]
        result_dfs = [
            # Recurse with single feature
            zonal_stats(
                f,
                data_raster,
                stats,
                features_field=features_field,
                wide_format=wide_format,
            )
            for f in features
        ]
        return dd.concat(result_dfs).repartition(npartitions=1)

    if isinstance(features, Raster):
        if features.crs != data_raster.crs:
            raise ValueError("Feature raster CRS must match data raster")
        if features.shape[1:] != data_raster.shape[1:]:
            raise ValueError("Feature raster shape must match data raster")

    # Rechunk based on the largest probable dtype to avoid overly large
    # chunks, which could cause memory issues down the pipeline. Feature
    # rasterization burns the smallest unsigned dtype that holds the zone ids
    # and falls back to int64, so a features raster can reach 8 bytes per cell
    # -- up to double the footprint of, say, an f32 data raster, for each
    # chunk. That triggers dask chunk-size warnings and raises the likelihood
    # of running out of memory at compute time. Rechunking to an 8-byte dtype
    # here mitigates that; it is done here because the data raster's chunksize
    # determines the features raster's chunksize.
    new_chunksize = da.empty((1, *data_raster.shape[1:]), dtype=F64).chunksize
    data_raster = data_raster.chunk(new_chunksize)
    features_raster = None
    if isinstance(features, Vector):
        if features_field is not None and features_field not in features.data:
            raise KeyError(
                "features_field must be a field name in the features input"
            )
        features_raster = features.to_raster(
            data_raster,
            field=features_field,
            use_spatial_aware=in_memory,
        )
    else:
        if features.nbands > 1:
            raise ValueError("features raster must have a single band")
        if features.shape[1:] != data_raster.shape[1:]:
            raise ValueError(
                "features raster shape must match the data raster. "
                f"Expected {data_raster.shape[1:]}, got {features.shape[1:]}."
            )
        features_raster = features

    zonal_result_df = _zonal_stats(features_raster, data_raster, stats)
    if not wide_format:
        # New DataFrame structure:
        #        zone  band  stat1  stat2  ...
        # index
        #     0     1     1     --     --  ...
        zonal_result_df = zonal_result_df.map_partitions(
            _melt_part, meta=_build_long_format_meta(zonal_result_df)
        )
    return zonal_result_df


def _create_dask_range_index(start, stop):
    # dask.dataframe only allows dask.dataframe.index objects but doesn't have
    # a way to create them. this is a hack to create one using from_pandas.
    dummy = pd.DataFrame(
        {"tmp": np.zeros(stop - start, dtype="u1")},
        index=pd.RangeIndex(start, stop),
    )
    return dd.from_pandas(dummy, 1).index


def extract_points_eager(
    points, raster, column_name="extracted", skip_validation=True, axis=0
):
    """Extract the raster cell values using point features

    Note
    ----
    This function is partially eager. The x and y values for the target points
    are computed. The result is still a lazy dask DataFrame.


    This finds the grid cells that the points fall into and extracts the value
    at each point. The input feature will be partially computed to make sure
    that all of the geometries are points, unless `skip_validation` is set to
    `True`.

    Parameters
    ----------
    points : str, Vector
        The points to use for extracting data.
    raster : str, Raster
        The raster to pull data from.
    column_name : str, optional
        The column name to use for the extracted data points. Default is
        `"extracted"`.
    skip_validation : bool, optional
        If `True`, the input `points` is not validated to make sure that all
        features are points. This prevents partially computing the data.
        Default is `True`.
    axis : int, optional
        If 0 band column and values will be appended to a dataframe. Otherwise
        band values will be append to the columns named after the prefix and
        band of a dataframe

    Returns
    -------
    dask.dataframe.DataFrame
        The columns names depend on the value of axis and are based on the
        "band" and `column_name` variable. If axis = 0, the output band column
        within the dataframe identifies the band the value was extracted from.
        The values within the column named after the column name variable are
        the extracted values from the given band. Otherwise, the column names
        within the dataframe are appended to the column_name prefix and provide
        the extracted values. NaN values in the extracted column are where
        there was missing data in the raster or the point was outside the
        raster's domain.
    """
    points = get_vector(points)
    raster = get_raster(raster)

    if not len(column_name):
        raise ValueError("column_name must not be empty")
    if (
        not skip_validation
        and not (points.geometry.geom_type == "Point").all().compute()
    ):
        raise TypeError("All geometries must be points.")

    if raster.crs is not None and raster.crs != points.crs:
        gdf = points.to_crs(raster.crs).data
    else:
        gdf = points.data
    x = gdf.geometry.x.to_dask_array()
    y = gdf.geometry.y.to_dask_array()
    r, c = raster.index(*dask.compute(x, y))
    nb, nr, nc = raster.shape
    valid = (r >= 0) & (r < nr) & (c >= 0) & (c < nc)
    n = len(valid)
    dfs = []
    for i in range(nb):
        bnd = i + 1
        extracted = da.full(n, np.nan, dtype=F64)
        extracted[valid] = raster.data.vindex[i, r[valid], c[valid]]
        # Mask out missing points within the valid zones
        exmask = da.zeros(n, dtype=bool)
        exmask[valid] = raster.mask.vindex[i, r[valid], c[valid]]
        extracted[exmask] = np.nan
        if axis == 0:
            index = _create_dask_range_index(n * i, n * bnd)
            extracted_df = (
                da.full(n, bnd, dtype=np.min_scalar_type(nb + 1))
                .to_dask_dataframe(index=index)
                .to_frame("band")
            )
            extracted_df[column_name] = extracted.to_dask_dataframe(
                index=index
            )
        else:
            extracted_df = extracted.to_dask_dataframe(
                columns=column_name + "_" + str(bnd)
            )
        assert extracted_df.known_divisions
        dfs.append(extracted_df)
    extracted_result = dd.concat(dfs, axis=axis)
    return extracted_result
