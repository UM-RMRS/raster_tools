from collections.abc import Sequence
from functools import partial

import dask
import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numba as nb
import numpy as np
import pandas as pd

from raster_tools.dtypes import F64, I64, is_int, is_str
from raster_tools.raster import Raster, get_raster
from raster_tools.utils import version_to_tuple
from raster_tools.vector import Vector, get_vector

__all__ = ["ZONAL_STAT_FUNCS", "extract_points_eager", "zonal_stats"]


@nb.jit(nopython=True, nogil=True)
def _entropy(counts):
    if len(counts) == 0:
        return np.nan
    # Manual stat calculation with a loop is faster than using vectorized numpy
    # operations, likely due to numpy ops allocating temporary arrays.
    entropy = 0.0
    frac = 1 / np.sum(counts)
    for cnt in counts:
        p = cnt * frac
        entropy -= p * np.log(p)
    return entropy


@nb.jit(nopython=True, nogil=True)
def _asm(counts):
    if len(counts) == 0:
        return np.nan
    # Manual stat calculation with a loop is faster than using vectorized numpy
    # operations, likely due to numpy ops allocating temporary arrays.
    asm = 0.0
    frac = 1 / np.sum(counts)
    for cnt in counts:
        p = cnt * frac
        asm += p * p
    return asm


def _unique_with_counts_chunk(grps):
    return grps.apply(lambda s: np.unique(s, return_counts=True))


def _combine_values_and_counts(s):
    result = (
        pd.DataFrame(
            {
                "values_": [vs for vs, _ in s.to_numpy()],
                "counts_": [cs for _, cs in s.to_numpy()],
            }
        )
        .explode(["values_", "counts_"])
        .groupby("values_")
        .sum()
    )
    # Convert to lists because of weird behavior in finalize step. If the
    # results are not converted to lists, the finalize step will get a series
    # where each row is a tuple of arrays of objects instead of a tuple of
    # lists of numbers. I don't know why this happens, but the following works.
    return (result.index.to_list(), result.counts_.to_list())


def _mode_fin(s):
    # Final operation of the mode calculation
    results = np.empty(len(s), dtype=F64)
    for i, (vs, cs) in enumerate(s.to_numpy()):
        if len(vs) != 0:
            # Order the counts descending with values ascending. In the case of
            # tied counts, lower values come first. This mirrors the behavior
            # of scipy's mode function.
            values_and_counts = pd.DataFrame(
                {"values_": vs, "counts_": cs}
            ).sort_values(by=["counts_", "values_"], ascending=[False, True])
            results[i] = values_and_counts.values_.iloc[0]
        else:
            results[i] = np.nan
    return pd.Series(results, index=s.index.copy())


def _asm_entropy_fin(s, func):
    # Final operation of the asm and entropy calculations
    if len(s) == 0:
        return pd.Series(np.array((), dtype=F64), index=s.index)
    result_list = []
    for _, cs in s.to_numpy():
        result_list.append(func(np.asarray(cs)))
    return pd.Series(result_list, index=s.index.copy())


# Notes for mode, asm, and entropy aggs:
# chunk:
#     Each row is now a 2-tuple of numpy arrays representing the unique values
#     and their counts
# agg:
#     Compbine the values and counts for zone value into a single 2-tuple of
#     values and counts lists.
# finalize:
#     Compute final stat from values and counts.
_mode_agg = dd.Aggregation(
    "mode",
    chunk=lambda grps: grps.apply(lambda s: np.unique(s, return_counts=True)),
    agg=lambda grps: grps.apply(_combine_values_and_counts),
    finalize=_mode_fin,
)
_asm_agg = dd.Aggregation(
    "asm",
    chunk=lambda grps: grps.apply(lambda s: np.unique(s, return_counts=True)),
    agg=lambda grps: grps.apply(_combine_values_and_counts),
    finalize=partial(_asm_entropy_fin, func=_asm),
)
_entropy_agg = dd.Aggregation(
    "entropy",
    chunk=_unique_with_counts_chunk,
    agg=lambda grps: grps.apply(_combine_values_and_counts),
    finalize=partial(_asm_entropy_fin, func=_entropy),
)
_nunique_agg = dd.Aggregation(
    name="nunique",
    chunk=lambda s: s.apply(lambda x: list(set(x.dropna()))),
    agg=lambda s0: s0.obj.groupby(
        level=list(range(s0.obj.index.nlevels))
    ).sum(),
    finalize=lambda s1: s1.apply(lambda final: len(set(final))),
)


_DASK_STAT_NAMES = frozenset(
    (
        "count",
        "max",
        "mean",
        "median",
        "min",
        "prod",
        "size",
        "std",
        "sum",
        "var",
    )
)
_CUSTOM_STAT_NAMES = frozenset(("asm", "entropy", "mode", "nunique"))
_CUSTOM_STAT_AGGS = {
    "asm": _asm_agg,
    "entropy": _entropy_agg,
    "mode": _mode_agg,
    "nunique": _nunique_agg,
}
ZONAL_STAT_FUNCS = frozenset(sorted(_DASK_STAT_NAMES | _CUSTOM_STAT_NAMES))


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
    # Convert the underlying dask array into a dask Series
    # DataArray.to_dask_dataframe is only available in newer versions of xarray
    if hasattr(raster.xdata, "to_dask_dataframe"):
        return raster.xdata.to_dask_dataframe(["band", "x", "y"])["raster"]
    return raster._ds.to_dask_dataframe(["band", "x", "y"])["raster"]


def _zonal_stats(features_raster, data_raster, stats):
    """
    Convert data to large dask DataFrame, group by the feature zone ids, and
    apply aggregations specified by `stats` for each band.
    """
    stat_names = stats
    stats, has_median = _find_problem_stats(stats)
    stats = [
        s if s in _DASK_STAT_NAMES else _CUSTOM_STAT_AGGS[s] for s in stats
    ]
    if features_raster.data.chunks[1:] != data_raster.data.chunks[1:]:
        # Force rasters to have matching chunksizes
        features_raster = features_raster.chunk(data_raster.data.chunks)
    # Convert the features raster and data raster bands to dataframes and join
    # them together. Each cell becomes a row and each chunk becomes a
    # partition. The order is the same and the chunk boundaries are the same so
    # they should concat together, cleanly.
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
    # DataFrame structure:
    #        zone  band_1  band_2  ...
    # index
    #     0    --      --      --  ...
    combined_raster_df = dd.concat(raster_dfs, axis=1)
    # Filter out non-feature areas
    combined_raster_df = combined_raster_df[
        combined_raster_df.zone != features_raster.null_value
    ]
    grouped = combined_raster_df.groupby("zone")
    agg_result_df = None
    if has_median:
        # median requires special treatment. It needs the shuffle arg and also
        # causes TypeErrors when used with custom agg functions.
        # ref: https://github.com/dask/dask/issues/10517
        # Stay backword compatible with older versions of dask
        if version_to_tuple(dask.__version__) < (2024, 1, 1):
            shuffle_kw = "shuffle"
        else:
            shuffle_kw = "shuffle_method"
        median_df = grouped.agg(["median"], **{shuffle_kw: "tasks"})
    if len(stats):
        agg_result_df = grouped.agg(stats)
        if has_median:
            agg_result_df = agg_result_df.join(median_df)
    else:
        # Only median was provided
        agg_result_df = median_df
    if has_median:
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
            Angular second moment. Applies -sum(P(g)**2) where P(g) gives the
            probability of g within the neighborhood.
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
            raise TypeError("Feature raster must be an integer type.")
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

    # Rechunk based on largest (probable) dtype to avoid overly large chunks,
    # which could cause memory issues down the pipeline. For instance, if the
    # data raster has dtype of f32 but the rasterization of the features
    # produces an i64 raster, the features raster will have double the memory
    # footprint, for each chunk, compared to the original data raster. This
    # causes dask to raise warnings about chunk sizes and drastically increases
    # the likelihood of running out of memory at compute time.
    # Rechunking to an 8-byte dtype helps mitigate the potential for memory
    # pressure at compute time. It is done here because the chunksize of the
    # data raster determines the chunksize of the features raster.
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
        "band" and `column_name variable. If axis = 0, the output band column
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
