from collections.abc import Iterable, Iterator, Sequence
from functools import partial

import dask
import dask.array as da
import dask.dataframe as dd
import numba as nb
import numpy as np
import pandas as pd
from dask_image import ndmeasure

from raster_tools.dask_utils import dask_nanmax, dask_nanmin
from raster_tools.dtypes import F64, I64, is_int, is_str
from raster_tools.raster import Raster, get_raster, xy_to_rowcol
from raster_tools.vector import Vector, get_vector

__all__ = ["ZONAL_STAT_FUNCS", "extract_points_eager", "zonal_stats"]


def _nan_count(x):
    return da.count_nonzero(~np.isnan(x))


def _nan_median(x):
    x = da.asarray(x)
    return da.nanmedian(x, axis=0)


def _nan_unique(x):
    return _nan_count(da.unique(x))


def _flatten_gen(x):
    """
    A generator that recursively yields numpy arrays from arbitrarily nested
    lists of arrays.
    """
    for xi in x:
        if isinstance(x, Iterable) and not isinstance(xi, np.ndarray):
            yield from _flatten_gen(xi)
        else:
            yield xi


def _flatten(x):
    """Flatten nested lists of arrays."""
    if isinstance(x, np.ndarray):
        return [x]
    return list(_flatten_gen(x))


def _recursive_map(func, *seqs):
    """Apply a function to items in nested sequences."""
    if isinstance(seqs[0], (list, Iterator)):
        return [_recursive_map(func, *items) for items in zip(*seqs)]
    return func(*seqs)


def _unique_with_counts_chunk(x, computing_meta=False, axis=(), **kwargs):
    """Reduce a dask chunk to a dict of unique values and counts.

    This is the leaf operation in the reduction tree.
    """
    if computing_meta:
        return x
    x_non_nan = x[~np.isnan(x)]
    values, counts = np.unique(x_non_nan, return_counts=True)
    while values.ndim < len(axis):
        values = np.expand_dims(values, axis=0)
        counts = np.expand_dims(counts, axis=0)
    return {"values": values, "counts": counts}


def _ravel_key(item, key):
    return item[key].ravel()


_ravel_values = partial(_ravel_key, key="values")
_ravel_counts = partial(_ravel_key, key="counts")


def _split_concat(pairs, split_func):
    # Split out a key from lists of dicts, ravel them, and concat all together
    split = _recursive_map(split_func, pairs)
    return np.concatenate(_flatten(split))


def _unique_with_counts_combine(
    pairs, computing_meta=False, axis=(), **kwargs
):
    """Merge/combine branches of the unique-with-counts reduction tree.

    This includes results from multiple _unique_with_counts_chunk calls and
    from prior _unique_with_counts_combine calls.
    """
    values = (
        _recursive_map(_ravel_values, pairs) if not computing_meta else pairs
    )
    values = np.concatenate(_flatten(values))
    if computing_meta:
        return np.array([[[0]]], dtype=pairs.dtype)

    counts = _split_concat(pairs, _ravel_counts)
    res = {v: 0 for v in values}
    for v, c in zip(values, counts):
        res[v] += c
    values = np.array(list(res.keys()))
    counts = np.array(list(res.values()))
    while values.ndim < len(axis):
        values = np.expand_dims(values, axis=0)
        counts = np.expand_dims(counts, axis=0)
    return {"values": values, "counts": counts}


def _mode_agg(pairs, computing_meta=False, axis=(), **kwargs):
    """Perform the final aggregation to a single mode value."""
    values = (
        _split_concat(pairs, _ravel_values) if not computing_meta else pairs
    )
    if computing_meta:
        return pairs.dtype.type(0)
    if len(values) == 0:
        # See note below about wrapping in np.array()
        return np.array(np.nan)
    counts = _split_concat(pairs, _ravel_counts)
    res = {v: 0 for v in values}
    for v, c in zip(values, counts):
        res[v] += c
    values = res.keys()
    counts = res.values()
    sorted_pairs = sorted(zip(counts, values), reverse=True)
    # Find the minimum mode when there is a tie. This is the same behavior as
    # scipy.
    i = -1
    c = sorted_pairs[0][0]
    for pair in sorted_pairs:
        if pair[0] == c:
            i += 1
        else:
            break
    # NOTE: wrapping the value in an array is a hack to prevent dask from
    # mishandling the return value as an array with dims, leading to index
    # errors. I can't pierce the veil of black magic that is causing the
    # mishandling so this is the best fix I can come up with.
    return np.array(sorted_pairs[i][1])


def _nan_mode(x):
    """
    Compute the statistical mode of an array using a dask reduction operation.
    """
    return da.reduction(
        x,
        chunk=_unique_with_counts_chunk,
        combine=_unique_with_counts_combine,
        aggregate=_mode_agg,
        # F64 to allow for potential empty input array. In that case a NaN is
        # returned.
        dtype=F64,
        # Turn off concatenation to prevent dask from trying to concat the
        # dicts of variable length values and counts. Dask tries to concat
        # along the wrong axis, which causes errors.
        concatenate=False,
    )


@nb.jit(nopython=True, nogil=True)
def _entropy(values, counts):
    if len(values) == 0:
        return np.nan
    res = {v: 0 for v in values}
    for v, c in zip(values, counts):
        res[v] += c
    counts = res.values()
    entropy = 0.0
    frac = 1 / len(res)
    for cnt in counts:
        p = cnt * frac
        entropy -= p * np.log(p)
    return entropy


@nb.jit(nopython=True, nogil=True)
def _asm(values, counts):
    if len(values) == 0:
        return np.nan
    res = {v: 0 for v in values}
    for v, c in zip(values, counts):
        res[v] += c
    counts = res.values()
    asm = 0.0
    frac = 1 / len(res)
    for cnt in counts:
        p = cnt * frac
        asm += p * p
    return asm


def _entropy_asm_agg(
    pairs, compute_entropy, computing_meta=False, axis=(), **kwargs
):
    """Perform the final aggregation to a single entropy or ASM value."""
    if computing_meta:
        return 0
    values = _split_concat(pairs, _ravel_values)
    if len(values) == 0:
        return np.array([])
    counts = _split_concat(pairs, _ravel_counts)
    # NOTE: wrapping the value in an array is a hack to prevent dask from
    # mishandling the return value as an array with dims, leading to index
    # errors. I can't pierce the veil of black magic that is causing the
    # mishandling so this is the best fix I can come up with.
    if compute_entropy:
        return np.array(_entropy(values, counts))
    return np.array(_asm(values, counts))


def _nan_entropy(x):
    """Compute the entropy of an array using a dask reduction operation."""
    return da.reduction(
        x,
        # mode chunk and combine funcs can be reused here
        chunk=_unique_with_counts_chunk,
        combine=_unique_with_counts_combine,
        aggregate=partial(_entropy_asm_agg, compute_entropy=True),
        dtype=F64,
        # Turn off concatenation to prevent dask from trying to concat the
        # dicts of variable length values and counts. Dask tries to concat
        # along the wrong axis, which causes errors.
        concatenate=False,
    )


def _nan_asm(x):
    """Compute the ASM of an array using a dask reduction operation.

    Angular second moment.
    """
    return da.reduction(
        x,
        # mode chunk and combine funcs can be reused here
        chunk=_unique_with_counts_chunk,
        combine=_unique_with_counts_combine,
        aggregate=partial(_entropy_asm_agg, compute_entropy=False),
        dtype=F64,
        # Turn off concatenation to prevent dask from trying to concat the
        # dicts of variable length values and counts. Dask tries to concat
        # along the wrong axis, which causes errors.
        concatenate=False,
    )


_ZONAL_STAT_FUNCS = {
    "asm": _nan_asm,
    "count": _nan_count,
    "entropy": _nan_entropy,
    "max": dask_nanmax,
    "mean": da.nanmean,
    "median": _nan_median,
    "min": dask_nanmin,
    "mode": _nan_mode,
    "std": da.nanstd,
    "sum": da.nansum,
    "unique": _nan_unique,
    "var": da.nanvar,
}
# The set of valid zonal function names/keys
ZONAL_STAT_FUNCS = frozenset(_ZONAL_STAT_FUNCS)


def _build_zonal_stats_data(data_raster, feat_raster, feat_labels, stats):
    nbands = data_raster.shape[0]
    feat_data = feat_raster.data
    # data will end up looking like:
    # {
    #   # band number
    #   1: {
    #     # Stat results
    #     "mean": [X, X, X], <- dask array
    #     "std": [X, X, X],
    #     ...
    #   },
    #   2: {
    #     # Stat results
    #     "mean": [X, X, X],
    #     "std": [X, X, X],
    #     ...
    #   },
    #   ...
    data = {}
    raster_data = get_raster(data_raster, null_to_nan=True).data
    for ibnd in range(nbands):
        ibnd += 1
        data[ibnd] = {}
        # Use range to keep band dimension intact
        band_data = raster_data[ibnd - 1 : ibnd]
        for f in stats:
            result_delayed = dask.delayed(ndmeasure.labeled_comprehension)(
                band_data,
                feat_data,
                feat_labels,
                _ZONAL_STAT_FUNCS[f],
                F64,
                np.nan,
            )
            data[ibnd][f] = da.from_delayed(
                result_delayed,
                feat_labels.shape,
                dtype=F64,
                meta=np.array([], dtype=F64),
            )
    return data


def _create_dask_range_index(start, stop):
    # dask.dataframe only allows dask.dataframe.index objects but doesn't have
    # a way to create them. this is a hack to create one using from_pandas.
    dummy = pd.DataFrame(
        {"tmp": np.zeros(stop - start, dtype="u1")},
        index=pd.RangeIndex(start, stop),
    )
    return dd.from_pandas(dummy, 1).index


def _build_zonal_stats_dataframe(zonal_data, nparts=None):
    bands = list(zonal_data)
    snames = list(zonal_data[bands[0]])
    n = zonal_data[bands[0]][snames[0]].size
    if nparts is None:
        # Get the number of partitions that dask thinks is reasonable. The data
        # arrays have chunks of size 1 so we need to rechunk later and then
        # repartition everything else in the dataframe to match.
        nparts = zonal_data[bands[0]][snames[0]].rechunk().npartitions

    df = None
    for bnd in bands:
        df_part = None
        band_data = zonal_data[bnd]
        band = da.full(n, bnd, dtype=I64)
        # We need to create an index because the concat operation later will
        # blindly paste in each dataframe's index. If an explicit index is not
        # set, the default is a range index from 0 to n. Thus the final
        # resulting dataframe would have identical indexes chained end-to-end:
        # [0, 1, ..., n-1, 0, 1, ..., n-1, 0, 1..., n-1]. By setting an index
        # we get [0, 1, ..., n, n+1, ..., n + n, ...].
        ind_start = n * (bnd - 1)
        ind_end = ind_start + n
        index = _create_dask_range_index(ind_start, ind_end)
        df_part = band.to_dask_dataframe("band", index=index).to_frame()
        # Repartition to match the data
        df_part = df_part.repartition(npartitions=nparts)
        index = index.repartition(npartitions=nparts)
        for name in snames:
            df_part[name] = (
                band_data[name].rechunk().to_dask_dataframe(name, index=index)
            )
        if df is None:
            df = df_part
        else:
            # Use interleave_partitions to keep partition and division info
            df = dd.concat([df, df_part], interleave_partitions=True)
    return df


def zonal_stats(features, data_raster, stats, raster_feature_values=None):
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
        A single string or list of strings corresponding to stat funcstions.
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
        'std'
            Calculate the standard deviation.
        'sum'
            Calculate the sum.
        'unique'
            Count unique values.
        'var'
            Calculate the variance.
    raster_feature_values : sequence of ints, optional
        Unique values to be used when the `features` argument is a Raster. If
        `features` is a Raster and this is not provided the unique values in
        the raster will be calculated.

    Returns
    -------
    dask.dataframe.DataFrame
        A delayed dask DataFrame. The columns are the values in `stats` plus a
        column indicating the band the calculation was carried out on. Each row
        is the set of statistical calculations carried out on data pulled from
        `data_raster` based on the corresponding feature in `features`. NaN
        values indicate where a feature was outside of the raster or all data
        under the feature was null.

    """
    if is_str(features) or isinstance(features, Vector):
        features = get_vector(features)
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
    if isinstance(features, Raster):
        if features.crs != data_raster.crs:
            raise ValueError("Feature raster CRS must match data raster")
        if features.shape[1:] != data_raster.shape[1:]:
            raise ValueError("Feature raster shape must match data raster")

    feature_labels = None
    features_raster = None
    if isinstance(features, Vector):
        feature_labels = np.arange(1, len(features) + 1)
        features_raster = features.to_raster(data_raster)
    else:
        if raster_feature_values is None:
            (raster_feature_values,) = dask.compute(np.unique(features.data))
        else:
            raster_feature_values = np.atleast_1d(raster_feature_values)
            raster_feature_values = raster_feature_values[
                raster_feature_values > 0
            ]
        feature_labels = raster_feature_values
        features_raster = features

    data = _build_zonal_stats_data(
        data_raster, features_raster, feature_labels, stats
    )
    df = _build_zonal_stats_dataframe(data)
    return df


def _xy_to_rowcol_wrapper(x, y, affine):
    return np.stack(xy_to_rowcol(x, y, affine), axis=0)


def _extract_points(data, r, c, valid_mask):
    r, c, valid_mask = dask.compute(r, c, valid_mask)
    extracted = np.full((len(valid_mask),), np.nan, dtype=F64)
    extracted[valid_mask] = data[r[valid_mask], c[valid_mask]]
    return extracted


def _build_zonal_stats_data_from_points(data, mask, x, y, affine):
    r, c = da.blockwise(
        _xy_to_rowcol_wrapper,
        "zi",
        x,
        "i",
        y,
        "i",
        affine=affine,
        new_axes={"z": 2},
        dtype=np.int64,
    )
    _, rn, cn = data.shape
    r, c = dask.compute(r, c)
    valid_mask = (r >= 0) & (r < rn) & (c >= 0) & (c < cn)
    out = {
        i + 1: {"extracted": da.full(len(valid_mask), np.nan, dtype=F64)}
        for i in range(data.shape[0])
    }
    for i in range(data.shape[0]):
        extracted = da.full(len(valid_mask), np.nan, dtype=F64)
        extracted[valid_mask] = data.vindex[i, r[valid_mask], c[valid_mask]]
        # Mask out missing points within the valid zones
        exmask = da.zeros(len(valid_mask), dtype=bool)
        exmask[valid_mask] = mask.vindex[i, r[valid_mask], c[valid_mask]]
        extracted[exmask] = np.nan
        out[i + 1]["extracted"] = extracted
    return out


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
            df = (
                da.full(n, bnd, dtype=np.min_scalar_type(nb + 1))
                .to_dask_dataframe(index=index)
                .to_frame("band")
            )
            df[column_name] = extracted.to_dask_dataframe(index=index)
        else:
            df = extracted.to_dask_dataframe(
                columns=column_name + "_" + str(bnd)
            )
        assert df.known_divisions
        dfs.append(df)
    df = dd.concat(dfs, axis=axis)
    return df
