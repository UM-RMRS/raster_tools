from collections.abc import Iterable, Iterator, Sequence
from functools import partial

import dask
import dask.array as da
import numba as nb
import numpy as np

from raster_tools._types import F64, I64
from raster_tools._utils import is_str
from raster_tools.raster import RasterNoDataError
from raster_tools.rsv_utils import get_raster, get_vector

__all__ = ["ZONAL_STAT_FUNCS", "zonal_stats"]


def _handle_empty(func):
    def wrapped(x, axis=None, keepdims=False):
        if x.size > 0 or np.isnan(x.size):
            try:
                return func(x, axis=axis, keepdims=keepdims)
            except ValueError:
                pass
        return np.array([], dtype=x.dtype)

    return wrapped


# np.nan{min, max} both throw errors for empty chumks. dask.array.nan{min, max}
# handles empty chunks but requires that the chunk sizes be known at runtime.
# This safely handles empty chunks. There may still be corner cases that have
# not been found but for now it works.
_nanmin_empty_safe = _handle_empty(np.nanmin)
_nanmax_empty_safe = _handle_empty(np.nanmax)


def _nan_min(x):
    return da.reduction(
        x,
        _nanmin_empty_safe,
        _nanmin_empty_safe,
        axis=None,
        keepdims=False,
        dtype=x.dtype,
    )


def _nan_max(x):
    return da.reduction(
        x,
        _nanmax_empty_safe,
        _nanmax_empty_safe,
        axis=None,
        keepdims=False,
        dtype=x.dtype,
    )


def _nan_count(x):
    return da.count_nonzero(~np.isnan(x))


def _nan_median(x):
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
        return np.array()
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
    "max": _nan_max,
    "mean": da.nanmean,
    "median": _nan_median,
    "min": _nan_min,
    "mode": _nan_mode,
    "std": da.nanstd,
    "sum": da.nansum,
    "unique": _nan_unique,
    "var": da.nanvar,
}
# The set of valid zonal function names/keys
ZONAL_STAT_FUNCS = frozenset(_ZONAL_STAT_FUNCS)


def _get_zonal_data(vec, raster, all_touched=True):
    if raster.shape[0] > 1:
        raise ValueError("Only single band rasters are allowed")
    try:
        raster_clipped = raster.clip_box(
            # Use persist here to kick off computation of the bounds in the
            # background. The bounds have to be computed one way or another at
            # this point and persist allows it to start before the block occurs
            # in clip_box. If a large number of zonal calculations are being
            # carried out, this can provide a significant times savings.
            *dask.persist(vec.to_crs(raster.crs.wkt).bounds)[0]
        )
    except RasterNoDataError:
        return da.from_array([], dtype=raster_clipped.dtype)
    vec_mask = vec.to_raster(raster_clipped, all_touched=all_touched) > 0
    values = raster_clipped._rs.data[vec_mask._rs.data]
    # Filter null values
    if raster._masked:
        nv_mask = raster_clipped._mask[vec_mask._rs.data]
        values = values[~nv_mask]
    # Output is a 1-D dask array with unknown size
    return values


def zonal_stats(feature_vecs, data_raster, stats):
    """
    Apply stat functions to data taken from a raster based on a set of vectors.

    Parameters
    ----------
    feature_vecs : Vector, str
        A `Vector` or path string pointing to a vector file. The vector
        features are used like cookie cutters to pull data from the
        `data_raster`.
    data_raster : Raster, str
        A `Raster` or path string pointing to a raster file. The data raster
        to pull data from and apply the stat functions to.
    stats : str, list of str
        A single string or list of strings corresponding to stat funcstions.
        These functions will be applied to the raster data for each of the
        features in `feature_vecs`. Valid string values:

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

    Returns
    -------
    dask.dataframe.DataFrame
        A delayed dask DataFrame. The columns are the values in `stats` plus a
        column indicating the band the calculation was carried out on. Each row
        is the set of statistical calculations carried out on data pulled from
        `data_raster` based on the corresponding feature in `feature_vecs`. NaN
        values indicate where a feature was outside of the raster or all data
        under the feature was null.

    """
    feature_vecs = get_vector(feature_vecs)
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

    nbands = data_raster.shape[0]
    seriess = {
        "band": da.zeros(len(feature_vecs) * nbands, dtype=I64),
        **{f: da.zeros(len(feature_vecs) * nbands, dtype=F64) for f in stats},
    }
    i = 0
    for ibnd in range(nbands):
        ibnd += 1
        rs = data_raster.get_bands(ibnd)
        # This loop is the major timesink. The bounds of each sub-vector are
        # computed which takes time.
        # TODO: find way to delay bounds calculations
        zonal_data = [_get_zonal_data(vec, rs) for vec in feature_vecs]
        for zd in zonal_data:
            seriess["band"][i] = ibnd
            for f in stats:
                series = seriess[f]
                if zd.size == 0:
                    # vector feature was outside the raster bounds
                    series[i] = np.nan
                else:
                    series[i] = _ZONAL_STAT_FUNCS[f](zd)
            i += 1
    df = seriess["band"].to_dask_dataframe().to_frame("band")
    for k in stats:
        df[k] = seriess[k]
    if hasattr(feature_vecs.table, "npartitions"):
        df = df.repartition(npartitions=feature_vecs.table.npartitions)
    return df
