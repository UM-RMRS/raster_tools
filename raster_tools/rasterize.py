import threading
import weakref
from functools import partial

import dask.array as da
import geopandas as gpd
import numba as nb
import numpy as np
import rasterio as rio
import shapely
from affine import Affine
from dask.diagnostics import ProgressBar
from packaging import version
from rasterio.enums import MergeAlg
from rasterio.env import GDALVersion
from rasterio.features import rasterize as rio_rasterize

from raster_tools._rasterize_numba import (
    _cast_burn_values,
    _dispatch_burn,
    _NumbaUnsupported,
)
from raster_tools.dtypes import (
    F64,
    I8,
    I16,
    I64,
    U8,
    U16,
    U32,
    U64,
    is_float,
    is_int,
)
from raster_tools.masking import get_default_null_value
from raster_tools.raster import data_to_raster_like, get_raster
from raster_tools.utils import list_reshape_2d
from raster_tools.vector import get_vector

__all__ = ["rasterize"]


# Selects the rasterization backend used by _rio_rasterize_wrapper and
# _rio_mask. "numba" burns with the kernels in _rasterize_numba, which
# reproduce GDAL's output pixel for pixel without rasterio's per-feature
# GeoJSON conversion, and falls back to rasterio for inputs it does not
# support. "rasterio" burns everything through GDAL.
RASTERIZE_BACKEND = "numba"


_RIO_64BIT_INTS_SUPPORTED = GDALVersion.runtime().at_least("3.5") and (
    version.parse(rio.__version__) >= version.parse("1.3")
)


# rasterio converts every geometry to a GeoJSON dict of nested tuples of
# Python floats before burning, then builds an OGR copy of each one. Both
# copies of the whole shape list are alive at once, so the peak memory of a
# single call scales with the total coordinate count. Geometries are burned
# in batches whose coordinate totals stay near this budget. At roughly 100
# bytes per GeoJSON coordinate, 1e6 coordinates is on the order of 100 MB.
RASTERIZE_COORD_BUDGET = 1_000_000
# Fixed per-geometry cost, in coordinate equivalents, for the GeoJSON dict,
# its ring/part lists, and the (geometry, value) pair rasterio builds.
GEOM_COORD_OVERHEAD = 8


def _get_rio_dtype(dtype):
    if dtype == I8:
        return I16
    # GDAL >= 3.5 and Rasterio >= 1.3 support 64-bit (u)ints
    if dtype in (I64, U64) and not _RIO_64BIT_INTS_SUPPORTED:
        return F64
    return dtype


def _iter_geom_batches(geometry, budget=None):
    """Yield (start, stop) index pairs that split geometry into batches.

    Each batch's total coordinate count stays under budget plus the size of
    its first geometry. A geometry that is larger than budget on its own is
    always yielded as its own batch. Batches preserve the input order.
    """
    if budget is None:
        budget = RASTERIZE_COORD_BUDGET
    budget = max(int(budget), 1)
    n = len(geometry)
    if n == 0:
        return
    weights = shapely.get_num_coordinates(geometry).astype(np.int64)
    weights += GEOM_COORD_OVERHEAD
    csum = np.cumsum(weights)
    total = int(csum[-1])
    # Cut wherever the running total crosses a multiple of the budget.
    targets = np.arange(budget, total, budget)
    cuts = np.searchsorted(csum, targets, side="right")
    # Isolate oversized geometries so they never share a batch.
    big = np.flatnonzero(weights > budget)
    bounds = np.unique(np.concatenate(([0], cuts, big, big + 1, [n])))
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        yield int(start), int(stop)


# shapely geometry type id -> numba-kernel family code. Ids absent here are
# not burnable by the numba kernels (GeometryCollection and the like).
_ID_TO_FAMILY_CODE = {0: 2, 4: 2, 1: 1, 5: 1, 3: 0, 6: 0}


def _numba_burn_runs(ids):
    """Split a geometry type-id array into maximal consecutive burn runs.

    Returns a list of (start, stop, kind) tuples in input order. kind is
    "numba" for a run of one geometry family the numba kernels can burn and
    "rio" for a run of geometries they cannot (GeometryCollections and other
    unsupported types). A missing geometry (id < 0) does not start a new run;
    it stays in the surrounding run and is dropped by whichever backend burns
    it. Burning the runs into one array in this order reproduces GDAL's
    last-wins overlap behaviour.
    """
    n = ids.shape[0]
    cat = np.full(n, -2, dtype=np.int8)
    cat[ids < 0] = -1
    for gid, fam in _ID_TO_FAMILY_CODE.items():
        cat[ids == gid] = fam
    concrete = np.flatnonzero(cat != -1)
    if concrete.size == 0:
        # Only missing geometries; the numba path drops them all.
        return [(0, n, "numba")]
    ccat = cat[concrete]
    change = np.flatnonzero(ccat[1:] != ccat[:-1]) + 1
    starts = np.concatenate(([0], concrete[change]))
    stops = np.concatenate((starts[1:], [n]))
    first_cat = np.concatenate(([ccat[0]], ccat[change]))
    return [
        (int(s), int(e), "rio" if c == -2 else "numba")
        for s, e, c in zip(starts, stops, first_cat, strict=True)
    ]


def _drop_missing_in_run(type_ids, geometry, values, start, stop):
    """Slice one run and drop missing geometries, keeping values aligned.

    A missing geometry (negative type id, i.e. None) can share a numba run
    with real geometries. Removing it, together with its value, keeps the
    value array aligned with the geometry array and lets the run reach the
    kernels without depending on how to_ragged_array treats None. The removed
    slot burns nothing, matching a None the rasterio path would skip.
    """
    geom_run = geometry[start:stop]
    value_run = values[start:stop]
    keep = type_ids[start:stop] >= 0
    if not keep.all():
        geom_run = geom_run[keep]
        value_run = value_run[keep]
    return geom_run, value_run


def _rio_ready_values(values):
    rio_values_dtype = _get_rio_dtype(values.dtype)
    if rio_values_dtype != values.dtype:
        return values.astype(rio_values_dtype)
    return values


def _numba_runs_rasterize(
    shape, transform, geometry, values, out_dtype, fill, all_touched
):
    """Burn a chunk with the numba kernels, falling back per run to rasterio.

    Returns the burned array, or None to signal that the whole chunk should
    be burned by rasterio instead (a rotated affine, which the kernels cannot
    handle at all, or an unsupported geometry combined with a dtype rasterio
    must widen). Supported geometries are burned with the numba kernels and
    unsupported ones with rasterio, into the same array and in input order,
    so the result is identical to burning the whole chunk with GDAL.
    """
    if transform.b != 0.0 or transform.d != 0.0:
        return None
    geometry = np.asarray(geometry, dtype=object)
    values = np.asarray(values)
    type_ids = shapely.get_type_id(geometry)
    runs = _numba_burn_runs(type_ids)
    has_rio = any(kind == "rio" for _, _, kind in runs)
    if has_rio and _get_rio_dtype(out_dtype) != out_dtype:
        # Unsupported geometry needs rasterio, but out_dtype is one rasterio
        # cannot write; let the all-rasterio path widen and cast back.
        return None
    if fill == 0:
        out = np.zeros(shape, dtype=out_dtype)
    else:
        out = np.full(shape, fill, dtype=out_dtype)
    numba_values = _cast_burn_values(values, out_dtype)
    rio_values = None
    try:
        for start, stop, kind in runs:
            if kind == "numba":
                geom_run, value_run = _drop_missing_in_run(
                    type_ids, geometry, numba_values, start, stop
                )
                _dispatch_burn(
                    out, transform, geom_run, value_run, all_touched
                )
            else:
                if rio_values is None:
                    rio_values = _rio_ready_values(values)
                rio_rasterize(
                    zip(
                        geometry[start:stop],
                        rio_values[start:stop],
                        strict=True,
                    ),
                    out=out,
                    transform=transform,
                    all_touched=all_touched,
                    merge_alg=MergeAlg.replace,
                )
    except _NumbaUnsupported:
        return None
    return out


def _rio_rasterize_wrapper(
    shape, transform, geometry, values, out_dtype, fill, all_touched
):
    if RASTERIZE_BACKEND == "numba":
        out = _numba_runs_rasterize(
            shape, transform, geometry, values, out_dtype, fill, all_touched
        )
        if out is not None:
            return out
    rio_dtype = _get_rio_dtype(out_dtype)
    values_dtype = _get_rio_dtype(values.dtype)
    if values_dtype != values.dtype:
        values = values.astype(values_dtype)
    geometry = np.asarray(geometry)

    # Burn into one preallocated array. With out given, rasterio ignores
    # fill, dtype, and out_shape, so fill and dtype are applied here. Later
    # batches overwrite earlier ones the same way later shapes already do
    # within a single call, so the result matches an unbatched call.
    out = np.full(shape, fill, dtype=rio_dtype)
    for start, stop in _iter_geom_batches(geometry):
        rio_rasterize(
            zip(geometry[start:stop], values[start:stop], strict=True),
            out=out,
            transform=transform,
            all_touched=all_touched,
            merge_alg=MergeAlg.replace,
        )

    if rio_dtype != out_dtype:
        out = out.astype(out_dtype)
    return out


def _numba_runs_mask(geoms, shape, transform, all_touched, invert):
    """Burn a mask chunk with the numba kernels, per-run rasterio fallback.

    Mirrors _numba_runs_rasterize for the mask path. Returns None only for a
    rotated affine, since the mask dtype is always uint8, which rasterio
    supports.
    """
    if transform.b != 0.0 or transform.d != 0.0:
        return None
    geoms = np.asarray(geoms, dtype=object)
    fill, geom_value = (1, 0) if invert else (0, 1)
    if fill == 0:
        out = np.zeros(shape, dtype=U8)
    else:
        out = np.full(shape, fill, dtype=U8)
    values = np.full(len(geoms), geom_value, dtype=U8)
    type_ids = shapely.get_type_id(geoms)
    try:
        for start, stop, kind in _numba_burn_runs(type_ids):
            if kind == "numba":
                geom_run, value_run = _drop_missing_in_run(
                    type_ids, geoms, values, start, stop
                )
                _dispatch_burn(
                    out, transform, geom_run, value_run, all_touched
                )
            else:
                rio_rasterize(
                    geoms[start:stop],
                    out=out,
                    transform=transform,
                    all_touched=all_touched,
                    default_value=geom_value,
                )
    except _NumbaUnsupported:
        return None
    return out


def _rio_mask(geoms, shape, transform, all_touched, invert):
    if RASTERIZE_BACKEND == "numba":
        out = _numba_runs_mask(geoms, shape, transform, all_touched, invert)
        if out is not None:
            return out
    fill, geom_value = (1, 0) if invert else (0, 1)
    geoms = np.asarray(geoms)
    out = np.full(shape, fill, dtype=U8)
    for start, stop in _iter_geom_batches(geoms):
        rio_rasterize(
            geoms[start:stop],
            out=out,
            transform=transform,
            all_touched=all_touched,
            default_value=geom_value,
        )
    return out


def _bounds_intersect(b, bounds):
    """Mark rows of a feature-bounds array that overlap the chunk bounds.

    b is an (n, 4) array of feature bounds (minx, miny, maxx, maxy). Missing
    and empty geometries have NaN bounds and fail every comparison, so they
    are dropped.
    """
    xmin, ymin, xmax, ymax = bounds
    return (
        (b[:, 0] <= xmax)
        & (b[:, 2] >= xmin)
        & (b[:, 1] <= ymax)
        & (b[:, 3] >= ymin)
    )


def _chunk_intersects_mask(geometry, bounds):
    """
    Return a boolean mask marking the geometries whose bounds overlap the
    chunk bounds. Missing and empty geometries have NaN bounds and fail
    every comparison, so they are dropped as well.
    """
    return _bounds_intersect(shapely.bounds(geometry), bounds)


# Per-partition cache of shapely.bounds over a partition's geometry, so the
# many chunk tasks that share one vector partition compute those bounds once
# instead of once per chunk. Keyed on id(partition) and guarded by a
# finalizer that drops the entry when the partition is collected, which also
# guarantees the id is never reused while an entry is live. The dask threaded
# scheduler runs chunk tasks concurrently, so the dict is lock-guarded and
# the cached bounds array is only ever read, never mutated.
_partition_bounds_cache = {}
_partition_bounds_lock = threading.Lock()


def _partition_bounds(gdf):
    key = id(gdf)
    with _partition_bounds_lock:
        cached = _partition_bounds_cache.get(key)
    if cached is not None:
        return cached
    computed = shapely.bounds(gdf.geometry.to_numpy())
    with _partition_bounds_lock:
        cached = _partition_bounds_cache.get(key)
        if cached is not None:
            return cached
        _partition_bounds_cache[key] = computed
        weakref.finalize(gdf, _partition_bounds_cache.pop, key, None)
    return computed


def _clip_polygons_to_chunk(geometry, bounds):
    """
    Clip polygons that extend past the chunk bounds. Rasterizing cost grows
    with the number of vertices handed to rasterio, so cutting polygons
    down to the chunk keeps large features cheap.

    Points and lines are never clipped. GDAL walks a line from its first
    vertex, so a clipped line can burn different cells than the original.
    A polygon whose clip collapses to nothing or to a lower dimension is
    passed through unchanged so that GDAL decides which cells it touches.
    The input array is not modified.
    """
    xmin, ymin, xmax, ymax = bounds
    b = shapely.bounds(geometry)
    inside = (
        (b[:, 0] >= xmin)
        & (b[:, 1] >= ymin)
        & (b[:, 2] <= xmax)
        & (b[:, 3] <= ymax)
    )
    to_clip = ~inside & (shapely.get_dimensions(geometry) == 2)
    if not to_clip.any():
        return geometry
    src = geometry[to_clip]
    clipped = shapely.clip_by_rect(src, xmin, ymin, xmax, ymax)
    degenerate = shapely.is_empty(clipped) | (
        shapely.get_dimensions(clipped) != 2
    )
    clipped[degenerate] = src[degenerate]
    geometry = geometry.copy()
    geometry[to_clip] = clipped
    return geometry


def _rasterize_onto_chunk(
    gdf,
    transform,
    out_dtype,
    fill,
    all_touched,
    overlap_resolve_method,
    block_info=None,
):
    """
    Rasterize a set of features onto a chunk. The cells that touch features
    receive the value of the feature given by the "values" column in gdf. If
    gdf doesn't have a "values" column, the index is used. Cells with
    overlapping features receive a value based on the sorting determined by
    overlap_resolve_method.
    """
    shape_2d = block_info[None]["chunk-shape"]
    use_index = "values" not in gdf
    if use_index and not gdf.index.is_unique:
        raise ValueError(
            "The dataframe index is used for feature values when no field is"
            " given, but it is not unique within a partition. Add a column of"
            " unique IDs with add_objectid_column and pass it as 'field'."
        )
    empty = np.zeros if fill == 0 else partial(np.full, fill_value=fill)
    bounds = rio.transform.array_bounds(*shape_2d, transform)
    keep = _bounds_intersect(_partition_bounds(gdf), bounds)
    if not keep.any():
        return empty(shape_2d, dtype=out_dtype)
    geometry = gdf.geometry.to_numpy()[keep]
    if use_index:
        values = gdf.index.to_numpy()[keep]
    else:
        values = gdf["values"].to_numpy()[keep]

    # Order the features so the desired 'winning' value burns last, since a
    # cell touched by several features keeps the value burned last. For 'max'
    # that is ascending order and for 'min' descending; order among equal
    # values does not matter because they burn the same value. NaN field
    # values go first so any valid value replaces them; a stable argsort
    # sends NaN to the end for ascending, so 'min' (a reversed ascending
    # argsort) already lands them first and 'max' rotates them to the front.
    if overlap_resolve_method == "first":
        order = slice(None, None, -1)
    elif overlap_resolve_method == "last":
        order = slice(None)
    elif overlap_resolve_method == "min":
        order = np.argsort(values, kind="stable")[::-1]
    else:
        # "max"
        order = np.argsort(values, kind="stable")
        n_nan = int(np.isnan(values).sum()) if is_float(values.dtype) else 0
        if n_nan:
            order = np.concatenate([order[-n_nan:], order[:-n_nan]])
    geometry = geometry[order]
    values = values[order]

    geometry = _clip_polygons_to_chunk(geometry, bounds)
    if use_index:
        # Burn the index plus one. Neither the add nor the cast range-checks:
        # numpy wraps a value too large for out_dtype, so correctness rests
        # on out_dtype being wide enough, which the divisions[-1] bound
        # guarantees, not on this cast.
        values = (values + 1).astype(out_dtype)

    return _rio_rasterize_wrapper(
        shape_2d, transform, geometry, values, out_dtype, fill, all_touched
    )


def _mask_onto_chunk(
    gdf, transform, all_touched=True, invert=False, block_info=None
):
    """
    Burn features onto a chunk. Cells that touch a feature get 1 and the rest
    get 0. This is flipped if invert is True.
    """
    fill = 1 if invert else 0
    shape_2d = block_info[None]["chunk-shape"]
    bounds = rio.transform.array_bounds(*shape_2d, transform)
    keep = _bounds_intersect(_partition_bounds(gdf), bounds)
    if not keep.any():
        empty = np.zeros if fill == 0 else partial(np.full, fill_value=fill)
        return empty(shape_2d, dtype="uint8")
    geometry = _clip_polygons_to_chunk(gdf.geometry.to_numpy()[keep], bounds)
    return _rio_mask(geometry, shape_2d, transform, all_touched, invert)


@nb.jit(nopython=True, nogil=True)
def _resolve_first(x, fill):
    # Take first valid values along band dimension
    out = np.where(x[-2] == fill, x[-1], x[-2])
    return np.expand_dims(out, 0)


@nb.jit(nopython=True, nogil=True)
def _resolve_last(x, fill):
    # Take last valid values along band dimension
    out = np.where(x[-1] == fill, x[-2], x[-1])
    return np.expand_dims(out, 0)


@nb.jit(nopython=True, nogil=True)
def _all_axis0(x):
    # Compute all along first axis
    out = np.empty((1, x.shape[1], x.shape[2]), dtype=np.bool_)
    nr, nc = x.shape[1:]
    for r in range(nr):
        for c in range(nc):
            out[0, r, c] = np.all(x[:, r, c])
    return out


@nb.jit(nopython=True, nogil=True)
def _min_axis0(x):
    # Compute min along first axis
    out = np.empty((1, x.shape[1], x.shape[2]), dtype=x.dtype)
    nr, nc = x.shape[1:]
    for r in range(nr):
        for c in range(nc):
            out[0, r, c] = np.min(x[:, r, c])
    return out


@nb.jit(nopython=True, nogil=True)
def _max_axis0(x):
    # Compute max along first axis
    out = np.empty((1, x.shape[1], x.shape[2]), dtype=x.dtype)
    nr, nc = x.shape[1:]
    for r in range(nr):
        for c in range(nc):
            out[0, r, c] = np.max(x[:, r, c])
    return out


@nb.jit(nopython=True, nogil=True)
def _resolve_min(x, fill):
    # Take min values along band dimension
    mask = x == fill
    out = np.where(mask, np.nanmax(x), x)
    out = _min_axis0(out)
    out = np.where(_all_axis0(mask), fill, out)
    return out


@nb.jit(nopython=True, nogil=True)
def _resolve_max(x, fill):
    # Take max values along band dimension
    mask = x == fill
    out = np.where(mask, np.nanmin(x), x)
    out = _max_axis0(out)
    out = np.where(_all_axis0(mask), fill, out)
    return out


_RESOLVE_KW_TO_FUNC = {
    "first": _resolve_first,
    "last": _resolve_last,
    "min": _resolve_min,
    "max": _resolve_max,
}


def _reduction_wrapper(
    x, fill, resolve_func, axis=None, keepdims=False, **kwargs
):
    if x.size == 0:
        # Do nothing for dask test calls. Ignore keepdims here because dask
        # gets confused otherwise.
        return x
    # Assuming x has dims (B, Y, X), where B in {1, 2}
    # Do nothing if input only has one band. Nothing to reduce. No copy is
    # needed: the resolvers below allocate fresh outputs via np.where, so this
    # reduction never mutates its input in place, and each stacked layer feeds
    # exactly one reduction, so its buffer is not shared.
    if x.shape[0] == 1:
        return x if keepdims else x[0]
    # A Python int fill makes np.where inside the resolvers promote small
    # unsigned arrays to int64, so give it the array's own dtype first.
    fill = x.dtype.type(fill)
    out = resolve_func(x, fill=fill)
    if keepdims:
        return out
    return out[0]


def _identity(x, *args, **kwargs):
    # The reduction's per-chunk step is a no-op. No copy is needed: the
    # resolvers allocate fresh outputs via np.where, so nothing mutates this
    # input in place, and each stacked layer feeds exactly one reduction.
    return x


def _reduce_stacked_feature_rasters_custom(
    stack, fill, overlap_resolve_method, keepdims
):
    agg_func = partial(
        _reduction_wrapper,
        fill=fill,
        resolve_func=_RESOLVE_KW_TO_FUNC[overlap_resolve_method],
    )
    reduced = da.reduction(
        stack,
        chunk=_identity,
        combine=agg_func,
        aggregate=agg_func,
        axis=0,
        keepdims=keepdims,
        dtype=stack.dtype,
        split_every=2,
    )
    return reduced


def _reduce_stacked_feature_rasters(
    chunk_stack,
    fill=None,
    overlap_resolve_method=None,
    mask=False,
    mask_invert=False,
    keepdims=False,
):
    if not mask:
        # Apply tailored reduction
        return _reduce_stacked_feature_rasters_custom(
            chunk_stack, fill, overlap_resolve_method, keepdims=keepdims
        )

    if mask_invert:
        # Features marked by 0. Use min to propagate 0s over 1s
        chunk = da.min(chunk_stack, axis=0, keepdims=keepdims)
    else:
        # Features marked by 1. Use max to propagate 1s over 0s
        chunk = da.max(chunk_stack, axis=0, keepdims=keepdims)
    return chunk


def _chunk_grid_specs(like):
    """Return one (shape, affine, box) triple per like-raster chunk.

    The list is in flattened row-major chunk order. Each shape is the
    chunk's (y, x) size, each affine is the chunk's transform, and each box
    is its bounding polygon. All three are derived arithmetically from the
    like raster's affine and chunk sizes; this matches the per-chunk affines
    of get_chunk_rasters() and the boxes of get_chunk_bounding_boxes()
    bit-for-bit without building an xarray object per chunk.
    """
    _, ychunks, xchunks = like.data.chunks
    base = like.affine
    specs = []
    row0 = 0
    for yc in ychunks:
        col0 = 0
        for xc in xchunks:
            affine = base * Affine.translation(col0, row0)
            shape_2d = (int(yc), int(xc))
            minx, miny, maxx, maxy = rio.transform.array_bounds(
                shape_2d[0], shape_2d[1], affine
            )
            specs.append(
                (shape_2d, affine, shapely.box(minx, miny, maxx, maxy))
            )
            col0 += xc
        row0 += yc
    return specs


def _compute_partition_chunk_matches(dgdf, like, specs=None):
    """Match vector partitions to the like-raster chunks they intersect.

    Returns a dataframe with one row per intersecting (partition, chunk)
    pair. part_idx indexes the vector partitions and flat_idx indexes the
    flattened grid of like-raster chunks. Rows are sorted so that
    partition order is preserved, which allows later partitions to take
    precedence over earlier partitions downstream. Chunk boxes are computed
    arithmetically (see _chunk_grid_specs); pass a precomputed specs list to
    reuse them.
    """
    if specs is None:
        specs = _chunk_grid_specs(like)
    sparts = dgdf.spatial_partitions.to_frame("geometry")
    sparts["part_idx"] = np.arange(dgdf.npartitions)
    chunk_gdf = gpd.GeoDataFrame(
        {"geometry": [s[2] for s in specs]}, crs=like.crs
    )
    chunk_gdf["flat_idx"] = chunk_gdf.index
    return sparts.sjoin(chunk_gdf).sort_values("part_idx")


def _rasterize_spatial_matches(
    matches,
    dgdf,
    chunk_specs,
    all_touched,
    fill=None,
    target_dtype=None,
    overlap_resolve_method=None,
    mask=False,
    mask_invert=False,
):
    # NOTE: Convert the partitions to delayed objects to work around some
    # flakey behavior in dask. The partitions are passed into map_blocks below
    # as args, which should work just fine. A very small and random percentage
    # of the time, however, this fails and one of the partitions will evaluate
    # to a pandas.Series object with a single element containing a dask graph
    # key(?). This started happening after dask-expr became the main dask
    # dataframe backend so I believe the issue originates there. Converting to
    # delayed objects avoids this behavior.
    partitions_as_delayed = dgdf.to_delayed()
    chunk_func = _mask_onto_chunk if mask else _rasterize_onto_chunk
    target_dtype = U8 if mask else target_dtype
    func_kwargs = {"all_touched": all_touched}
    if mask:
        func_kwargs["invert"] = mask_invert
    else:
        func_kwargs["out_dtype"] = target_dtype
        func_kwargs["fill"] = fill
        func_kwargs["overlap_resolve_method"] = overlap_resolve_method
    # Create a list for holding rasterization results. Each element corresponds
    # to a chunk in the like raster. All elements start as None. Elements will
    # will be replaced by a stack of dask arrays, if that chunk intersects a
    # vector partition. Each array is a vector partition that has been
    # rasterized to the corresponding like-chunk's grid.
    out_chunks = [None] * len(chunk_specs)
    # Group by partition and iterate over the groups
    for ipart, grp in matches.groupby("part_idx"):
        # Get the vector partition
        part = partitions_as_delayed[ipart]
        # Iterate over the chunks that intersected the vector partition and
        # rasterize the partition to each intersecting chunk's grid
        for _, row in grp.iterrows():
            chunk_shape, chunk_affine, _ = chunk_specs[row.flat_idx]
            func_kwargs["transform"] = chunk_affine
            chunk = da.map_blocks(
                chunk_func,
                part,
                dtype=target_dtype,
                chunks=chunk_shape,
                meta=np.array((), dtype=target_dtype),
                # func args
                **func_kwargs,
            )
            if out_chunks[row.flat_idx] is None:
                out_chunks[row.flat_idx] = []
            out_chunks[row.flat_idx].append(chunk)
    return out_chunks


def _raw_rasterized_chunks_to_dask_array(
    raw_chunk_list,
    chunk_specs,
    like_blocks_shape,
    fill,
    target_dtype=None,
    overlap_resolve_method=None,
    mask=False,
    mask_invert=False,
):
    processed_chunks = []
    for fi, oc in enumerate(raw_chunk_list):
        if oc is None:
            # Chunk did not intersect any partitions. Fill with the fill
            # value. da.zeros for a zero fill gets calloc's lazy pages, so an
            # untouched chunk costs nothing to allocate.
            chunk_shape = chunk_specs[fi][0]
            if fill == 0:
                processed_chunks.append(
                    da.zeros(
                        chunk_shape, dtype=target_dtype, chunks=chunk_shape
                    )
                )
            else:
                processed_chunks.append(
                    da.full(
                        chunk_shape,
                        fill,
                        dtype=target_dtype,
                        chunks=chunk_shape,
                    )
                )
        elif len(oc) == 1:
            # A single matched partition needs no reduction. oc[0] is the
            # fresh per-task map_blocks output of _rasterize_onto_chunk or
            # _mask_onto_chunk: 2D, already on the chunk's grid and in
            # target_dtype. It is a distinct array per task, so no defensive
            # copy is needed. Skipping the stack-and-reduce saves three graph
            # tasks per chunk.
            processed_chunks.append(oc[0])
        else:
            # The chunk intersected multiple partitions. Reduce the stack of
            # arrays (partitions rasterized to the chunk's grid) to a single
            # array using the specified overlap resolution method or by merging
            # masks together.
            chunk = _reduce_stacked_feature_rasters(
                da.stack(oc, axis=0),
                fill=fill,
                overlap_resolve_method=overlap_resolve_method,
                mask=mask,
                mask_invert=mask_invert,
                keepdims=False,
            )
            processed_chunks.append(chunk)
    # Stack back into 2D grid of chunks
    processed_chunks = list_reshape_2d(processed_chunks, like_blocks_shape)
    # Convert to Dask array made up of the processed chunks and add band dim of
    # size 1.
    return da.block([processed_chunks])


def _rasterize_spatial_aware(
    dgdf,
    like,
    field=None,
    fill=None,
    target_dtype=None,
    overlap_resolve_method=None,
    all_touched=True,
    mask=False,
    mask_invert=False,
):
    if dgdf.spatial_partitions is None:
        raise ValueError("No spatial partitions found on input dataframe.")

    if like.nbands > 1:
        # Only need one band
        like = like.get_bands(1)

    # One (shape, affine, box) per like chunk, derived arithmetically; the
    # burn never needs the like raster's data, only its grid.
    chunk_specs = _chunk_grid_specs(like)
    matches = _compute_partition_chunk_matches(dgdf, like, chunk_specs)
    # The null value can be different from the fill value when mask=True so set
    # a separate variable.
    nv = fill
    if mask:
        target_dtype = U8
        # The cells that will eventually be set to null values will always have
        # 0 in them after the rasterize operation when mask=True. final_nv,
        # below, will then be used to set the actual null values.
        nv = target_dtype.type(0)
        final_nv = None
        if fill is not None:
            # Store the original value to set later
            final_nv = fill
        fill = target_dtype.type(1 if mask_invert else 0)
        # Only need geometry for masking
        dgdf = dgdf.geometry.to_frame("geometry")
    else:
        # Transform to minimal dask dataframe.
        if field is not None:
            dgdf = dgdf[[field, "geometry"]].rename(columns={field: "values"})
        else:
            dgdf = dgdf.geometry.to_frame("geometry")
    # Each element is either None or a list of 2D dask arrays
    raw_chunk_list = _rasterize_spatial_matches(
        matches,
        dgdf,
        chunk_specs,
        all_touched,
        fill=fill,
        target_dtype=target_dtype,
        overlap_resolve_method=overlap_resolve_method,
        mask=mask,
        mask_invert=mask_invert,
    )
    # Replace None elements with array of fill and squash lists of 2D arrays to
    # single chunk.
    out_data = _raw_rasterized_chunks_to_dask_array(
        raw_chunk_list,
        chunk_specs,
        like.data.blocks.shape[1:],
        fill=fill,
        target_dtype=target_dtype,
        overlap_resolve_method=overlap_resolve_method,
        mask=mask,
        mask_invert=mask_invert,
    )
    raster = data_to_raster_like(out_data, like, nv=nv)
    if mask and final_nv is not None:
        raster = raster.set_null_value(final_nv)
    return raster


def _rasterize_spatial_naive(
    df,
    like,
    field=None,
    fill=None,
    target_dtype=None,
    overlap_resolve_method="max",
    all_touched=True,
    mask=False,
    mask_invert=False,
):
    # There is no way to neatly align the dataframe partitions with the raster
    # chunks, if spatial bounds are not known. Because of this, we burn in each
    # partition on its own like-sized raster and then merge the results. We do
    # this by creating mock spatial partitions where each partition covers all
    # of like. The spatial aware code can then handle it from here.
    nparts = df.npartitions
    like_bbox = shapely.geometry.box(*like.bounds)
    sparts = gpd.GeoSeries([like_bbox for i in range(nparts)], crs=like.crs)
    df.spatial_partitions = sparts
    return _rasterize_spatial_aware(
        df,
        like,
        field=field,
        fill=fill,
        target_dtype=target_dtype,
        overlap_resolve_method=overlap_resolve_method,
        all_touched=all_touched,
        mask=mask,
        mask_invert=mask_invert,
    )


def _resolve_index_value_dtype(gdf, null_value):
    """Pick the smallest unsigned dtype for the no-field index values.

    When no field is given, each feature is burned as its index label plus
    one. Return the smallest unsigned dtype (uint8/uint16/uint32) wide
    enough to hold every such value and an explicit null_value, if one was
    given. The largest possible index-plus-one value is ``divisions[-1] +
    1``; divisions are sorted at construction, so this is never an
    underestimate of the true maximum label.

    Fall back to I64 -- the prior fixed behavior -- when the index is not an
    integer type, the index has a negative label (which would wrap around in
    an unsigned dtype), or null_value is not a non-negative integer. A
    negative index label colliding with the fill value is a pre-existing
    edge case this does not change.
    """
    if not is_int(gdf.index.dtype):
        return I64
    lo, hi = gdf.divisions[0], gdf.divisions[-1]
    if lo is None or hi is None or lo < 0:
        return I64
    needed = int(hi) + 1
    if null_value is not None:
        if not is_int(null_value) or null_value < 0:
            return I64
        needed = max(needed, int(null_value))
    for dtype, bits in ((U8, 8), (U16, 16), (U32, 32)):
        if needed < 2**bits:
            return dtype
    return I64


def rasterize(
    features,
    like,
    field=None,
    overlap_resolve_method="last",
    mask=False,
    mask_invert=False,
    null_value=None,
    all_touched=True,
    use_spatial_aware=False,
    show_progress=False,
):
    """Convert vector feature data to a raster.

    This function can be used to either rasterize features using values from a
    particular data field or to create a raster mask of zeros and ones. Using
    values to rasterize is the default. Use `mask=True` to generate a raster
    mask. If no data field is specified, the underlying dataframe's index plus
    one is used, burned into the smallest unsigned dtype that holds every
    index-plus-one value and the null value (see `field` below). Vectors
    opened from a file carry a global, contiguous integer index across all
    partitions, so each feature receives a unique value. Cells that do not
    touch or overlap any features are marked as null.

    .. note::
        :func:`raster_tools.zonal.zonal_stats` rasterizes its zone features
        with no field, so the zone ids it reports inherit the dtype rule
        described under `field` rather than always being int64.

    A dask dataframe supplied directly must have a unique index for the
    resulting values to identify features. Duplicate index values within a
    partition raise a ``ValueError`` when the result is computed. To add a
    column of unique IDs for each feature, see
    :func:`raster_tools.vector.add_objectid_column` or
    :meth:`raster_tools.vector.Vector.add_objectid_column` and pass the new
    column as `field`.

    This operation can be greatly accelerated if the provided `features`
    object has been spatially shuffled or had spatial partitions calculated.
    There are a few ways to do this. For `Vector` or `GeoDataFrame`/`GeoSeries`
    objects, you can use the `spatial_shuffle` or
    `calculate_spatial_partitions` methods. `calculate_spatial_partitions`
    simply computes the spatial bounds of each partition in the data.
    `spatial_shuffle` shuffles the data into partitions of spatially near
    groups and calculates the spatial bounds at the same time. This second
    method is more expensive but provides a potentially greater speed up for
    rasterization. The `use_spatial_aware` flag can also be provided to this
    function. This causes the spatial partitions to be calculated before
    rasterization.

    .. note::
        If the CRS for `features` does not match the CRS for `like`, `features`
        will be transformed to `like`'s CRS. This operation causes spatial
        partition information to be lost. It is recommended that the CRSs for
        both are matched ahead of time.

    Parameters
    ----------
    features : Vector, GeoDataFrame, dask_geopandas.GeoDataFrame
        Vector data to rasterize.
    like : Raster
        A raster to use for grid and CRS information. The resulting raster will
        be on the same grid as `like`.
    field : str, optional
        The name of a field to use for cell values when rasterizing the
        vector features. If None or not specified, the underlying dataframe's
        index plus 1 is used. The default is to use the index plus 1. Without
        a field, the result uses the smallest unsigned dtype (uint8, uint16,
        or uint32) that holds every index-plus-one value and the null value,
        falling back to int64 if the index is not an integer type, contains a
        negative label, or the null value is negative or non-integer. When
        `field` is given, the result uses that field's dtype.
    overlap_resolve_method : str, optional
        The method used to resolve overlaping features. Default is `"last"`.
        The available methods are:

        'first'
            Cells with overlapping features will receive the value from the
            feature that appears first in the feature table.
        'last'
            Cells with overlapping features will receive the value from the
            feature that appears last in the feature table.
        'min'
            Cells with overlap will receive the value from the feature with the
            smallest value.
        'max'
            Cells with overlap will receive the value from the feature with the
            largest value.

    mask : bool, optional
        If ``True``, the features are rasterized as a mask. Cells that do not
        touch a feature are masked out. Cells that touch the features are set
        to ``1``. If `mask_invert` is also ``True``, this is inverted. If
        `mask` is ``False``, the features are rasterized using `field` to
        retrieve values from the underlying dataframe. `field` is ignored, if
        this option is a used. Default is ``False``.
    mask_invert : bool, optional
        If ``True`` cells that are inside or touch a feature are masked out. If
        ``False``, cells that do not touch a feature are masked out. Default is
        ``False``.
    null_value : scalar, optional
        The value to use in cells with no feature data, when not masking.
    all_touched : bool, optional
        If ``True``, grid cells that the vector touches will be burned in.
        If False, only cells with a center point inside of the vector
        perimeter will be burned in.
    use_spatial_aware : bool, optional
        Force the use of spatial aware rasterization. If ``True`` and
        `features` is not already spatially indexed, a spatial index will be
        computed. Alternatively, if ``True`` and `features`'s CRS differs from
        `like`, a new spatial index in a common CRS will be computed. If
        `features` already has a spatial index and its CRS matches `like`, this
        argument is ignored. Default is ``False``.
    show_progress : bool, optional
        If `use_spatial_aware` is ``True``, this flag causes a progress bar to
        be displayed for spatial indexing. Default is ``False``.

    Returns
    -------
    Raster
        The resulting single band raster of rasterized features. This raster
        will be on the same grid as `like`.

    """
    gdf = get_vector(features).data

    like = get_raster(like)
    if not mask:
        if null_value is not None and not (
            is_int(null_value) or is_float(null_value)
        ):
            raise TypeError("null_value must be a scalar")

        if isinstance(field, str):
            if field not in gdf:
                raise ValueError(f"Invalid field name: {repr(field)}")
            dtype = gdf[field].dtype
            if not is_int(dtype) and not is_float(dtype):
                raise ValueError(
                    "The specified field must be a scalar data type"
                )
            target_dtype = dtype
            if null_value is None:
                null_value = get_default_null_value(target_dtype)
        elif field is not None:
            raise ValueError(f"Could not understand 'field' value: {field!r}")
        else:
            target_dtype = _resolve_index_value_dtype(gdf, null_value)
            if null_value is None:
                null_value = 0

        if overlap_resolve_method not in {"first", "last", "min", "max"}:
            raise ValueError(
                "Invalid value for overlap_resolve_method: "
                f"{overlap_resolve_method!r}"
            )
    else:
        target_dtype = U8

    if gdf.crs != like.crs and like.crs is not None:
        # This will clear spatial_partitions
        gdf = gdf.to_crs(like.crs)

    if use_spatial_aware and gdf.spatial_partitions is None:
        gdf = gdf.copy()
        if show_progress:
            with ProgressBar():
                gdf.calculate_spatial_partitions()
        else:
            gdf.calculate_spatial_partitions()

    if gdf.spatial_partitions is not None:
        return _rasterize_spatial_aware(
            gdf,
            like,
            field,
            fill=null_value,
            target_dtype=target_dtype,
            all_touched=all_touched,
            overlap_resolve_method=overlap_resolve_method,
            mask=mask,
            mask_invert=mask_invert,
        )
    else:
        return _rasterize_spatial_naive(
            gdf,
            like=like,
            field=field,
            fill=null_value,
            target_dtype=target_dtype,
            overlap_resolve_method=overlap_resolve_method,
            all_touched=all_touched,
            mask=mask,
            mask_invert=mask_invert,
        )
