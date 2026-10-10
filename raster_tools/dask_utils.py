import dask.array as da
import numpy as np


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


def dask_nanmin(x):
    """
    Retrieves the minimum value, ignoring nan values and handling empty blocks.
    """
    return da.reduction(
        x,
        _nanmin_empty_safe,
        _nanmin_empty_safe,
        axis=None,
        keepdims=False,
        dtype=x.dtype,
    )


def dask_nanmax(x):
    """
    Retrieves the maximum value, ignoring nan values and handling empty blocks.
    """
    return da.reduction(
        x,
        _nanmax_empty_safe,
        _nanmax_empty_safe,
        axis=None,
        keepdims=False,
        dtype=x.dtype,
    )


def chunks_to_array_locations(dim_chunks):
    """Return a list of range tuples for a dask dimension's chunks.

    These range tuples can be used to create a range object.
    """
    spans = []
    start = 0
    for c in dim_chunks:
        spans.append((start, start + c))
        start += c
    return spans


# np.pad modes that extend an array the way dask's named overlap boundaries
# do. Unlike dask, np.pad also handles a pad wider than the axis.
_BOUNDARY_TO_PAD_MODE = {
    "reflect": "symmetric",
    "periodic": "wrap",
    "nearest": "edge",
}


def _is_no_boundary(boundary):
    return boundary is None or (
        isinstance(boundary, str) and boundary.lower() == "none"
    )


def _max_axis_depth(depth, axis):
    d = depth.get(axis, 0)
    return max(d) if isinstance(d, tuple) else d


def _pad_block(block, pad_width, boundary):
    if isinstance(boundary, str):
        mode = _BOUNDARY_TO_PAD_MODE[boundary.lower()]
        return np.pad(block, pad_width, mode=mode)
    return np.pad(block, pad_width, constant_values=boundary)


def pad_short_axes(arrays, depth, boundaries):
    """Extend the axes that are shorter than the overlap depth.

    dask's overlap rejects a depth larger than the array along an axis. Each
    such axis is merged into a single chunk and extended by its depth on both
    sides, filled the way dask fills the boundary: with a constant, or by
    'reflect', 'periodic' or 'nearest' repeated as often as needed. An
    overlap taken with the original depth then gives every original cell
    the same neighborhood it would have near the edge of a larger array.
    With no boundary (``None`` or ``'none'``) nothing lies beyond the edge,
    so the axis is only merged into one chunk and its depth set to 0.

    Parameters
    ----------
    arrays : list of dask.array.Array
        Arrays with the same shape and chunks.
    depth : dict
        Maps an axis to an int or a ``(before, after)`` tuple.
    boundaries : list
        The dask boundary for each array: a scalar, ``None``, or one of
        'none', 'reflect', 'periodic' or 'nearest'.

    Returns
    -------
    arrays : list of dask.array.Array
        The extended arrays.
    depth : dict
        The depth to use with the extended arrays.
    unpad : callable
        Cuts a result computed on the extended arrays back to the original
        shape and chunks.

    """
    shape = arrays[0].shape
    chunks = arrays[0].chunks
    short = [
        ax
        for ax in range(len(shape))
        if _max_axis_depth(depth, ax) > shape[ax]
    ]
    if not short:
        return list(arrays), depth, lambda x: x

    no_boundary = [_is_no_boundary(b) for b in boundaries]
    if any(no_boundary) and not all(no_boundary):
        raise ValueError(
            "Cannot mix boundary None/'none' with other boundaries when the "
            "depth is larger than the array"
        )
    single = {ax: shape[ax] for ax in short}
    arrays = [a.rechunk(single) for a in arrays]
    depth = dict(depth)
    if all(no_boundary):
        for ax in short:
            depth[ax] = 0
        widths = {}
    else:
        widths = {ax: _max_axis_depth(depth, ax) for ax in short}
        pad_width = tuple((widths.get(ax, 0),) * 2 for ax in range(len(shape)))
        new_chunks = tuple(
            (c[0] + 2 * widths[ax],) if ax in widths else c
            for ax, c in enumerate(arrays[0].chunks)
        )
        arrays = [
            a.map_blocks(
                _pad_block,
                pad_width=pad_width,
                boundary=b,
                chunks=new_chunks,
                dtype=a.dtype,
                meta=np.array((), dtype=a.dtype),
            )
            for a, b in zip(arrays, boundaries, strict=True)
        ]

    def unpad(x):
        if widths:
            x = x[
                tuple(
                    slice(widths[ax], x.shape[ax] - widths[ax])
                    if ax in widths
                    else slice(None)
                    for ax in range(x.ndim)
                )
            ]
        restore = {
            ax: chunks[ax] for ax in short if x.chunks[ax] != chunks[ax]
        }
        return x.rechunk(restore) if restore else x

    return arrays, depth, unpad


def map_overlap_any_depth(func, *arrays, depth, boundary, **kwargs):
    """:func:`dask.array.overlap.map_overlap` allowing any depth.

    A depth larger than the arrays along an axis is handled by
    :func:`pad_short_axes`. `depth` is a dict mapping an axis to an int or
    a ``(before, after)`` tuple and applies to every array. `boundary` is
    one boundary for all arrays or a list with one per array. Other
    keyword arguments go to dask.

    """
    if isinstance(boundary, list):
        boundaries = boundary
    else:
        boundaries = [boundary] * len(arrays)
    arrays, depth, unpad = pad_short_axes(arrays, depth, boundaries)
    out = da.overlap.map_overlap(
        func, *arrays, depth=depth, boundary=boundaries, **kwargs
    )
    return unpad(out)
