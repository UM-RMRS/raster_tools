import sys
from functools import partial

import dask.array as da
import geopandas as gpd
import numpy as np
import xarray as xr

from raster_tools.dtypes import F32, I8, is_bool, is_scalar, is_str
from raster_tools.raster import Raster, get_raster
from raster_tools.vector import _rasterize_block, get_vector

PY_38_PLUS = sys.version_info >= (3, 8)


def _trim(x, slices):
    return x[tuple(slices)]


def _build_cell_shapes(indices, xc, yc, radius, crs):
    if PY_38_PLUS:
        geom = gpd.GeoSeries.from_xy(xc[indices[1]], yc[indices[0]], crs=crs)
    else:
        geom = gpd.GeoSeries(
            gpd.points_from_xy(xc[indices[1]], yc[indices[0]], crs=crs)
        )
    return geom.buffer(radius)


def _compute_lengths(geometry, targets, weights=None):
    geom_df = geometry.to_frame()
    if weights is not None:
        geom_df["weight"] = weights
    targets_df = targets.to_frame()
    # Add column of index values so that clipped geoms can be mapped back to
    # the the target that clipped them after the overlay operation. Column
    # values are preserved in overlay.
    targets_df["target_idx"] = targets_df.index
    overlay = geom_df.overlay(targets_df)
    overlay["len"] = overlay.length
    lengths = np.zeros(len(targets), dtype=F32)
    if weights is None:
        # Unweighted sum
        for row in overlay.itertuples():
            lengths[row.target_idx] += row.len
    else:
        # Multiply lenths by weights taken from weight column
        for row in overlay.itertuples():
            lengths[row.target_idx] += row.len * row.weight
    return lengths


def _get_valid_slice(xc):
    left = None
    i = 0
    while np.isnan(xc[i]):
        left = i + 1
        i += 1
    right = None
    i = -1
    while np.isnan(xc[i]):
        right = i
        i -= 1
    return slice(left, right)


def _length_chunk(xc, yc, gdf, radius, field=None, block_info=None):
    xc = xc.ravel()
    yc = yc.ravel()
    # The overlap operation extends the raster into regions with no coords
    # given. The coords in these regions are set to nan. Find the nan coord
    # spans and trim them off.
    xslice = _get_valid_slice(xc)
    yslice = _get_valid_slice(yc)
    xc_valid = xc[xslice]
    yc_valid = yc[yslice]
    out_shape = block_info[None]["chunk-shape"]
    is_3d = len(out_shape) == 3
    if is_3d:
        out_shape = out_shape[1:]
    chunk_buffered_bounds = (
        xc_valid.min() - radius,
        yc_valid.min() - radius,
        xc_valid.max() + radius,
        yc_valid.max() + radius,
    )
    # Clip out any extraneous feature data
    gdf = gdf.clip(chunk_buffered_bounds)
    geometry = gdf.geometry
    # Use field as weights
    weight_values = gdf[field] if field is not None else None
    if field is not None and is_bool(weight_values.dtype):
        weight_values = weight_values.astype(I8)
    # Rasterize to create mask of cells-of-interest
    values = np.ones(len(geometry), dtype="int16")
    raster = _rasterize_block(
        xc_valid,
        yc_valid,
        geometry.buffer(radius),
        values,
        I8,
        0,
        True,
        block_info={None: {"chunk-shape": (yc_valid.size, xc_valid.size)}},
    )
    indices = raster.nonzero()
    buffered_cells = _build_cell_shapes(
        indices, xc_valid, yc_valid, radius, gdf.crs
    )
    lengths = _compute_lengths(geometry, buffered_cells, weight_values)

    output = np.zeros(out_shape, dtype=F32)
    output[yslice, xslice][indices] = lengths

    if is_3d:
        output = np.expand_dims(output, axis=0)
    return output


def length(features, like_rast, radius, weighting_field=None):
    """
    Calculate a raster where each cell is the net length of all features within
    a given radius.

    This function returns a raster where each cell contains the sum of the
    lengths of all features that fall within a radius of the cell's center. The
    features are clipped to the circular neighborhood before the length is
    computed. Optional weighting can be added with `weighting_field`.

    Parameters
    ----------
    features : Vector, str
        The line features to compute lengths from.
    like_rast : Raster, str
        A raster to use as a reference grid and CRS. The output raster will be
        on the same grid.
    radius : scalar
        The radius around each grid cell's center to use. Features that fall
        within the radius are clipped to the resulting circle and their
        resulting lengths are summed. Additional weighting of the sum can be
        done using `weighting_field`. This should be in the same units as
        `like_rast`'s CRS.
    weighting_field : str, optional
        If not ``None``, this selects a field in `features` to use as a
        weighting factor in the sum of lengths.

    Returns
    -------
    Raster
        The resulting raster where each cell contains the sum of all feature
        lengths in the circular neighborhood determined by `radius`.


    """
    features = get_vector(features)
    like_rast = get_raster(like_rast)

    if not is_scalar(radius):
        raise TypeError(f"radius must be a scalar. Got type: {type(radius)}.")
    if radius <= 0:
        raise ValueError(f"radius must be greater than zero: Got: {radius!r}.")
    if like_rast.crs is None:
        raise ValueError("like_rast must have a CRS set.")
    if weighting_field is not None:
        if not is_str(weighting_field):
            raise TypeError("weighting_field must be a string.")
        if weighting_field not in features.data:
            raise ValueError(
                "weighting_field must be a field name in features. "
                f"Got: {weighting_field!r}."
            )
        field_dtype = features.data[weighting_field].dtype
        if not (is_scalar(field_dtype) or is_bool(field_dtype)):
            raise TypeError(
                "The field specified by weighting_field must be a scalar type."
                f" Found: {field_dtype}."
            )

    features = features.to_crs(like_rast.crs)
    gdf = features.data
    out_chunks = ((1,), *like_rast.xrs.chunks[1:])

    xc, yc = like_rast.get_chunked_coords()
    xdepth, ydepth = np.ceil(radius / np.abs(like_rast.resolution)).astype(int)

    oxc = da.overlap.overlap(
        xc, depth={0: 0, 1: 0, 2: xdepth}, boundary=np.nan
    )
    oyc = da.overlap.overlap(
        yc, depth={0: 0, 1: ydepth, 2: 0}, boundary=np.nan
    )
    overlap_chunks = ((1,), oyc.chunks[1], oxc.chunks[2])
    rasters = []
    for part in gdf.partitions:
        data = da.map_blocks(
            _length_chunk,
            oxc,
            oyc,
            gdf=part,
            radius=radius,
            field=weighting_field,
            chunks=overlap_chunks,
            meta=np.array((), dtype=F32),
        )
        rasters.append(data[0])
    data = da.stack(rasters)
    # Trim off overlap regions
    trim_func = partial(
        _trim,
        slices=(slice(None), slice(ydepth, -ydepth), slice(xdepth, -xdepth)),
    )
    data = da.map_blocks(
        trim_func,
        data,
        chunks=((1,) * data.shape[0], *out_chunks[1:]),
        meta=np.array((), dtype=F32),
    )
    data = da.sum(data, axis=0, keepdims=True)
    xrs = xr.DataArray(
        data, dims=("band", "y", "x"), coords=([1], like_rast.y, like_rast.x)
    ).rio.write_crs(like_rast.crs)
    return Raster(xrs)
